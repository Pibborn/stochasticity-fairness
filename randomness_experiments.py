import sys
import os
sys.path.append('../src')

from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import demographic_parity_difference
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import argparse
import random
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import optuna

from load_data import DATALOADER

from helpers import vfae_check

# Import models
from src.FairModels.ICVAE import ICVAE
from src.FairModels.BinaryMI import BinaryMI
from src.FairModels.VFAE import VFAE

#Import plots:
from plots import all_plots

def _hp_optimization(model, hps, X, y, S,n_trials, n_folds, random_seed, stratified=True):
    X = X.to_numpy()
    fixed_params = {}
    for hp in hps:
        if hp[1] == "fixed":
            fixed_params[hp[0]] = hp[2]
    
    def objective(trial):
        hp_trial_dict = {}
        
        for hp in hps:
            if hp[1] == "categorical":
                hp_trial_dict[hp[0]] = trial.suggest_categorical(hp[0], hp[2])
            elif hp[1] == "int":
                hp_trial_dict[hp[0]] = trial.suggest_int(hp[0], hp[2][0], hp[2][1])
            elif hp[1] == "float":
                hp_trial_dict[hp[0]] = trial.suggest_float(hp[0], hp[2][0], hp[2][1])
        
        cur_model = model(**hp_trial_dict,**fixed_params)
        
        if stratified:
            kf = StratifiedKFold(n_splits=n_folds, random_state=random_seed, shuffle=True)
        else:
            kf = KFold(n_splits=n_folds, random_state=random_seed, shuffle=True)
        auc = []
        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            cur_model.fit(X[train_index], y[train_index], S[train_index])
            y_pred = cur_model.predict_proba(X[test_index])
            y_pred = [1 if x > 0.5 else 0 for x in y_pred]
            cur_acc = roc_auc_score(y[test_index], y_pred)
            auc.append(cur_acc)
        return np.mean(auc)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_value ,{**study.best_params, **fixed_params}

def entropy(p):
    if p == 0.0 or p == 1.0:
        return 0
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

def mean_predictions(model, X_test, threshold=0.5, m=None, p=1,batch_size=None,expgrad=False):
    # get m random samples from X_test
    if m is None or m > X_test.shape[0]:
        m = X_test.shape[0]
        samples = X_test 
    elif m <= 0:
        raise ValueError("Invalid value for m")
    else:
        random_rows = np.random.choice(X_test.index, size=m,replace=False)
        samples = X_test.loc[random_rows]
        
        
    mean = np.zeros(m)
    std = np.zeros(m)
    mean_five = np.zeros(m)
    std_five = np.zeros(m)
    # repeat samples p times and make predictions
    
    #num_features = samples.iloc[0].to_numpy().shape[0]
    for i in range(0,m,batch_size):
        cur_batch_size = min(batch_size,m-i)
        #print(f"{i+1}-{i+cur_batch_size} of {m} samples")
        multiple_s = np.repeat(samples.iloc[i:i+cur_batch_size].to_numpy(), p,axis=0)
        if not expgrad:
            pred = model.predict_proba(multiple_s)
        else:
            pred = model._pmf_predict(multiple_s)[:,1]
        # optimal threshold
        pred = np.array([1 if x > threshold else 0 for x in pred])
        pred_reshape = pred.reshape(cur_batch_size,p)
        mean[i:i+cur_batch_size] = np.mean(pred_reshape,axis=1)
        std[i:i+cur_batch_size] = np.std(pred_reshape,axis=1)
        # 0.5 threshold
        pred_five = np.array([1 if x > 0.5 else 0 for x in pred])
        pred_five = pred_five.reshape(cur_batch_size, p)
        mean_five[i:i+cur_batch_size] = np.mean(pred_five, axis=1)
        mean_five[i:i+cur_batch_size] = np.std(pred_five, axis=1)

    return mean, std, mean_five, std_five, samples.index

def test_loop(model,X_train,X_test,y_train,y_test,S_train,S_test,p,m,batch_size,gammas,path,hps={},exp_grad=False,
              debug=False, do_hp_opt=True):
    res = {}
    accuracies = pd.DataFrame(columns=["gamma","acc_mean","acc_std", "dp_mean", "dp_std"])
    accuracies.set_index("gamma", inplace=True)
    accuracies_five = pd.DataFrame(columns=["gamma", "acc_mean", "acc_std", "dp_mean", "dp_std"])
    accuracies_five.set_index("gamma", inplace=True)
    hps.append(("",""))
    
    if exp_grad:
        gammas = np.linspace(0.01,0.1,11)
        
    for gamma in gammas: 
        hps = hps[:-1]
        hps.append(("gamma","fixed",gamma))
        print(f"Gamma: {gamma}")
        # get best hyperparameters
        if not exp_grad and do_hp_opt:
            _, best_hps = _hp_optimization(model, hps, X_train, y_train, S_train, 10, 3, SEED)
            cur_model = model(**best_hps)
            cur_model.fit(X_train, y_train, S_train)
        elif exp_grad:
            cur_model = model(DecisionTreeClassifier(), constraints=DemographicParity(difference_bound=gamma))
            cur_model.fit(X_train, y_train, sensitive_features=S_train)
        else:
            # use default hyperparameters
            cur_model = model()
            cur_model.fit(X_train, y_train, S_train)
        
        # Finding optimal threshold and accuracy of the model
        X_val, X_test_holdout, y_val, y_test_holdout, s_val, s_test_holdout = train_test_split(X_test, y_test, S_test, test_size=0.66, random_state=SEED)
        acc = np.zeros(5)
        dp = np.zeros(5)
        acc_five = np.zeros(5)
        dp_five = np.zeros(5)
        t = np.zeros(5)
        for i in range(5):
            # test accuracy
            if not exp_grad:
                y_prob = cur_model.predict_proba(X_val)
            else:
                y_prob = cur_model._pmf_predict(X_val)[:,1]
            # compute acc wrt optimal threshold
            fpr, tpr, thresholds = roc_curve(y_val, y_prob)
            youden_j = tpr - fpr
            optimal_idx = np.argmax(youden_j) 
            t[i] = thresholds[optimal_idx]
            y_pred = [1 if x > t[i] else 0 for x in y_prob]#(y_prob >= t[i]).astype(int)
            acc[i] = accuracy_score(y_val, y_pred)
            dp[i] = demographic_parity_difference(y_val, y_pred, sensitive_features=s_val)
            # compute acc wrt 0.5 threshold
            y_pred = [1 if x > 0.5 else 0 for x in y_prob]
            acc_five[i] = accuracy_score(y_val, y_pred)
            dp_five[i] = demographic_parity_difference(y_val, y_pred, sensitive_features=s_val)
        # Filter out inf
        t = t[~np.isinf(t)]
        np.savetxt(f"{path}/thresholds_{round(gamma,3)}.txt",t)
        accuracies.loc[gamma] = acc.mean(), acc.std(), dp.mean(), dp.std()
        accuracies.to_csv(f"{path}/accuracies.csv")
        accuracies_five.loc[gamma] = acc_five.mean(), acc_five.std(), dp_five.mean(), dp_five.std()
        accuracies.to_csv(f"{path}/accuracies_05.csv")
        optimal_threshold = t.mean()
        print(f"Accuracies, opt threshold: {acc}")
        # save results from randomness test
        print(f"Starting calculating the mean prediction for {p} samples each")
        res, std, res_five, std_five, index = mean_predictions(cur_model, X_test, threshold=optimal_threshold, p=p, m=m, batch_size=batch_size, expgrad=expgrad)
        if debug and not expgrad:
            vfae_check(X_test, cur_model, path, gamma)
        # save results to csv
        new_indices = [X_test.index.get_loc(idx) for idx in index] # Some workaround because S is a numpy array and X a pandas dataframe
        S = S_test.flatten()[new_indices] # If we use less samples then in the test dataset, we need to adjust S_test for further steps
        total_results = pd.DataFrame({"mean_prediction":res,"std_prediction": std,"entropy":[entropy(m_prd) for m_prd in res],"S":S,
                                      "mean_prediction_05": res_five, "std_prediction_05": std_five, "entropy_five":[entropy(m_prd) for m_prd in res_five]})
        total_results.set_index(index,inplace=True)
        total_results = pd.concat([total_results, X_test], axis=1,join="inner")
        if not exp_grad:
            total_results.to_csv(f"{path}/results_{round(gamma,1)}.csv",index=False)
        else:
            total_results.to_csv(f"{path}/results_{round(gamma,3)}.csv",index=False)

MODELS = {
    "BinaryMI": BinaryMI,
    "VFAE": VFAE,
    "ICVAE": ICVAE,
    "ExponentiatedGradient": ExponentiatedGradient,
}


HYPERPARAMS = {
    "BinaryMI": [('num_hidden_layers',"int",(2,5)),('size_hidden_layers',"int",(5,30)),("kernel_regularizer","categorical",[0,0.01,0.001]),
                 ("drop_out","categorical",[0.0,0.2,0.5]),("set_quantized_position","fixed",True),("run_eagerly","fixed",False),
                 ("batch_size","fixed",256),("epoch","fixed",10)],
    "VFAE": [('num_hidden_layers',"int",(2,5)),('size_hidden_layers',"int",(5,30)),("kernel_regularizer","categorical",[0,0.01,0.001]),
                 ("drop_out","categorical",[0.0,0.2,0.5]),("batch_size","fixed",256),("epoch","fixed",10),("dim_z","int",(5,30))],
    "ICVAE": [('num_hidden_layers',"int",(2,5)),('size_hidden_layers',"int",(5,30)),("kernel_regularizer","categorical",[0,0.01,0.001]),
                ("drop_out","categorical",[0.0,0.2,0.5]),("batch_size","fixed",256),("epoch","fixed",10),("dim_z","int",(5,30))],
    "ExponentiatedGradient": []
}

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True)
    argparser.add_argument("--dataset", type=str, required=True)
    argparser.add_argument("--path", type=str, default=os.getcwd())
    argparser.add_argument('--seed',type=int, help='Seed for randomness',default=42)
    argparser.add_argument('--p',type=int, help='Number of prediction per sample',default=1000)
    argparser.add_argument('--num_samples',type=int, help='Number of samples which undergo the test',default=None)
    argparser.add_argument('--batch_size',type=int, help='Batch size for prediction',default=64)
    argparser.add_argument('--hp-opt', action='store_true', help='Run an hp opt study for each gamma')
    argparser.add_argument("--s",type=int,default=2,help="Use 0 or 1 to only use one sensitive group")

    args = argparser.parse_args()
    model_name = args.model
    dataset = args.dataset
    path = args.path
    if not os.path.exists(path):
        raise ValueError("Path does not exist")
    p = args.p
    m = args.num_samples
    batch_size = args.batch_size
    if args.s==2:
        path = f"{path}/{dataset}_{model_name}_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
    else:
        path = f"{path}/{dataset}_{model_name}_S{args.s}_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
    os.makedirs(path, exist_ok=True)
    
    SEED = args.seed
    # set seed
    random.seed(SEED)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    
    # Special-case for ExponentiatedGradient
    expgrad = True if model_name == "ExponentiatedGradient" else False
    
    X_train, X_test, y_train, y_test, S_train, S_test = DATALOADER[dataset](SEED)

    if args.s == 0:
        S_train = S_train.flatten()
        X_train = X_train[S_train == 0]
        y_train = y_train[S_train == 0]
        S_train = S_train[S_train == 0]

        S_test = S_test.flatten()
        X_test = X_test[S_test == 0]
        y_test = y_test[S_test == 0]
        S_test = S_test[S_test == 0]

    elif args.s == 1:
        S_train = S_train.flatten()
        X_train = X_train[S_train == 1]
        y_train = y_train[S_train == 1]
        S_train = S_train[S_train == 1]

        S_test = S_test.flatten()
        X_test = X_test[S_test == 1]
        y_test = y_test[S_test == 1]
        S_test = S_test[S_test == 1]


    model = MODELS[model_name]
    hps = HYPERPARAMS[model_name]

    with open(f"{path}/results.txt", 'a') as file:  
        print(dataset,file=file)
        print(model_name,file=file)
        print(SEED,file=file)

    gammas = np.linspace(0,1,11)
    test_loop(model,X_train,X_test,y_train,y_test,S_train,S_test,p,m,batch_size,gammas,path,hps,expgrad,do_hp_opt=args.hp_opt)
    if s == 2:
        all_plots(path,expgrad)
        
     
