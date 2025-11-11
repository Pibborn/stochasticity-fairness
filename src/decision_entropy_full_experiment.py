import sys
import os

from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import demographic_parity_difference
from sklearn.tree import DecisionTreeClassifier
import argparse
import random
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import optuna

from load_data import DATALOADER

# Import models
from FairModels.ICVAE import ICVAE
from FairModels.BinaryMI import BinaryMI
from FairModels.VFAE import VFAE

from entropy_scores import entropy_scores

import hashlib

MODELS = {
    "BinaryMI": BinaryMI,
    "VFAE": VFAE,
    "ICVAE": ICVAE
}


HYPERPARAMS = {
    "BinaryMI": [('num_hidden_layers',"int",(2,5)),('size_hidden_layers',"int",(5,30)),("kernel_regularizer","categorical",[0,0.01,0.001]),
                 ("drop_out","categorical",[0.0,0.2,0.5]),("set_quantized_position","fixed",True),("run_eagerly","fixed",False),
                 ("batch_size","fixed",256),("epoch","int",(10,50))],
    "VFAE": [('num_hidden_layers',"int",(2,5)),("size_hidden_layers","int",(5,30)),("kernel_regularizer","categorical",[0,0.01,0.001]),
                 ("drop_out","categorical",[0.0,0.2,0.5]),("batch_size","fixed",256),("epoch","int",(10,50)),("dim_z","int",(5,30))],
    "ICVAE": [('num_hidden_layers',"int",(2,5)),('size_hidden_layers',"int",(5,30)),("kernel_regularizer","categorical",[0,0.01,0.001]),
                ("drop_out","categorical",[0.0,0.2,0.5]),("batch_size","fixed",256),("epoch","int",(10,50)),("dim_z","int",(5,30))]
}

def entropy(p):
    if p == 0.0 or p == 1.0:
        return 0
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

entropy_vec = np.vectorize(entropy)

def entropy_expgrad(model, X_test,m=None,p=1,batch_size=None):
    if m is None or m > X_test.shape[0]:
        m = X_test.shape[0]
        samples = X_test 
    elif m <= 0:
        raise ValueError("Invalid value for m")
    else:
        random_rows = np.random.choice(X_test.index, size=m,replace=False)
        samples = X_test.loc[random_rows]
  
    mean = np.zeros(m)

    for i in range(0,m,batch_size):
        cur_batch_size = min(batch_size,m-i)
        multiple_samples = np.repeat(samples.iloc[i:i+cur_batch_size].to_numpy(), p,axis=0)
        pred = model.predict(multiple_samples)
        pred_reshape = pred.reshape(cur_batch_size,p)

        mean[i:i+cur_batch_size] = np.mean(pred_reshape,axis=1)
    
    entropy_vec = np.vectorize(entropy)
    entropies = entropy_vec(mean)
    
    return entropies, mean ,samples.index

    

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

def entropy_for_all(cur_model, X_test,optimal_threshold,p,m=None,batch_size=64):
    if m is None or m > X_test.shape[0]:
        m = X_test.shape[0]
        samples = X_test 
    elif m <= 0:
        raise ValueError("Invalid value for m")
    else:
        random_rows = np.random.choice(X_test.index, size=m,replace=False)
        samples = X_test.loc[random_rows]
    
    entropies_1 = np.zeros(m)
    entropies_2 = np.zeros(m)
    entropies_3 = np.zeros((m,10))
    entropies_3_scaled = np.zeros((m,10))
    bins = np.zeros(m)

    scores = []
    for i in range(0,m,batch_size):
        cur_batch_size = min(batch_size,m-i)
        multiple_s = np.repeat(samples.iloc[i:i+cur_batch_size].to_numpy(), p,axis=0)

        pred = cur_model.predict_proba(multiple_s)
        pred_reshape = pred.reshape(cur_batch_size,p)
        scores.append(pred_reshape)

        # Entropy variant 1) 0.5 thresholds
        y_pred = (pred_reshape > 0.5).astype(int)
        mean_prediction = np.mean(y_pred,axis=1)
        entropies_1[i:i+cur_batch_size] = entropy_vec(mean_prediction)

        # Entropy variant 2) Optimal threshold
        y_pred = (pred_reshape > optimal_threshold).astype(int)
        mean_prediction = np.mean(y_pred,axis=1)
        entropies_2[i:i+cur_batch_size] = entropy_vec(mean_prediction)

        # Entropy variant 3) 
        for c in range(cur_batch_size):
            entropies_3[i+c,:], entropies_3_scaled[i+c,:],bins[i+c] = entropy_scores(pred_reshape[c,:],rng=np.random)

    every_score = np.vstack(scores)
    return entropies_1, entropies_2, entropies_3, entropies_3_scaled, bins ,samples.index, every_score

def test_loop(model,X_train,X_test,y_train,y_test,S_train,S_test,p,m,batch_size,gammas,path,hps={}, do_hp_opt=True):
    accuracies = pd.DataFrame(columns=["gamma","acc","dp","acc_threshold","dp_threshold","optimal_threshold"])
    accuracies.set_index("gamma", inplace=True)

    if do_hp_opt:
        hps.append(("gamma","fixed",0))
        _, best_hps = _hp_optimization(model, hps, X_train, y_train, S_train, 3, 3, SEED)
        cur_model = model(**best_hps)
        cur_model.fit(X_train, y_train, S_train)
        
    for gamma in gammas: 
        if do_hp_opt:
            best_hps["gamma"] = gamma
            print(best_hps)
            cur_model = model(**best_hps)
            cur_model.fit(X_train, y_train, S_train)
        else:
            # use default hyperparameters
            cur_model = model(gamma=gamma)
            cur_model.fit(X_train, y_train, S_train)

        # Get optimal threshold
        X_val, X_test_holdout, y_val, y_test_holdout, s_val, s_test_holdout = train_test_split(X_test, y_test, S_test, test_size=0.66, random_state=SEED)
        t = np.zeros(5)
        for i in range(5):
            y_prob = cur_model.predict_proba(X_val)
            fpr, tpr, thresholds = roc_curve(y_val, y_prob)
            youden_j = tpr - fpr
            optimal_idx = np.argmax(youden_j) 
            t[i] = thresholds[optimal_idx]
        t = t[~np.isinf(t)]
        optimal_threshold = t.mean()

        # Evaluate model and save
        y_prob = cur_model.predict_proba(X_test)
        y_pred = [1 if x > 0.5 else 0 for x in y_prob]
        acc = accuracy_score(y_test, y_pred)
        dp = demographic_parity_difference(y_test, y_pred, sensitive_features=S_test)

        y_pred = [1 if x > optimal_threshold else 0 for x in y_prob]
        acc_threshold = accuracy_score(y_test, y_pred)
        dp_threshold = demographic_parity_difference(y_test, y_pred, sensitive_features=S_test)

        accuracies.loc[gamma] = acc, dp, acc_threshold, dp_threshold ,optimal_threshold
        accuracies.to_csv(f"{path}/accuracies.csv")

        # Calculate decision entropies in 3 ways
        entropies_1, entropies_2, entropies_3, entropies_3_scaled, bins, index, every_score = entropy_for_all(cur_model,X_test,optimal_threshold,p=p,m=m,batch_size=batch_size)

        # save results to csv
        new_indices = [X_test.index.get_loc(idx) for idx in index] # Some workaround because S is a numpy array and X a pandas dataframe
        S = S_test.flatten()[new_indices] # If we use less samples then in the test dataset, we need to adjust S_test for further steps
        total_results = pd.DataFrame({"entropy_1":entropies_1, "entropy_2":entropies_2 ,"mean_entropy_3":np.mean(entropies_3,axis=1),"std_entropy_3":np.std(entropies_3,axis=1),"mean_entropy_3_scaled":np.mean(entropies_3_scaled,axis=1),"std_entropy_3_scaled":np.std(entropies_3_scaled,axis=1),"n_bins":bins})
        S_df = pd.DataFrame({"S":S})
        total_results = pd.concat([total_results,S_df],axis=1)
        total_results.set_index(index,inplace=True)
        total_results = pd.concat([total_results ,X_test], axis=1,join="inner")
        total_results.to_csv(f"{path}/results_{round(gamma,1)}.csv",index=False)
        #np.savetxt(f"{path}/scores_{round(gamma,1)}", every_score, delimiter=",")


def test_loop_expgrad(X_train,X_test,y_train,y_test,S_train,S_test,p,m,batch_size,epsilons,path):
    res = {}
    accuracies = pd.DataFrame(columns=["epsilon","acc","dp"])
    accuracies.set_index("epsilon", inplace=True)
        
    for epsilon in epsilons: 

        print(f"Epsilon: {epsilon}")
        cur_model = ExponentiatedGradient(DecisionTreeClassifier(), constraints=DemographicParity(difference_bound=epsilon))
        cur_model.fit(X_train, y_train, sensitive_features=S_train)
        
        y_pred = cur_model.predict(X_test)
        
        # Evaluate model and save
        acc = accuracy_score(y_test, y_pred)
        dp = demographic_parity_difference(y_test, y_pred, sensitive_features=S_test)
        accuracies.loc[epsilon] = acc, dp
        accuracies.to_csv(f"{path}/accuracies.csv")

        entropies, mean, index = entropy_expgrad(cur_model,X_test,p=p,m=m,batch_size=batch_size)

        # save results to csv
        new_indices = [X_test.index.get_loc(idx) for idx in index] # Some workaround because S is a numpy array and X a pandas dataframe
        S = S_test.flatten()[new_indices] # If we use less samples then in the test dataset, we need to adjust S_test for further steps
        total_results = pd.DataFrame({"entropy_1":entropies})
        S_df = pd.DataFrame({"S":S})
        total_results = pd.concat([total_results,S_df],axis=1)
        total_results.set_index(index,inplace=True)
        total_results = pd.concat([total_results ,X_test], axis=1,join="inner")
        total_results.to_csv(f"{path}/results_{round(epsilon,3)}.csv",index=False)


if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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

    X_train, X_test, y_train, y_test, S_train, S_test = DATALOADER[dataset](SEED)

    # Filter data to only contain one sensitive group
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

    if model_name=="ExpGrad":
        epsilons = np.linspace(0.01,0.1,11)
        test_loop_expgrad(X_train,X_test,y_train,y_test,S_train,S_test,p,m,batch_size,epsilons,path)
    
    else:
        model = MODELS[model_name]
        hps = HYPERPARAMS[model_name]

        gammas = np.linspace(0,1,11)
        test_loop(model,X_train,X_test,y_train,y_test,S_train,S_test,p,m,batch_size,gammas,path,hps,do_hp_opt=args.hp_opt)
