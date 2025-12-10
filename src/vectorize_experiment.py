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

from load_data import DATALOADER

def entropy(p):
    if p == 0.0 or p == 1.0:
        return 0
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

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
    entropies = []

    for i in range(0,m,batch_size):
        cur_batch_size = min(batch_size,m-i)
        multiple_samples = np.repeat(samples.iloc[i:i+cur_batch_size].to_numpy(), p,axis=0)
        pred = model.predict(multiple_samples)
        pred_reshape = pred.reshape(cur_batch_size,p)
        if i==0:
            datatype = pred_reshape.dtype

        mean[i:i+cur_batch_size] = np.mean(pred_reshape,axis=1)
    

    entropy_vec = np.vectorize(entropy)
    entropies.append(entropy_vec(mean))

    for dt in [np.float16, np.float32, np.float64, np.longdouble]:
        entropy_vec = np.vectorize(entropy,otypes=[dt])
        entropies.append(entropy_vec(mean))
    
    return entropies, mean ,samples.index, datatype

def test_loop_expgrad(X_train,X_test,y_train,y_test,S_train,S_test,p,m,batch_size,epsilons,path):
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

        entropies, mean, index, datatype = entropy_expgrad(cur_model,X_test,p=p,m=m,batch_size=batch_size)

        # save results to csv
        new_indices = [X_test.index.get_loc(idx) for idx in index] # Some workaround because S is a numpy array and X a pandas dataframe
        S = S_test.flatten()[new_indices] # If we use less samples then in the test dataset, we need to adjust S_test for further steps
        total_results = pd.DataFrame({"entropy_nodtype":entropies[0],"entropy_float16":entropies[1],"entropy_float32":entropies[2],"entropy_float64":entropies[3],"entropy_longdouble":entropies[4]})
        S_df = pd.DataFrame({"S":S})
        total_results = pd.concat([total_results,S_df],axis=1)
        total_results.set_index(index,inplace=True)
        total_results = pd.concat([total_results ,X_test], axis=1,join="inner")
        total_results.to_csv(f"{path}/results_{round(epsilon,3)}.csv",index=False)

        with open(f"{path}/datatype.txt","a") as file:
            file.write(datatype)


if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--path", type=str, default=os.getcwd())
    argparser.add_argument('--seed',type=int, help='Seed for randomness',default=42)
    argparser.add_argument('--p',type=int, help='Number of prediction per sample',default=1000)
    argparser.add_argument('--num_samples',type=int, help='Number of samples which undergo the test',default=None)
    argparser.add_argument('--batch_size',type=int, help='Batch size for prediction',default=64)
    argparser.add_argument('--hp-opt', action='store_true', help='Run an hp opt study for each gamma')
    argparser.add_argument("--dataset", type=str, required=True)
    
    args = argparser.parse_args()
    model_name = "ExpGrad"
    dataset = args.dataset
    path = args.path
    if not os.path.exists(path):
        raise ValueError("Path does not exist")
    p = args.p
    m = args.num_samples
    batch_size = args.batch_size
    
    path = f"{path}/vectorize_experiment{dataset}_{model_name}_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
    os.makedirs(path, exist_ok=True)
    
    SEED = args.seed
    # set seed
    random.seed(SEED)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    X_train, X_test, y_train, y_test, S_train, S_test = DATALOADER[dataset](SEED)

    epsilons = np.linspace(0.01,0.1,11)
    test_loop_expgrad(X_train,X_test,y_train,y_test,S_train,S_test,p,m,batch_size,epsilons,path)
    
