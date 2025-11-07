import sys
import os
sys.path.append('../src')

from datetime import datetime
import pandas as pd
import numpy as np
from src.FairModels.BinaryMI import BinaryMI
from src.FairModels.DebiasClassifier import DebiasClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
from scipy.stats import kstest
import random
from matplotlib.lines import Line2D
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
import optuna
import json
from itertools import product

from src.combined_project.pipelines.preprocess_data.nodes import split_into_X_y_S
from src.combined_project.pipelines.general_nested_cv_scikit_learn.nodes import get_hps

import platform
import psutil
import cpuinfo
import GPUtil

def prepare_data_compas():
    data = pd.read_csv('../data/01_raw/compas-cls.data')
    dataset = {"data": "compas-cls_data"
      ,"label": 'two_year_recid'
      ,"sep": ','
      ,"pos_label": 1
      ,"neg_label": 0
      ,"privileged_sens_attr": 'African-American'
      ,"unprivileged_sens_attr": 'Caucasian'
      ,"filter": False
      ,"sensitive_attributes":['race']
      ,"discard_features": [None]
      ,"categorial_features":['sex','age_cat','c_charge_degree']
      ,"continuous_features":['priors_count']}

    X, y, S, name, sens_attrs, cont, label = split_into_X_y_S(data,dataset)
    
    y = y.to_numpy()
    S = S.to_numpy()
    
    X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, train_size=0.666)

    scaler = StandardScaler()
    X_train[cont] = scaler.fit_transform(X_train[cont])
    X_test[cont] = scaler.transform(X_test[cont])

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    return X_train, X_test, y_train, y_test, S_train, S_test

def prepare_data_banks():
    data = pd.read_csv('../data/01_raw/banks.data',sep=';')
    dataset = {"data": "banks_data"
      ,"label": 'y'
      ,"sep": ';'
      ,"pos_label": 'yes'
      ,"neg_label": 'no'
      ,"privileged_sens_attr": 'age > 25 and age < 60'
      ,"unprivileged_sens_attr": 'age <= 25 or age >= 60'
      ,"filter": True
      ,"sensitive_attributes":['age']
      ,"discard_features": [None]
      ,"categorial_features":['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
      ,"continuous_features":['duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']}

    X, y, S, name, sens_attrs, cont, label = split_into_X_y_S(data,dataset)
    
    y = y.to_numpy()
    S = S.to_numpy()
    
    X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, train_size=0.666)

    scaler = StandardScaler()
    X_train[cont] = scaler.fit_transform(X_train[cont])
    X_test[cont] = scaler.transform(X_test[cont])

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    return X_train, X_test, y_train, y_test, S_train, S_test

def get_accuracies(gamma, outer, hppath):
    with open(f"{hppath}/acc_gamma_{gamma}_results_outer_{outer}.json") as f_in:
        data = json.load(f_in)
    acc = data["ACC"]
    if acc < 0.5:
        acc = 1 - acc
    return acc

def test(model_name, data_name, model, get_data, hppath, path):
    X_train, X_test, y_train, y_test, S_train, S_test = get_data()
        
    with open(f"{path}/results_{model_name}_{data_name}.txt", 'a') as file:
        print(f"SEED: {SEED}",file=file)

        tested_accs = {outer:[] for outer in range(15)}
        prev_accs = {outer:[] for outer in range(15)}
        
        for outer in range(15):
            gammas = np.linspace(0,1,10)

            hps = np.load(f"{hppath}/best_params_outer_{outer}.npy",allow_pickle=True).item()
            if model_name == "BinaryMI":
                del hps['mi_loss']
                del hps['hybrid_layer']
                del hps['hybrid']
                hps["set_quantized_position"] = True
            
            print(f"hps:{hps}",file=file)
            
            prev_accs[outer] = [get_accuracies(gamma, outer, hppath) for gamma in gammas]

            accuracies = []
            for gamma in gammas:
                
                hps["gamma"] = gamma
                
                # to do: Same for other model
                model_instance = model(**hps)
                model_instance.fit(X_train, y_train, S_train)
                y_pred = model_instance.predict_proba(X_test)
                y_pred = np.array([1 if pr >=0.5 else 0 for pr in y_pred])
                cur_acc = accuracy_score(y_test, y_pred)
                
                print(f"Gamma: {gamma} | Accuracy: {cur_acc}",file=file)
                accuracies.append(cur_acc)
            tested_accs[outer] = accuracies
        
        t_accs = pd.DataFrame(tested_accs)
        p_accs = pd.DataFrame(prev_accs)
        t_accs.to_csv(f"{path}/tested_accs_{model_name}_{data_name}.csv")
        p_accs.to_csv(f"{path}/exp_accs_{model_name}_{data_name}.csv")
        
        plot(tested_accs, prev_accs, gammas, model_name, data_name, path)

def plot(tested_accs, prev_accs, gammas, model_name, data_name, path):
    fig, ax = plt.subplots(1,1, figsize=(20,6))
    for outer in range(15):
        ax.plot(gammas,tested_accs[outer],color='skyblue', alpha=0.5)
        ax.plot(gammas,prev_accs[outer],color='salmon', alpha=0.5)
    avg_tested_accs = np.mean(np.array(list(tested_accs.values())), axis=0)
    avg_prev_accs = np.mean(np.array(list(prev_accs.values())), axis=0)
    ax.plot(gammas,avg_tested_accs,label="Tested",color='blue',linewidth= 4)
    ax.plot(gammas,avg_prev_accs,label="Experiment",color='red',linewidth= 4)
        
    ax.set_title("Tested Accs vs Experiment Accs")
    ax.set_xlabel('Gamma')
    ax.set_ylabel('Acc')
    plt.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"{path}/accuracies_{model_name}_{data_name}.pdf") 
        

def test_accuracies_of_hp(SEED):
    
    hppaths = {"compas_BinaryMI": "../results/nested-cv-results/Nested-CV-Results-07-05/compas-cls_data_BinaryMI-2023-11-13-13:48:45.185995",
               "banks_BinaryMI": "../results/nested-cv-results/Nested-CV-Results-07-05/banks_data_BinaryMI-2023-11-21-16:27:59.101938",
               "compas_DebiasClassifier": "../results/nested-cv-results/Nested-CV-Results-07-05/compas-cls_data_DebiasClassifier-2023-11-01-20:56:11.306029",
               "banks_DebiasClassifier": "../results/nested-cv-results/Nested-CV-Results-07-05/banks_data_DebiasClassifier-2023-11-13-13:47:30.199295"}
    
    data = {"compas": prepare_data_compas,
            "banks": prepare_data_banks}
    
    models = {"BinaryMI": BinaryMI,
              "DebiasClassifier": DebiasClassifier}

    path = f"acc_test_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
    os.makedirs(path, exist_ok=True)
    
    for d,m in product(data,models):
        test(m, d, models[m], data[d], hppaths[f"{d}_{m}"], path)
    
    

if __name__ == "__main__":
    
    
    SEED = 69
    
    # set seed
    random.seed(SEED)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    test_accuracies_of_hp(SEED)