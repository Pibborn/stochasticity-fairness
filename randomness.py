"""Read-me:
    This script is used to test the randomness of the predictions of a stochastic fairness model.
    Some stuff is hardcoded into the functions, like e.g. the model used.
    The most important part is the test_randomness function and maybe the test_loop function.
    """
import sys
import os
sys.path.append('../src')

from datetime import datetime
import pandas as pd
import numpy as np
from src.FairModels.BinaryMI import BinaryMI
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

from src.combined_project.pipelines.preprocess_data.nodes import split_into_X_y_S
from src.combined_project.pipelines.general_nested_cv_scikit_learn.nodes import get_hps

import platform
import psutil
import cpuinfo
import GPUtil

# print hardware information in a file
def print_hardware(path):
    with open(f"{path}/hardware.txt", 'a') as file:
        # General system information
        print("System:", platform.system(),file=file)
        print("Node Name:", platform.node(),file=file)
        print("Release:", platform.release(),file=file)
        print("Version:", platform.version(),file=file)
        print("Machine:", platform.machine(),file=file)
        print("Processor:", platform.processor(),file=file)
        print("Architecture:", platform.architecture(),file=file)

        # CPU information
        cpu_info = cpuinfo.get_cpu_info()
        print("CPU:",file=file)
        print(f"  Name: {cpu_info['brand_raw']}",file=file)
        print(f"  Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical",file=file)
        print(f"  Max Frequency: {psutil.cpu_freq().max} MHz",file=file)
        print(f"  Current Frequency: {psutil.cpu_freq().current} MHz",file=file)

        # Memory information
        memory_info = psutil.virtual_memory()
        print("Memory:",file=file)
        print(f"  Total: {memory_info.total / (1024 ** 3):.2f} GB",file=file)
        print(f"  Available: {memory_info.available / (1024 ** 3):.2f} GB",file=file)
        print(f"  Used: {memory_info.used / (1024 ** 3):.2f} GB",file=file)
        print(f"  Percentage: {memory_info.percent}%",file=file)

        # GPU information
        gpus = GPUtil.getGPUs()
        if gpus:
            print("GPU(s):",file=file)
            for gpu in gpus:
                print(f"  Name: {gpu.name}",file=file)
                print(f"  Load: {gpu.load * 100}%",file=file)
                print(f"  Free Memory: {gpu.memoryFree}MB",file=file)
                print(f"  Used Memory: {gpu.memoryUsed}MB",file=file)
                print(f"  Total Memory: {gpu.memoryTotal}MB",file=file)
                print(f"  Driver Version: {gpu.driver}",file=file)
                print(f"  Temperature: {gpu.temperature} °C",file=file)
        else:
            print("No GPU found.")

def entropy(p):
    if p == 0.0 or p == 1.0:
        return 0
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

def test_randomness(model, X_test, m=None, p=1):
    """Test the randomness of the prediction for a already trained model.

    Args:
        model (_type_): Trained model
        X_test (_type_): Testdata
        m (_type_, optional): Number of row to be tested. Defaults to whole testdata.
        p (int, optional): Number of predictions per datapoint. Defaults to 1.

    Raises:
        ValueError: Raises error if m is not in the correct range.

    Returns:
        _type_: Numpy array with the mean of the predictions for each datapoint.
    """
    
    # get m random samples from X_test
    if m is None:
        m = X_test.shape[0]
        samples = X_test 
    elif m > X_test.shape[0] or m <= 0:
        raise ValueError("Invalid value for m")
    else:
        random_rows = np.random.randint(0, X_test.shape[0], size=m)
        samples = X_test[random_rows,:]
        
    # repeat samples p times and make predictions
    rep_samples = np.repeat(samples, p, axis=0)
    preds = model.predict_proba(rep_samples)
    preds = np.array([1 if pr >=0.5 else 0 for pr in preds])
    preds = preds.reshape(m,p)
    return np.mean(preds, axis=1), samples.index

#main loop
def test_loop(X_train,X_test,y_train,y_test,S_train,S_test,hps,p,gammas,path):
    res = {}
    accuracies = []
    with open(f"{path}/results.txt", 'a') as file:
        for gamma in gammas: 
            hps["gamma"] = gamma

            # fit model
            model = BinaryMI(**hps)
            model.fit(X_train, y_train, S_train)

            # test accuracy
            y_pred = model.predict_proba(X_test)
            y_pred = np.array([1 if pr >=0.5 else 0 for pr in y_pred])
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
            print(f"Gamma: {gamma} | Accuracy: {acc}",file=file)

            # save results from randomness test
            res[f"{gamma}"], index = test_randomness(model, X_test, p=p)

            # save results to csv
            total_results = pd.DataFrame({"index": index,"mean_prediction":res[f"{gamma}"],"S":S_test.flatten()})
            total_results.to_csv(f"{path}/results_{gamma}.csv",index=False)
        # plot the accuracy
        plot_acc(accuracies,gammas,path)
    return res

# preprocessing data just like in our framework
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

    X_train = X_train#.to_numpy()
    X_test = X_test#.to_numpy()
    
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

    X_train = X_train#.to_numpy()
    X_test = X_test#.to_numpy()
    
    return X_train, X_test, y_train, y_test, S_train, S_test

def prepare_data_adult():
    data = pd.read_csv('../data/01_raw/adult.data',sep=',')
    dataset = {"data": "adult_data"
      ,"label": 'income'
      ,"sep": ','
      ,"pos_label": ' <=50K'
      ,"neg_label": ' >50K'
      ,"privileged_sens_attr": ' Male'
      ,"unprivileged_sens_attr": ' Female'
      ,"filter": False
      ,"sensitive_attributes":['sex']
      ,"discard_features": [None]
      ,"categorial_features":['workclass','education','marital-status','occupation','relationship','native-country','race']
      ,"continuous_features":['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']}

    X, y, S, name, sens_attrs, cont, label = split_into_X_y_S(data,dataset)
    
    y = y.to_numpy()
    S = S.to_numpy()
    
    X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, train_size=0.666)

    scaler = StandardScaler()
    X_train[cont] = scaler.fit_transform(X_train[cont])
    X_test[cont] = scaler.transform(X_test[cont])

    X_train = X_train#.to_numpy()
    X_test = X_test#.to_numpy()
    
    return X_train, X_test, y_train, y_test, S_train, S_test

# you can ignore this
def hp_optimizing(X_train,y_train,S_train, path, random_seed, gamma):
    
    n_holdouts = 3
    n_hp = 100
    
    
    batch_size = 256
    set_quantized_position = True

    def objective(trial):
        print(f"Trial: {trial.number}")
        
        num_hidden_layers = trial.suggest_int('num_hidden_layers', 2, 5)
        size_hidden_layers = trial.suggest_int('size_hidden_layers',5, 30)
        kernel_regularizer = trial.suggest_categorical('kernel_regularizer', [0,0.01,0.001])
        drop_out = trial.suggest_categorical('drop_out', [0,0.2,0.5])
        epoch = trial.suggest_int('epoch', 10, 50)
        
        model = BinaryMI(num_hidden_layers=num_hidden_layers, size_hidden_layers=size_hidden_layers, kernel_regularizer=kernel_regularizer, 
                         drop_out=drop_out, set_quantized_position=set_quantized_position,batch_size=batch_size,epoch=epoch,gamma=gamma)
        
        skf = StratifiedKFold(n_splits=n_holdouts, random_state=random_seed, shuffle=True)
        acc = []
        for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
            model.fit(X_train[train_index], y_train[train_index], S_train[train_index])
            y_pred = model.predict_proba(X_train[test_index])
            y_pred = np.array([1 if pr >=0.5 else 0 for pr in y_pred])
            cur_acc = accuracy_score(y_train[test_index], y_pred)
            acc.append(cur_acc)
            
        return np.mean(acc)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_hp)
    
    return BinaryMI(batch_size = 256, set_quantized_position = True, **study.best_params)
    
# plotting the entropy histograms
def make_plots(results,S,path,gammas):
    fig, ax = plt.subplots(len(gammas),3, figsize=(20,40))
    plt.subplots_adjust(hspace=0.5)
    
    for i,gamma in enumerate(gammas):
        res = results[f"{gamma}"]
        # calculate entropy of predictions for each datapoint
        entr_res = np.array([entropy(p) for p in res])

        S = S.flatten()

        # Split into privileged and unprivileged group
        res_priv = entr_res[np.where(S == 1)]
        res_unpriv = entr_res[np.where(S == 0)]
        
        ax[i,0].hist(entr_res, bins=20, range=(0,1), color='skyblue', edgecolor='black', alpha=0.7)#,weights=weights)
        ax[i,0].set_title(f"Gamma = {gamma} |   All Groups")
        ax[i,0].set_xlabel('Entropy')
        ax[i,0].set_ylabel('Number')
        ax[i,0].grid(True, linestyle=':', alpha=0.6)
        ax[i,0].set_xlim(0,1)
        mean = np.mean(entr_res)
        std = np.std(entr_res)
        ax[i,0].axvline(mean, color='blue', linestyle='dashed', linewidth=2)
        ax[i,0].axvline(mean + std, color='grey', linestyle='dashed', linewidth=1)
        ax[i,0].axvline(mean - std, color='grey', linestyle='dashed', linewidth=1)

        ax[i,1].hist(res_priv, bins=20,range=(0,1), color='limegreen', edgecolor='black', alpha=0.7)#,weights=weights)
        ax[i,1].set_title('S=1')
        ax[i,1].set_xlabel('Entropy')
        ax[i,1].set_ylabel('Number')
        ax[i,1].grid(True, linestyle=':', alpha=0.6)
        ax[i,1].set_xlim(0,1)
        mean = np.mean(res_priv)
        std = np.std(res_priv)
        ax[i,1].axvline(mean, color='green', linestyle='dashed', linewidth=2)
        ax[i,1].axvline(mean + std, color='grey', linestyle='dashed', linewidth=1)
        ax[i,1].axvline(mean - std, color='grey', linestyle='dashed', linewidth=1)

        ax[i,2].hist(res_unpriv, bins=20,range=(0,1), color='coral', edgecolor='black', alpha=0.7)#,weights=weights)
        ax[i,2].set_title('S=0')
        ax[i,2].set_xlabel('Entropy')
        ax[i,2].set_ylabel('Number')
        ax[i,2].grid(True, linestyle=':', alpha=0.6)
        ax[i,2].set_xlim(0,1)
        mean = np.mean(res_unpriv)
        std = np.std(res_unpriv)
        ax[i,2].axvline(mean, color='red', linestyle='dashed', linewidth=2)
        ax[i,2].axvline(mean + std, color='grey', linestyle='dashed', linewidth=1)
        ax[i,2].axvline(mean - std, color='grey', linestyle='dashed', linewidth=1)

    plt.savefig(f"{path}/entropy_predictions.pdf")

#   making a Kolmogorov-Smirnov test and plot results  
def ks_tests(results,S,path,gammas):
    fig, ax = plt.subplots(1,1, figsize=(20,6))
    with open(f"{path}/results.txt", 'a') as file:
        kstest_results = []
        for gamma in gammas:
            res = results[f"{gamma}"]
            # calculate entropy of predictions for each datapoint
            entr_res = np.array([entropy(p) for p in res])

            S = S.flatten()

            # Split into privileged and unprivileged group
            res_priv = entr_res[np.where(S == 1)]
            res_unpriv = entr_res[np.where(S == 0)]
            
            ks_result = kstest(res_priv,res_unpriv)
            print(f"Gamma: {gamma} | KS-Test: {ks_result}",file=file)
            kstest_results.append(ks_result)
        
        # Making barplot for kstest
        values = [r.statistic for r in kstest_results]
        max_value = max(values)
        ax.bar(gammas,values,color='skyblue', edgecolor='black', alpha=0.7,width=0.1)
        ax.set_ylim(0,max_value*1.2)
        for i, value in enumerate(values):
            ax.text(gammas[i], value + 0.1*max_value, f"{kstest_results[i].pvalue:.3f}", ha='center', va='bottom')
        stars_legend = [Line2D([0], [0], color='w', marker='', markerfacecolor='black', markersize=10, label='p_value')]
        ax.legend(handles=stars_legend, loc='upper right', title='Legend')
        ax.set_title("KS-Test")
        ax.set_xlabel('Gamma')
        ax.set_ylabel('Statistic')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        plt.savefig(f"{path}/ks_test.pdf")

# simple barplot for accuracies
def plot_acc(accuracies,gammas,path):
    fig, ax = plt.subplots(1,1, figsize=(20,6))
    ax.bar(gammas,accuracies,color='skyblue', edgecolor='black', alpha=0.7,width=0.1)
    ax.set_ylim(0,1)
    ax.set_title("accuracies of trained models")
    ax.set_xlabel('Gamma')
    ax.set_ylabel('Acc')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"{path}/accuracies.pdf")
    
# running all this stuff 
# ignore the commented code 
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Discovering randomness of predictions of stochastic frl models.")
    
    #parser.add_argument('hppath', type=str, help='Path to folder where the best hyperparameters are stored.')
    #parser.add_argument('--hp',type=int, help='0-14', default=0)
    
    parser.add_argument('--seed',type=int, help='Seed for randomness',default=42)
    
    args = parser.parse_args()

    #hppath = args.hppath
    #outer = args.hp
    
    SEED = args.seed
    
    # set seed
    random.seed(SEED)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    path = f"randomness_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
    os.makedirs(path, exist_ok=True)
    
    print_hardware(path)
    
    gammas = np.linspace(0,1,11)
    p = 1000
       
    #hps = np.load(f"{hppath}/best_params_outer_{outer}.npy",allow_pickle=True).item()
    #del hps['mi_loss']
    #del hps['hybrid_layer']
    #del hps['hybrid']
    #hps["set_quantized_position"] = True
    hps = {'size_hidden_layers': 12,'num_hidden_layers': 3,'kernel_regularizer': 0,'gamma': 0,'epoch': 28,'drop_out': 0.0,'batch_size': 256, 'set_quantized_position': True}
    #hps = {'size_hidden_layers': 21, 'num_hidden_layers': 3, 'kernel_regularizer': 0, 'gamma': 0, 'epoch': 95, 'drop_out': 0.0, 'batch_size': 256, 'set_quantized_position': True}

    with open(f"{path}/results.txt", 'a') as file:
        print(hps,file=file)
        print(SEED,file=file)

    X_train, X_test, y_train, y_test, S_train, S_test = prepare_data_adult()
    res = test_loop(X_train,X_test,y_train,y_test,S_train,S_test,hps,p,gammas,path)
    make_plots(res,S_test,path,gammas)
    ks_tests(res,S_test,path,gammas)
    
    # hp tuning tests
    #X_train, X_test, y_train, y_test, S_train, S_test = prepare_data()
    #opt_model = hp_optimizing(X_train,y_train,S_train, "..", SEED, 0.5)
    #opt_model.fit(X_train, y_train, S_train)
    #y_pred = opt_model.predict_proba(X_test)
    #y_pred = np.array([1 if pr >=0.5 else 0 for pr in y_pred])
    #cur_acc = accuracy_score(y_test, y_pred)
    #print(cur_acc)

    
