import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import argparse
from scipy.stats import kstest
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score

def test_performance(model,X,y):
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    mse = np.zeros(5)
    r_squared = np.zeros(5)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        model.fit(X.iloc[train_index],y.iloc[train_index])
        y_pred = model.predict(X.iloc[test_index])
        mse[i] = mean_squared_error(y.iloc[test_index],y_pred)
        r_squared[i] = r2_score(y.iloc[test_index], y_pred)
    return mse, r_squared

# get feature weights either by quantile or threshold, filter by S
def get_feature_weights(path,gamma,S=None):
    
    #get and preprocess data
    data = pd.read_csv(f"{path}/results_{gamma}.csv")
    
    if S is not None: 
        data = data[data["S"] == S]
    
    y = data["entropy"] 
    X = data.drop(columns=["mean_prediction","entropy"],inplace=False)
    
    # fit linreg
    linreg = LinearRegression()
    mse, r_squared = test_performance(linreg,X,y)
    linreg.fit(X,y)
    return pd.DataFrame({"feature":X.columns,"weight":linreg.coef_}), mse.mean(), r_squared.mean()

def entropy_infos(gamma,path,S=None):
    df = pd.read_csv(f"{path}/results_{gamma}.csv")
    if S is not None:
        df = df[df["S"] == S]
    mean = df["entropy"].mean()
    max = df["entropy"].max()
    min = df["entropy"].min()
    std = df["entropy"].std()
    return mean, std, max, min

def heatmap_feature_weights(path,features=None,S=None,fraction=False,suffix="",expgrad=False):
    if not expgrad:
        gammas = np.linspace(0,1,11)
    else:
        gammas = np.linspace(0.01,0.1,11)
    #fig, ax = plt.subplots(2,1,layout="constrained",figsize=(10,30))
    # Create a figure
    fig = plt.figure(figsize=(10,30))

    gs = gridspec.GridSpec(6, 1)  
    ax1 = fig.add_subplot(gs[0:5, 0])  
    ax2 = fig.add_subplot(gs[5, 0]) 
    ax = [ax1,ax2]
    
    df, _, _ = get_feature_weights(path,0.1,S)
    data = np.zeros((df.shape[0],len(gammas)))
    ent_mean = np.zeros(len(gammas))
    ent_std = np.zeros(len(gammas))
    ent_max = np.zeros(len(gammas))
    ent_min = np.zeros(len(gammas))
    mse = np.zeros(len(gammas))
    r_squared = np.zeros(len(gammas))
    for i,gamma in enumerate(gammas):
        if not expgrad:
            gamma = round(gamma,1)
        else:
            gamma = round(gamma,3)
        df, mse[i], r_squared[i] = get_feature_weights(path,gamma,S)
        data[:,i] = df["weight"]
        ent_mean[i], ent_std[i], ent_max[i], ent_min[i] = entropy_infos(gamma,path,S)
        if fraction:
            data[:,i] = data[:,i]/(ent_max[i]-ent_min[i])
    if features is not None:
        ind = df["feature"][df["feature"].isin(features)].index
        data = data[ind,:]
    else:
        ind = df["feature"].index
    im = ax[0].imshow(data)
    ax[0].set_xticks(range(len(gammas)),labels=[round(gamma,1) for gamma in gammas])
    ax[0].set_yticks(range(len(ind)),df["feature"][ind])
    ax[0].set_title("Feature coefficients for predicting the entropy with LinReg")
    ax[0].set_xlabel("Gamma")
    
    for i in range(len(gammas)):
        for j in range(data.shape[0]):
            if data[j,i] != 0.0:
                text = ax[0].text(i, j, round(data[j,i],2),ha="center", va="center", color="w",
                               fontsize=5,alpha=0.6)
    
    #fig.tight_layout()
    ax[0].set_aspect(aspect=0.5)
    plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    
    if not expgrad:
        infos = pd.DataFrame({"Gamma":[round(gamma,1) for gamma in gammas],"MSE":mse,"r2":r_squared,"Entropy: Mean":ent_mean,"Std":ent_std,"Max":ent_max,"Min":ent_min})
    else:
        infos = pd.DataFrame({"Eps":[round(gamma,3) for gamma in gammas],"MSE":mse,"r2":r_squared,"Entropy: Mean":ent_mean,"Std":ent_std,"Max":ent_max,"Min":ent_min})
    table = ax[1].table(cellText=infos.values, colLabels=infos.columns, cellLoc = 'center', loc='center')
    #ax[1].set_title("Entropy")
    ax[1].axis('tight')
    ax[1].axis('off')
    # set the style of the header
    for col in range(7):
        header_cell = table[0, col]  # Access header cells (row 0)
        header_cell.set_facecolor('lightgrey')  # Header background color
        header_cell.set_text_props(weight='bold', fontsize=12)  # Header text style
        header_cell.set_linewidth(2)  # Thicker border for header
    
    plt.colorbar(im)
    frac = "_frac" if fraction else ""
    plt.savefig(f"{path}/heatmap{frac}{suffix}.pdf")
    
# plotting the entropy histograms
def entropy_histograms(path,expgrad=False):
    if not expgrad:
        gammas = np.linspace(0,1,11)
    else:
        gammas = np.linspace(0.01,0.1,11)

        
    fig, ax = plt.subplots(len(gammas),3, figsize=(20,40))
    plt.subplots_adjust(hspace=0.5)
    
    for i,gamma in enumerate(gammas):
        if not expgrad:
            gamma = round(gamma,1)
        else:
            gamma = round(gamma,3)
        data = pd.read_csv(f"{path}/results_{gamma}.csv")
        entr_res = data["entropy"].to_numpy()
        S = data["S"].to_numpy()

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

def entropy_lineplot(path,expgrad=False):
    if not expgrad:
        gammas = np.linspace(0,1,11)
    else:
        gammas = np.linspace(0.01,0.1,11)
    fig, ax = plt.subplots(1,1, figsize=(20,6))
    mean_entropies = []
    std_entropies = []
    mean_entropies_priv = []
    std_entropies_priv = []
    mean_entropies_unpriv = []
    std_entropies_unpriv = []
    
    for gamma in gammas:
        if not expgrad:
            gamma = round(gamma,1)
        else:
            gamma = round(gamma,3)
        data = pd.read_csv(f"{path}/results_{gamma}.csv")
        entr_res = data["entropy"].to_numpy()
        S = data["S"].to_numpy()
        # Split into privileged and unprivileged group
        res_priv = entr_res[np.where(S == 1)]
        res_unpriv = entr_res[np.where(S == 0)]
        
        mean_entropies.append(np.mean(entr_res))
        std_entropies.append(np.std(entr_res))
        mean_entropies_priv.append(np.mean(res_priv))
        std_entropies_priv.append(np.std(res_priv))
        mean_entropies_unpriv.append(np.mean(res_unpriv))
        std_entropies_unpriv.append(np.std(res_unpriv))
    
    ax.set_title("Mean of Entropies")
    ax.set_xlabel('Gamma')
    ax.set_ylabel('Entropy')
        
    ax.plot(gammas,mean_entropies, "-o",color='skyblue', label='All Groups')
    ax.fill_between(gammas, np.array(mean_entropies) - np.array(std_entropies), np.array(mean_entropies) + np.array(std_entropies), color='skyblue', alpha=0.1)
    ax.plot(gammas,mean_entropies_priv, "-o", color='limegreen', label='S=1')
    ax.fill_between(gammas, np.array(mean_entropies_priv) - np.array(std_entropies_priv), np.array(mean_entropies_priv) + np.array(std_entropies_priv), color='limegreen', alpha=0.1)
    ax.plot(gammas,mean_entropies_unpriv, "-o", color='coral', label='S=0')
    ax.fill_between(gammas, np.array(mean_entropies_unpriv) - np.array(std_entropies_unpriv), np.array(mean_entropies_unpriv) + np.array(std_entropies_unpriv), color='coral', alpha=0.1)
    
    ax.legend()
    
    plt.savefig(f"{path}/entropy_lineplot.pdf")
        
def plot_acc(path):
    fig, ax = plt.subplots(1,1, figsize=(20,6))
    data = pd.read_csv(f"{path}/accuracies.csv",header=0)
    gammas = data["gamma"]
    mean_accuracies = data["acc_mean"]
    std_accurs = data["acc_std"]
    ax.plot(gammas,mean_accuracies,"-o" ,color='skyblue')
    ax.errorbar(gammas,mean_accuracies, yerr=std_accurs, fmt='o', color='skyblue')
    ax.set_ylim(0,1)
    ax.set_title("Accuracies of trained models")
    ax.set_xlabel('Gamma')
    ax.set_ylabel('Acc')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"{path}/accuracies.pdf")
    
# making a Kolmogorov-Smirnov test and plot results  
def ks_tests(path,expgrad=False):
    if not expgrad:
        gammas = np.linspace(0,1,11)
    else:
        gammas = np.linspace(0.01,0.1,11)
    fig, ax = plt.subplots(1,1, figsize=(20,6))
    with open(f"{path}/ks_test.txt", 'a') as file:
        kstest_results = []
        for gamma in gammas:
            if not expgrad:
                gamma = round(gamma,1)
            else:
                gamma = round(gamma,3)
            
            data = pd.read_csv(f"{path}/results_{gamma}.csv")
            S = data["S"].to_numpy()
            res = data["entropy"].to_numpy()

            # Split into privileged and unprivileged group
            res_priv = res[np.where(S == 1)]
            res_unpriv = res[np.where(S == 0)]
            
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

def std_histogram(path,expgrad=False):
    if not expgrad:
        gammas = np.linspace(0,1,11)
    else:
        gammas = np.linspace(0.01,0.1,11)
    fig, ax = plt.subplots(len(gammas),3, figsize=(20,40))
    plt.subplots_adjust(hspace=0.5)
    
    for i,gamma in enumerate(gammas):
        if not expgrad:
            gamma = round(gamma,1)
        else:
            gamma = round(gamma,3)
        data = pd.read_csv(f"{path}/results_{gamma}.csv")
        entr_res = data["std_prediction"].to_numpy()
        S = data["S"].to_numpy()

        # Split into privileged and unprivileged group
        res_priv = entr_res[np.where(S == 1)]
        res_unpriv = entr_res[np.where(S == 0)]
        
        ax[i,0].hist(entr_res, bins=20, range=(0,1), color='skyblue', edgecolor='black', alpha=0.7)#,weights=weights)
        ax[i,0].set_title(f"Gamma = {gamma} |   All Groups")
        ax[i,0].set_xlabel('Std')
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
        ax[i,1].set_xlabel('Std')
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
        ax[i,2].set_xlabel('Std')
        ax[i,2].set_ylabel('Number')
        ax[i,2].grid(True, linestyle=':', alpha=0.6)
        ax[i,2].set_xlim(0,1)
        mean = np.mean(res_unpriv)
        std = np.std(res_unpriv)
        ax[i,2].axvline(mean, color='red', linestyle='dashed', linewidth=2)
        ax[i,2].axvline(mean + std, color='grey', linestyle='dashed', linewidth=1)
        ax[i,2].axvline(mean - std, color='grey', linestyle='dashed', linewidth=1)

    plt.savefig(f"{path}/std_histogram.pdf")


def all_plots(path,expgrad):
    plot_acc(path)
    ks_tests(path,expgrad)
    entropy_histograms(path,expgrad)
    heatmap_feature_weights(path,expgrad=expgrad)
    entropy_lineplot(path,expgrad)
    std_histogram(path,expgrad)

if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--path", type=str, required=True)
    argparser.add_argument("--expgrad", type=str, default=False)
    
    args = argparser.parse_args()
    path = args.path
    expgrad = args.expgrad
    
    all_plots(path,expgrad)