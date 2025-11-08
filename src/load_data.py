from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np

# preprocessing data just like in our framework
def load_compas(seed):
	data = pd.read_csv('../data/compas-cls.csv')

	X = data.drop(["race","two_year_recid"],axis=1)
	y = data["two_year_recid"].copy()
	S = data["race"].copy()
	S = S.replace({"African-American": 0, "Caucasian": 1}).astype(int)	
	cat_features = ['sex','age_cat','c_charge_degree']
	cont_features = ['priors_count']	
	X = pd.get_dummies(X, columns=cat_features,dtype=int)

	y = y.to_numpy()
	S = S.to_numpy()

	X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, train_size=0.666, random_state=seed)	
	scaler = StandardScaler()
	X_train[cont_features] = scaler.fit_transform(X_train[cont_features])
	X_test[cont_features] = scaler.transform(X_test[cont_features])	
	X_train = X_train#.to_numpy()
	X_test = X_test#.to_numpy()

	return X_train, X_test, y_train, y_test, S_train, S_test

def load_banks(seed):
	data = pd.read_csv('../data/banks.csv',sep=';')

	X = data.drop(["age","y"],axis=1)
	y = data["y"].copy()
	y = y.replace({"yes":1,"no":0}).astype(int)
	S = data["age"].copy()
	S = S.replace({'age > 25 and age < 60':1,'age <= 25 or age >= 60':0}).astype(int)

	cat_features = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
	cont_features = ['duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']

	X = pd.get_dummies(X, columns=cat_features,dtype=int)
    
	y = y.to_numpy()
	S = S.to_numpy()
    
	X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, train_size=0.666, random_state=seed)

	scaler = StandardScaler()
	X_train[cont_features] = scaler.fit_transform(X_train[cont_features])
	X_test[cont_features] = scaler.transform(X_test[cont_features])

	X_train = X_train#.to_numpy()
	X_test = X_test#.to_numpy()
    
	return X_train, X_test, y_train, y_test, S_train, S_test

def load_adult(seed):
	data = pd.read_csv('../data/adult.csv',sep=',')

	X = data.drop(["income","sex"],axis=1)
	y = data["income"].copy()
	y = y.replace({" <=50K": 1, " >50K": 0}).astype(int)
	S = data["sex"].copy()
	S = S.replace({" Male":1, " Female":0}).astype(int) 

	cat_features = ['workclass','education','marital-status','occupation','relationship','native-country','race']
	cont_features = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
    
	X = pd.get_dummies(X, columns=cat_features,dtype=int)

	y = y.to_numpy()
	S = S.to_numpy()
    
	X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, train_size=0.666, random_state=seed)

	scaler = StandardScaler()
	X_train[cont_features] = scaler.fit_transform(X_train[cont_features])
	X_test[cont_features] = scaler.transform(X_test[cont_features])

	X_train = X_train#.to_numpy()
	X_test = X_test#.to_numpy()
    
	return X_train, X_test, y_train, y_test, S_train, S_test

def load_german(seed):
	data = pd.read_csv('../data/german.csv',sep=',')

	X = data.drop(["y","Sex"],axis=1)
	y = data["y"].copy()
	y = y.replace({2: 0}).astype(int)
	S = data["Sex"].copy()
	S = S.replace({"male":1, "female":0}).astype(int) 

	cat_features = ['Job','Housing','Saving accounts','Checking account','Purpose']
	cont_features = ['Age','Credit amount','Duration']
    
	X = pd.get_dummies(X, columns=cat_features,dtype=int)

	y = y.to_numpy()
	S = S.to_numpy()
    
	X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, train_size=0.666, random_state=seed)

	scaler = StandardScaler()
	X_train[cont_features] = scaler.fit_transform(X_train[cont_features])
	X_test[cont_features] = scaler.transform(X_test[cont_features])

	X_train = X_train#.to_numpy()
	X_test = X_test#.to_numpy()
    
	return X_train, X_test, y_train, y_test, S_train, S_test

def load_folktables_AK(seed):
	data = pd.read_csv('../data/folktables_AK_Income_2017.csv',sep=',')
	
	X = data.drop(["PINCP","RAC1P"],axis=1)
	y = data["PINCP"].copy()
	S = data["RAC1P"].copy()

	cat_features = ["COW","SCHL","MAR","POBP","RELP","SEX"]
	cont_features = ["AGEP","OCCP","WKHP"]
    
	X = pd.get_dummies(X, columns=cat_features,dtype=int)

	y = y.to_numpy()
	S = S.to_numpy()
    
	X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, train_size=0.666, random_state=seed)

	scaler = StandardScaler()
	X_train[cont_features] = scaler.fit_transform(X_train[cont_features])
	X_test[cont_features] = scaler.transform(X_test[cont_features])

	X_train = X_train#.to_numpy()
	X_test = X_test#.to_numpy()
    
	return X_train, X_test, y_train, y_test, S_train, S_test

def load_folktables_HI(seed):
	data = pd.read_csv('../data/folktables_HI_Income_2017.csv',sep=',')

	X = data.drop(["PINCP","RAC1P"],axis=1)
	y = data["PINCP"].copy()
	S = data["RAC1P"].copy()

	cat_features = ["COW","SCHL","MAR","POBP","RELP","SEX"]
	cont_features = ["AGEP","OCCP","WKHP"]
    
	X = pd.get_dummies(X, columns=cat_features,dtype=int)
    
	y = y.to_numpy()
	S = S.to_numpy()
    
	X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, train_size=0.666, random_state=seed)

	scaler = StandardScaler()
	X_train[cont_features] = scaler.fit_transform(X_train[cont_features])
	X_test[cont_features] = scaler.transform(X_test[cont_features])

	X_train = X_train#.to_numpy()
	X_test = X_test#.to_numpy()
    
	return X_train, X_test, y_train, y_test, S_train, S_test

def load_toydata(seed):
    data = pd.read_csv("../data/toydata_uncertainty.csv")
    X = data[["x1","x2"]]
    y = data["y"].to_numpy()
    S = data["s"].to_numpy()

    X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, train_size=0.666, random_state=seed)

    return X_train, X_test, y_train, y_test, S_train, S_test

DATALOADER = {
    "compas": load_compas,  
    "banks": load_banks,
    "adult": load_adult,
    "german": load_german,
    "folktables_AK": load_folktables_AK,
    "folktables_HI": load_folktables_HI,
    "toy": load_toydata
}

if __name__=="__main__":
	X_train, X_test, y_train, y_test, S_train, S_test = load_compas(123)
	X_train, X_test, y_train, y_test, S_train, S_test = load_adult(123)
	X_train, X_test, y_train, y_test, S_train, S_test = load_german(123)
	X_train, X_test, y_train, y_test, S_train, S_test = load_banks(123)
	X_train, X_test, y_train, y_test, S_train, S_test = load_folktables_AK(123)
	X_train, X_test, y_train, y_test, S_train, S_test = load_folktables_HI(123)
	X_train, X_test, y_train, y_test, S_train, S_test = load_toydata(123)

