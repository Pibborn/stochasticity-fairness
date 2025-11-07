from src.combined_project.pipelines.preprocess_data.nodes import split_into_X_y_S
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd


# preprocessing data just like in our framework
def load_compas(seed):
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
    
    X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, train_size=0.666, random_state=seed)

    scaler = StandardScaler()
    X_train[cont] = scaler.fit_transform(X_train[cont])
    X_test[cont] = scaler.transform(X_test[cont])

    X_train = X_train#.to_numpy()
    X_test = X_test#.to_numpy()
    
    return X_train, X_test, y_train, y_test, S_train, S_test

def load_banks(seed):
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
    
    X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, train_size=0.666, random_state=seed)

    scaler = StandardScaler()
    X_train[cont] = scaler.fit_transform(X_train[cont])
    X_test[cont] = scaler.transform(X_test[cont])

    X_train = X_train#.to_numpy()
    X_test = X_test#.to_numpy()
    
    return X_train, X_test, y_train, y_test, S_train, S_test

def load_adult(seed):
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
    
    X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, train_size=0.666, random_state=seed)

    scaler = StandardScaler()
    X_train[cont] = scaler.fit_transform(X_train[cont])
    X_test[cont] = scaler.transform(X_test[cont])

    X_train = X_train#.to_numpy()
    X_test = X_test#.to_numpy()
    
    return X_train, X_test, y_train, y_test, S_train, S_test

def load_german(seed):
    data = pd.read_csv('../data/01_raw/german.data',sep=',')
    dataset = {"data": "german-sex_data"
      ,"label": 'y'
      ,"sep": ','
      ,"pos_label": 1
      ,"neg_label": 2
      ,"privileged_sens_attr": 'male'
      ,"unprivileged_sens_attr": 'female'
      ,"filter": False
      ,"sensitive_attributes":['Sex']
      ,"discard_features": [None]
      ,"categorial_features":['Job','Housing','Saving accounts','Checking account','Purpose']
      ,"continuous_features":['Age','Credit amount','Duration']}

    X, y, S, name, sens_attrs, cont, label = split_into_X_y_S(data,dataset)
    
    y = y.to_numpy()
    S = S.to_numpy()
    
    X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, train_size=0.666, random_state=seed)

    scaler = StandardScaler()
    X_train[cont] = scaler.fit_transform(X_train[cont])
    X_test[cont] = scaler.transform(X_test[cont])

    X_train = X_train#.to_numpy()
    X_test = X_test#.to_numpy()
    
    return X_train, X_test, y_train, y_test, S_train, S_test

def load_folktables_AK(seed):
    data = pd.read_csv('../data/01_raw/folktables_AK_Income_2017.csv',sep=',')
    dataset = {"data": "folktables_AK_Income_2017"
      ,"label": 'PINCP'
      ,"sep": ','
      ,"pos_label": 1
      ,"neg_label": 0
      ,"privileged_sens_attr": 1
      ,"unprivileged_sens_attr": 0
      ,"filter": False
      ,"sensitive_attributes":['RAC1P']
      ,"discard_features": [None]
      ,"categorial_features":["COW","SCHL","MAR","POBP","RELP","SEX"]
      ,"continuous_features":["AGEP","OCCP","WKHP"]}

    X, y, S, name, sens_attrs, cont, label = split_into_X_y_S(data,dataset)
    
    y = y.to_numpy()
    S = S.to_numpy()
    
    X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, train_size=0.666, random_state=seed)

    scaler = StandardScaler()
    X_train[cont] = scaler.fit_transform(X_train[cont])
    X_test[cont] = scaler.transform(X_test[cont])

    X_train = X_train#.to_numpy()
    X_test = X_test#.to_numpy()
    
    return X_train, X_test, y_train, y_test, S_train, S_test

def load_folktables_HI(seed):
    data = pd.read_csv('../data/01_raw/folktables_HI_Income_2017.csv',sep=',')
    dataset = {"data": "folktables_HI_Income_2017"
      ,"label": 'PINCP'
      ,"sep": ','
      ,"pos_label": 1
      ,"neg_label": 0
      ,"privileged_sens_attr": 1
      ,"unprivileged_sens_attr": 0
      ,"filter": False
      ,"sensitive_attributes":['RAC1P']
      ,"discard_features": [None]
      ,"categorial_features":["COW","SCHL","MAR","POBP","RELP","SEX"]
      ,"continuous_features":["AGEP","OCCP","WKHP"]}

    X, y, S, name, sens_attrs, cont, label = split_into_X_y_S(data,dataset)
    
    y = y.to_numpy()
    S = S.to_numpy()
    
    X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, train_size=0.666, random_state=seed)

    scaler = StandardScaler()
    X_train[cont] = scaler.fit_transform(X_train[cont])
    X_test[cont] = scaler.transform(X_test[cont])

    X_train = X_train#.to_numpy()
    X_test = X_test#.to_numpy()
    
    return X_train, X_test, y_train, y_test, S_train, S_test

DATALOADER = {
    "compas": load_compas,  
    "banks": load_banks,
    "adult": load_adult,
    "german": load_german,
    "folktables_AK": load_folktables_AK,
    "folktables_HI": load_folktables_HI
}