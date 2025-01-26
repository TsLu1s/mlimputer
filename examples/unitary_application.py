import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from mlimputer.imputation import MLimputer 
import mlimputer.parameters as params      

import warnings
warnings.filterwarnings("ignore", category=Warning) #-> For a clean console

## Dataset Selection
sel_dataset = "Dataset 1" # "Dataset 2" , "Dataset 3"

########################################## Dataset 1
if sel_dataset == "Dataset 1":
    # Source Data: https://www.openml.org/search?type=datastatus=activeid=41506
    data = pd.read_csv('https://raw.githubusercontent.com/TsLu1s/MLimputer/main/data/NewFuelCar.csv', encoding='latin', delimiter=',')
    data = data.drop('X', axis=1)
    target = "Tmax"

########################################## Dataset 2

elif sel_dataset == "Dataset 2":
    # Source Data: "https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset"
    data=pd.read_csv('https://github.com/TsLu1s/MLimputer/raw/main/data/body_measurement.csv', encoding='latin', delimiter=',') 
    target="BodyFat"

########################################## Dataset 3

elif sel_dataset == "Dataset 3":
    # Source Data: "https://www.kaggle.com/code/sagardubey3/admission-prediction-with-linear-regression"
    data = pd.read_csv('https://raw.githubusercontent.com/TsLu1s/MLimputer/main/data/Admission_Predict.csv', encoding='latin', delimiter=',') 
    target = "Chance of Admit "
    
sel_cols = [col for col in data.columns if col != target] + [target]
data = data[sel_cols]
missing_ratio = 0.1  # 10% missing values
for col in sel_cols[:-1]:
    missing_mask = np.random.rand(data.shape[0]) < missing_ratio
    data.loc[missing_mask, col] = np.nan

data = data[data[target].isnull()==False]
data = data.reset_index(drop=True)
# Important Note: If Classification, target should be categorical.  -> data[target]=data[target].astype('object')

train,test = train_test_split(data, train_size=0.8) 
train,test = train.reset_index(drop=True), test.reset_index(drop=True) # <- Required
train.isna().sum(), test.isna().sum()

# All model imputation options ->  "RandomForest","ExtraTrees","GBR","KNN","XGBoost","Lightgbm","Catboost"

# Customizing Hyperparameters Example
hparameters = params.imputer_parameters()
print(hparameters)
hparameters["KNN"]["n_neighbors"] = 5
hparameters["RandomForest"]["n_estimators"] = 30
    
# Imputation Example 1 : KNN

mli = MLimputer(imput_model = "KNN", imputer_configs = hparameters)
mli.fit_imput(X=train)
train_knn = mli.transform_imput(X=train)
test_knn = mli.transform_imput(X=test)

# Imputation Example 2 : RandomForest

mli = MLimputer(imput_model = "RandomForest", imputer_configs = hparameters)
mli.fit_imput(X=train)
train_rf = mli.transform_imput(X=train)
test_rf = mli.transform_imput(X=test)
    

#(...)

## Export Imputation Metadata

#import pickle 
#output = open("imputer_rf.pkl", 'wb')
#pickle.dump(mli, output)