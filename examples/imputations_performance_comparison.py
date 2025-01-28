import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBRegressor

from mlimputer.evaluation import Evaluator                     
from mlimputer.parameters import imputer_parameters                     

import warnings
warnings.filterwarnings("ignore", category=Warning) #-> For a clean console

sel_dataset = "Dataset 2"

########################################## Dataset 1
if sel_dataset == "Dataset 1":
    # Source Data: https://www.openml.org/search?type=datastatus=activeid=41506
    data = pd.read_csv('https://raw.githubusercontent.com/TsLu1s/MLimputer/main/data/NewFuelCar.csv', encoding='latin', delimiter=',')
    data = data.drop('X', axis=1)
    target = "Tmax"

########################################## Dataset 2

elif sel_dataset == "Dataset 2":
    # Source Data: "https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset"
    data = pd.read_csv('https://github.com/TsLu1s/MLimputer/raw/main/data/body_measurement.csv', encoding='latin', delimiter=',') 
    target = "BodyFat"

########################################## Dataset 3

elif sel_dataset == "Dataset 3":
    # Source Data: "https://www.kaggle.com/code/sagardubey3/admission-prediction-with-linear-regression"
    data = pd.read_csv('https://raw.githubusercontent.com/TsLu1s/MLimputer/main/data/Admission_Predict.csv', encoding='latin', delimiter=',') 
    target = "Chance of Admit "
    
########################################## Dataset 4
if sel_dataset == "Dataset 4":
    # Source Data: https://github.com/airtlab/machine-learning-for-quality-prediction-in-plastic-injection-molding
    url = 'https://raw.githubusercontent.com/TsLu1s/MLimputer/main/data/injection_quality.csv'
    data = pd.read_csv(url, encoding='latin', delimiter=';')
    target = "quality"
    data[target] = data[target].astype('object') 

## Generate Random Null Values in Dataset
sel_cols = [col for col in data.columns if col != target] + [target]
data = data[sel_cols]
missing_ratio = 0.1 # 10% missing values
for col in sel_cols[:-1]:
        missing_mask = np.random.rand(data.shape[0]) < missing_ratio
        data.loc[missing_mask, col] = np.nan

##############################################################################################################################
##### Preprocessing Data
data = data[data[target].isnull() == False]
data = data.reset_index(drop = True)
# IMPORTANT NOTE: If Classification, target should be categorical.  -> data[target]=data[target].astype('object')

train,test = train_test_split(data, train_size = 0.8)
train,test = train.reset_index(drop = True), test.reset_index(drop = True) 
train.isna().sum(), test.isna().sum()

## Customizing parameters example
hparameters = imputer_parameters()
print(hparameters)
hparameters["RandomForest"]["n_estimators"] = 15
hparameters["ExtraTrees"]["n_estimators"] = 15
hparameters["GBR"]["n_estimators"] = 15
hparameters["KNN"]["n_neighbors"] = 5
hparameters["Lightgbm"]["learning_rate"] = 0.01
hparameters["Catboost"]["loss_function"] = "MAE"

"""
Evaluation Process Overview
The framework implements a comprehensive two-stage evaluation approach:

Cross-Validation Assessment: Evaluates multiple imputation models using k-fold cross-validation to ensure robust performance metrics.
Test Set Validation: Validates the selected imputation strategy on a separate test set to confirm generalization capability.
"""

############ Define your inputs
imputation_models = ["RandomForest", "ExtraTrees", "GBR", "KNN",]
                    #"XGBoost", "Lightgbm", "Catboost"]   # List of imputation models to evaluate
n_splits = 3  # Number of splits for cross-validation

# Selected models for classification and regression
if train[target].dtypes == "object":                                      
            models = [RandomForestClassifier(), DecisionTreeClassifier()]
else:
    models = [XGBRegressor(), RandomForestRegressor()]


# Initialize the evaluator
evaluator = Evaluator(
    imputation_models = imputation_models,  
    train = train,
    target = target,
    n_splits = n_splits,     
    hparameters = hparameters                                 
)

# Perform evaluations
cv_results = evaluator.evaluate_imputation_models(
    models = models 
)

best_imputer = evaluator.get_best_imputer()  # Get best-performing imputation model

test_results = evaluator.evaluate_test_set(
    test = test,
    imput_model = best_imputer,
    models = models
)
