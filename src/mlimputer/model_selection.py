import pandas as pd
from mlimputer.models_imputation import (RandomForestImputation,         
                                         ExtraTreesImputation,
                                         GBRImputation,
                                         KNNImputation,
                                         XGBoostImputation,
                                         CatBoostImputation,
                                         LightGBMImputation)
from mlimputer.parameters import imputer_parameters                      
 
parameters=imputer_parameters()

def imput_models(train : pd.DataFrame,
                 target : str = "y",
                 algo : str = 'RandomForest',
                 parameters : dict = parameters):
    """
    This function trains and returns a regression model based on the input data and specified algorithm.
    
    Parameters:
    train (pd.DataFrame): The input training data
    target (str, optional): The target column name in the training data. Default is 'y'
    algo (str, optional): The algorithm to be used for training. Default is 'RandomForest'
    parameters (dict, optional): The hyperparameters for the specified algorithm. Default is 'parameters'
    
    Returns:
    model: trained machine learning model.
    """
    
    sel_cols = [col for col in train.columns if col != target] + [target]
    train = train[sel_cols]
    
    X_train = train.iloc[:, 0:(len(sel_cols)-1)].values
    y_train = train.iloc[:, (len(sel_cols)-1)].values
    
    if algo == 'RandomForest':
        rf_params = parameters['RandomForest']
        model = RandomForestImputation(**rf_params)
        model.fit(X_train, y_train)
        
    elif algo == 'ExtraTrees':
        et_params = parameters['ExtraTrees']
        model = ExtraTreesImputation(**et_params)
        model.fit(X_train, y_train)
        
    elif algo == 'GBR':
        gbr_params = parameters['GBR']
        model = GBRImputation(**gbr_params)
        model.fit(X_train, y_train)
        
    elif algo == 'KNN':
        knn_params = parameters['KNN']
        model = KNNImputation(**knn_params)
        model.fit(X_train, y_train)
        
    elif algo == 'XGBoost':
        xg_params = parameters['XGBoost']
        model = XGBoostImputation(**xg_params)
        model.fit(X_train, y_train)
    
    elif algo == "Catboost":
        cb_params = parameters['Catboost']
        model = CatBoostImputation(**cb_params)
        model.fit(X_train, y_train)
        
    elif algo == "Lightgbm":
        lb_params = parameters['Lightgbm']
        model = LightGBMImputation(**lb_params)
        model.fit(X_train, y_train)
        
    else:
        raise ValueError('Invalid model')
   
    return model

