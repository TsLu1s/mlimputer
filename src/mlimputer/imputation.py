import pandas as pd
from typing import Dict
from tqdm import tqdm
from atlantic.processing.encoders import AutoLabelEncoder
from atlantic.imputers.imputation import AutoSimpleImputer
from mlimputer.parameters import imputer_parameters                         
from mlimputer.model_selection import imput_models          

parameters=imputer_parameters()

class MLimputer:
    """
    A supervised machine learning imputation class for handling numerical missing values.
    The class uses a two-step approach:
    1. Initial simple imputation for input features
    2. ML model-based imputation for target columns with missing values
    
    Attributes:
        imput_model (str): Name of the ML model to use for imputation
        imputer_configs (dict): Configuration parameters for the imputation models
        imp_config (dict): Storage for fitted imputers and preprocessors
        numeric_dtypes (set): Set of supported numerical datatypes
        _is_fitted (bool): Flag to track if the imputer has been fitted
    """
    def __init__ (self, 
                  imput_model : str,
                  imputer_configs : dict=parameters):

        self.imput_model = imput_model
        self.imputer_configs = imputer_configs
        self.imp_config = {}
        self.numeric_dtypes = {'int16', 'int32', 'int64', 'float16', 'float32', 'float64'}
        self._is_fitted = False
        self.encoder = None
    
    def _validate_input(self, X: pd.DataFrame, method: str) -> None:
        """
        Validate input DataFrame.
        
        Args:
            X: Input DataFrame
            method: Method name for error messages ('fit' or 'transform')
            
        Raises:
            ValueError: For invalid inputs
            TypeError: For incorrect data types
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        if X.empty:
            raise ValueError("Input DataFrame is empty")
            
        if method == 'transform' and not self._is_fitted:
            raise ValueError("MLimputer must be fitted before transform")
            
        # Check for columns with all null values
        null_cols = X.columns[X.isnull().all()].tolist()
        if null_cols:
            raise ValueError(f"Columns {null_cols} contain all null values")
    
    def _get_missing_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get report of columns with missing values.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with missing value statistics
        """
        num_cols = X.select_dtypes(include=self.numeric_dtypes).columns
        missing_stats = pd.DataFrame({
            'null_count': X[num_cols].isna().sum()[X[num_cols].isna().sum() > 0]
        })
        
        if missing_stats.empty:
            return pd.DataFrame()
            
        missing_stats['null_percentage'] = missing_stats['null_count'] / len(X)
        missing_stats['columns'] = missing_stats.index
        return missing_stats.sort_values('null_percentage', ascending=True).reset_index(drop=True)
    
    def fit_imput(self, 
                  X:pd.DataFrame):
        """
        
        This method fits missing data in a dataframe using the imputation method specified by the user.
        
        Parameters:
        X (pd.DataFrame): The input pandas dataframe that needs to be imputed.
        
        """
        
        self._validate_input(X, method='fit')
        X_ = X.copy()
        
        missing_report = self._get_missing_columns(X_)
        if missing_report.empty:
            raise ValueError("No missing values found in numerical columns")
            
        imp_targets = missing_report['columns'].tolist()
        
        # Iterate over each column with missing data and fit the imputation method
        for target in tqdm(imp_targets, 
                           desc = "Fitting Missing Data", 
                           ncols = 80):
            
            # Split the data into train and test sets
            total_index = X_.index.tolist()
            test_index = X_[X_[target].isnull()].index.tolist()
            train_index = [ value for value in total_index if value not in test_index ]
            
            train = X_.iloc[train_index]
            
            cat_cols = [ col for col in train.select_dtypes(include = ['object','category']).columns if col != target ]
            
            if len(cat_cols) > 0:
                ## Create Label Encoder
                self.encoder = AutoLabelEncoder()
                ## Fit Label Encoder
                self.encoder.fit(X = train[cat_cols])
                # Transform the DataFrame using Label
                train = self.encoder.transform(X = train)
            
            # Fit the simple imputation method in input columns
            simple_imputer = AutoSimpleImputer(strategy = 'mean')
            simple_imputer.fit(train)  # Fit on the Train DataFrame
            train = simple_imputer.transform(train.copy())  # Transform the Train DataFrame
            
            # Fit the imputation model
            model = imput_models(train = train,
                                 target = target,
                                 parameters = self.imputer_configs,
                                 algo = self.imput_model)
            
            # Store fitted components
            self.imp_config[target] = {'model' : model,
                                       'pre_process' : self.encoder,
                                       'imputer' : simple_imputer}
        
        self._is_fitted = True
        return self
    
    def transform_imput(self,
                        X : pd.DataFrame):
        """
        Imputation of missing values in a X using a pre-fit imputation model.
        
        Parameters:
        -----------
        X: pd.DataFrame
            The X containing missing values to be imputed.
            
        Returns:
        --------
        X_: pd.DataFrame
            The original X with missing values imputed.
        """
        self._validate_input(X, method='transform')
        X_ = X.copy()
        
        for col in tqdm(list(self.imp_config.keys()) , desc = "Imputing Missing Data", ncols = 80):
            
            target = col
            test_index = X_[X_[target].isnull()].index.tolist()
            test = X_.iloc[test_index]
            
            encoder = self.imp_config[target]['pre_process']
            # Transform the DataFrame using Label
            if encoder is not None: 
                test = encoder.transform(X=test)
            
            # Impute the DataFrame using Simple Imputer
            simple_imputer = self.imp_config[target]['imputer']
            test = simple_imputer.transform(test.copy())  
            
            sel_cols = [col for col in test.columns if col != target] + [target]
            test = test[sel_cols]
            X_test = test.iloc[:, 0:(len(sel_cols)-1)].values
    
            model = self.imp_config[target]['model']
        
            y_predict = model.predict(X_test)
    
            X_[target].iloc[test_index] = y_predict
    
        return X_

    def get_model_info(self) -> Dict[str, Dict]:
        """
        Get information about fitted models and their configurations.
        
        Returns:
            Dictionary containing model configurations and statistics
        """
        if not self._is_fitted:
            raise ValueError("MLimputer must be fitted first")
            
        return {
            'imputation_model': self.imput_model,
            'imputated_columns': list(self.imp_config.keys()),
            'model_configs': self.imputer_configs[self.imput_model]
        }