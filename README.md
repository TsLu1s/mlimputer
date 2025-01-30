[![LinkedIn][linkedin-shield]][linkedin-url]
[![Contributors][contributors-shield]][contributors-url]
[![Stargazers][stars-shield]][stars-url]
[![MIT License][license-shield]][license-url]
[![Downloads][downloads-shield]][downloads-url]
[![Month Downloads][downloads-month-shield]][downloads-month-url]

[contributors-shield]: https://img.shields.io/github/contributors/TsLu1s/MLimputer.svg?style=for-the-badge&logo=github&logoColor=white
[contributors-url]: https://github.com/TsLu1s/MLimputer/graphs/contributors
[stars-shield]: https://img.shields.io/github/stars/TsLu1s/MLimputer.svg?style=for-the-badge&logo=github&logoColor=white
[stars-url]: https://github.com/TsLu1s/MLimputer/stargazers
[license-shield]: https://img.shields.io/github/license/TsLu1s/MLimputer.svg?style=for-the-badge&logo=opensource&logoColor=white
[license-url]: https://github.com/TsLu1s/MLimputer/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/luísfssantos/
[downloads-shield]: https://static.pepy.tech/personalized-badge/mlimputer?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Total%20Downloads
[downloads-url]: https://pepy.tech/project/mlimputer
[downloads-month-shield]: https://static.pepy.tech/personalized-badge/mlimputer?period=month&units=international_system&left_color=grey&right_color=blue&left_text=Month%20Downloads
[downloads-month-url]: https://pepy.tech/project/mlimputer

<br>
<p align="center">
  <h2 align="center"> MLimputer: Missing Data Imputation Framework for Machine Learning
  <br>
  
## Framework Contextualization <a name = "ta"></a>

The `MLimputer` project constitutes an complete and integrated pipeline to automate the handling of missing values in datasets through regression prediction and aims at reducing bias and increase the precision of imputation results when compared to more classic imputation methods.
This package provides multiple algorithm options to impute your data, in which every observed data column with existing missing values is fitted with a robust preprocessing approach and subsequently predicted.

The architecture design includes three main sections, these being: missing data analysis, data preprocessing and supervised model imputation which are organized in a customizable pipeline structure.

This project aims at providing the following application capabilities:

* General applicability on tabular datasets: The developed imputation procedures are applicable on any data table associated with any Supervised ML scopes, based on missing data columns to be imputed.
    
* Robustness and improvement of predictive results: The application of the MLimputer preprocessing aims at improve the predictive performance through customization and optimization of existing missing values imputation in the dataset input columns. 
   
#### Main Development Tools <a name = "pre1"></a>

Major frameworks used to built this project: 

* [Pandas](https://pandas.pydata.org/)
* [Sklearn](https://scikit-learn.org/stable/)
* [CatBoost](https://catboost.ai/)
    
## Where to get it <a name = "ta"></a>
    
Binary installer for the latest released version is available at the Python Package Index [(PyPI)](https://pypi.org/project/mlimputer/).   

## Installation  

To install this package from Pypi repository run the following command:

```
pip install mlimputer
```

# MLImputer - Usage Examples
    
The first needed step after importing the package is to load a dataset (split it) and define your choosen imputation model.
The imputation model options for handling the missing data in your dataset are the following:
* `RandomForest`
* `ExtraTrees`
* `GBR`
* `KNN`
* `XGBoost`
* `Lightgbm`
* `Catboost`

After creating a `MLimputer` object with your imputation selected model, you can then fit the missing data through the `fit_imput` method. From there you can impute the future datasets with `transform_imput` (validate, test ...) with the same data properties. Note, as it shows in the example bellow, you can also customize your model imputer parameters by changing it's configurations and then, implementing them in the `imputer_configs` parameter.

Through the `cross_validation` function you can also compare the predictive performance evalution of multiple imputations, allowing you to validate which imputation model fits better your future predictions.

```py

from mlimputer.imputation import MLimputer
import mlimputer.model_selection as ms
from mlimputer.parameters import imputer_parameters
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=Warning) #-> For a clean console

data = pd.read_csv('csv_directory_path') # Dataframe Loading Example
# Important note: If Classification, target should be categorical.  -> data[target]=data[target].astype('object')

train,test = train_test_split(data, train_size=0.8)
train,test = train.reset_index(drop=True), test.reset_index(drop=True) # <- Required

# All model imputation options ->  "RandomForest","ExtraTrees","GBR","KNN","XGBoost","Lightgbm","Catboost"

# Customizing Hyperparameters Example

hparameters = imputer_parameters()
print(hparameters)
hparameters["KNN"]["n_neighbors"] = 5
hparameters["RandomForest"]["n_estimators"] = 30
    
# Imputation Example 1 : KNN

mli_knn = MLimputer(imput_model = "KNN", imputer_configs = hparameters)
mli_knn.fit_imput(X = train)
train_knn = mli_knn.transform_imput(X = train)
test_knn = mli_knn.transform_imput(X = test)

# Imputation Example 2 : RandomForest

mli_rf = MLimputer(imput_model = "RandomForest", imputer_configs = hparameters)
mli_rf.fit_imput(X = train)
train_rf = mli_rf.transform_imput(X = train)
test_rf = mli_rf.transform_imput(X = test)
    
#(...)

## Export Imputation Metadata
import pickle 
output = open("imputer_rf.pkl", 'wb')
pickle.dump(mli_rf, output)

```

## Performance Evaluation
The MLimputer framework includes a robust evaluation module that enables users to assess and compare the performance of different imputation strategies. This evaluation process is crucial for selecting the most effective imputation approach for your specific dataset and use case.

### Evaluation Process Overview
The framework implements a comprehensive two-stage evaluation approach:
1. Cross-Validation Assessment: Evaluates multiple imputation models using k-fold cross-validation to ensure robust performance metrics.
2. Test Set Validation: Validates the selected imputation strategy on a separate test set to confirm generalization capability.

### Implementation Example:
The following example demonstrates how to evaluate imputation models and select the best performing approach for your data:

```py
import mlimputer.evaluation as Evaluator                   
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor

# Define evaluation parameters
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
    hparameters = hparameters)

# Perform evaluations
cv_results = evaluator.evaluate_imputation_models(
    models = models)

best_imputer = evaluator.get_best_imputer()  # Get best-performing imputation model

test_results = evaluator.evaluate_test_set(
    test = test,
    imput_model = best_imputer,
    models = models)

```
    
## License

Distributed under the MIT License. See [LICENSE](https://github.com/TsLu1s/TSForecasting/blob/main/LICENSE) for more information.

## Contact 
 
Luis Santos - [LinkedIn](https://www.linkedin.com/in/lu%C3%ADsfssantos/)
