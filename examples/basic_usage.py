from mlimputer import MLimputer
from mlimputer.schemas.parameters import imputer_parameters
from mlimputer.data.data_generator import ImputationDatasetGenerator
from mlimputer.utils.splitter import DataSplitter

import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# Generate Dataset
# ============================================================================
print("="*60)
print("MLIMPUTER BASIC USAGE")
print("="*60)

# Generate dataset with missing values
generator = ImputationDatasetGenerator(random_state=42)
X, y = generator.quick_multiclass(n_samples=2000, missing_rate=0.15, n_categorical=5) # multiclass classification dataset

# ========== Other data generation options
#generator.quick_binary
#generator.quick_regression

print(f"\nDataset: {X.shape}")
print(f"Missing: {X.isnull().sum().sum()} values ({X.isnull().sum().sum()/X.size:.1%})")

# Split data with automatic index reset
splitter = DataSplitter(random_state=42)
X_train, X_test, y_train, y_test = splitter.split(X, y, test_size=0.2)

# ============================================================================
# Imputation
# ============================================================================
print("\n" + "="*60)
print("IMPUTATION")
print("="*60)

# Configure and run
params = imputer_parameters()
params["RandomForest"]["n_estimators"] = 50

imputer = MLimputer(imput_model="RandomForest", imputer_configs=params)
imputer.fit(X=X_train)

X_train_imputed = imputer.transform(X=X_train)
X_test_imputed = imputer.transform(X=X_test)

print(f"Train: {X_train.isnull().sum().sum()} → 0 missing")
print(f"Test:  {X_test.isnull().sum().sum()} → 0 missing")

# ============================================================================
# Save Imputer
# ============================================================================
print("\n" + "="*60)
print("SAVE IMPUTER")
print("="*60)

import pickle

# Save the fitted imputer
with open("fitted_imputer.pkl", 'wb') as f:
    pickle.dump(imputer, f)
print("Imputer saved to 'fitted_imputer.pkl'")

# Example: Load and use on new data
with open("fitted_imputer.pkl", 'rb') as f:
    loaded_imputer = pickle.load(f)
print("Imputer loaded successfully")

# Test on new data
# Generate new data WITH the same structure (including categorical columns)
new_data = generator.quick_multiclass(
    n_samples=100, 
    missing_rate=0.2, 
    n_categorical=5  # Same as training data
)[0]

new_data_imputed = loaded_imputer.transform(new_data)
print(f"New data imputed: {new_data.isnull().sum().sum()} → {new_data_imputed.isnull().sum().sum()} missing")

# ============================================================================