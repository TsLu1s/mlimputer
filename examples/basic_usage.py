from sklearn.model_selection import train_test_split
from mlimputer import MLimputer
from mlimputer.schemas.parameters import imputer_parameters
from mlimputer.data.data_generator import ImputationDatasetGenerator

import warnings
warnings.filterwarnings("ignore")

print("="*60)
print("MLIMPUTER - BASIC USAGE")
print("="*60)

# ============================================================================
# Generate Dataset
# ============================================================================
generator = ImputationDatasetGenerator(random_state=42)
X, y = generator.quick_multiclass(n_samples=2000, missing_rate=0.15, n_categorical=5)

print(f"\nDataset: {X.shape}")
print(f"Missing values: {X.isnull().sum().sum()} ({X.isnull().sum().sum()/X.size:.1%})")

# Split data and reset index
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

# ============================================================================
# Configure Imputation
# ============================================================================
print("\n" + "="*60)
print("IMPUTATION")
print("="*60)

# Configure parameters
params = imputer_parameters()
params["KNN"]["n_neighbors"] = 5
params["KNN"]["weights"] = "distance"

# Create and fit imputer
imputer = MLimputer(
    imput_model="KNN",
    imputer_configs=params
)

imputer.fit(X_train)

# Transform both sets
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)

# ============================================================================
# Results
# ============================================================================
print("\n" + "="*60)
print("IMPUTATION RESULTS")
print("="*60)

print(f"\nTraining Set:")
print(f"  Missing values: {X_train.isnull().sum().sum():,} → {X_train_imputed.isnull().sum().sum():,}")
print(f"  Imputed: {X_train.isnull().sum().sum():,} values")

print(f"\nTest Set:")
print(f"  Missing values: {X_test.isnull().sum().sum():,} → {X_test_imputed.isnull().sum().sum():,}")
print(f"  Imputed: {X_test.isnull().sum().sum():,} values")

# ============================================================================
# Save Imputer
# ============================================================================
print("\n" + "="*60)
print("SAVE & LOAD")
print("="*60)

import pickle

# Save fitted imputer
with open("fitted_imputer.pkl", 'wb') as f:
    pickle.dump(imputer, f)
print("✓ Imputer saved to 'fitted_imputer.pkl'")

# Load and test
with open("fitted_imputer.pkl", 'rb') as f:
    loaded_imputer = pickle.load(f)
print("✓ Imputer loaded successfully")

# Test on new data
new_data = generator.quick_multiclass(n_samples=100, missing_rate=0.2, n_categorical=5)[0]
new_data_imputed = loaded_imputer.transform(new_data)
print(f"✓ New data imputed: {new_data.isnull().sum().sum()} → {new_data_imputed.isnull().sum().sum()} missing")

print("\n" + "="*60)
print("BASIC USAGE COMPLETED")
print("="*60)