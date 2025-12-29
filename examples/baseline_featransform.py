from sklearn.model_selection import train_test_split
from featransform.pipeline import Featransform
from featransform.configs.baseline import FTconfig
from featransform.utils.serializer import PipelineSerializer
from featransform.utils.data_generator import DatasetGenerator, make_dataset
import warnings
warnings.filterwarnings("ignore", category=Warning)

"""Baseline example of Featransform usage."""

###################################################### Dataset Generation Examples
###### Select your synthetic dataset or generate your own
##################################################################################

print("=" * 60)
print("Dataset Generation Examples")
print("=" * 60)

# Simple binary classification
X_simple, y_simple = make_dataset(task='binary_classification', n_samples=1500, complexity='simple')
print(f"\nSimple dataset: {X_simple.shape}, target classes: {y_simple.nunique()}")

# Medium complexity with datetime and categorical
X_medium, y_medium = make_dataset(task='multiclass_classification', n_samples=5000, complexity='medium', n_classes=3)
print(f"Medium dataset: {X_medium.shape}, target classes: {y_medium.nunique()}")
print(f"Column types: {X_medium.dtypes.value_counts().to_dict()}")

# Complex dataset with all features
X_complex, y_complex = make_dataset(task='regression', n_samples=10000, complexity='complex')
print(f"Complex dataset: {X_complex.shape}, target type: {y_complex.dtype}")

###################################################### Main Example

print("\n" + "=" * 60)
print("Basic Featransform Example")
print("=" * 60)

# Generate your own dataset for examples
X, y = DatasetGenerator.generate(
    task='multiclass_classification',                  ## Options: 'regression', 'binary_classification', 'multiclass_classification'
    n_samples=5000,
    n_features=20,
    n_informative=15,
    add_datetime=True,
    n_datetime_cols=2,
    add_categorical=True,
    n_categorical=3,
    add_missing=True,
    missing_rate=0.1,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nDataset shape: {X.shape}")
print(f"Features with nulls: {X.isnull().sum().sum()}")
print(f"Datetime columns: {X.select_dtypes(include=['datetime64']).shape[1]}")
print(f"Categorical columns: {X.select_dtypes(include=['category']).shape[1]}")

###################################################### Pipeline Configuration & Execution

print("\n" + "=" * 60)
print("Featransform Pipeline - Baseline Template")
print("=" * 60)

# Create pipeline
config = FTconfig.complete(task_type="multiclass_classification")        # Options: minimal(), standard(), optimized(), complete()
                                                                         # TaskType: REGRESSION, BINARY_CLASSIFICATION OR MULTICLASS_CLASSIFICATION
pipeline = Featransform(config)

# Fit and transform
pipeline.fit(X_train, y_train)
X_train_transformed = pipeline.transform(X_train)
X_test_transformed = pipeline.transform(X_test)

# Results
pipeline.optimization_report()

###################################################### Save & Load Pipeline
SAVE : bool = False

if SAVE:
    print("\n4. Save and Load Pipeline")
    print("-" * 40)

    # Save pipeline
    serializer = PipelineSerializer()
    serializer.save(pipeline, 'fitted_pipeline.pkl')
    print("Pipeline saved to 'fitted_pipeline.pkl'")

    # Load pipeline
    loaded_pipeline = serializer.load('fitted_pipeline.pkl')
    print("Pipeline loaded successfully")

    # Verify loaded pipeline works
    X_loaded_transform = loaded_pipeline.transform(X_test)
    print(f"Loaded pipeline output shape: {X_loaded_transform.shape}")

    ###################################################### One-liner Example

    print("\n5. One-liner Fit-Transform")
    print("-" * 40)

    # Fit and transform in one line
    X_final = Featransform(FTconfig.complete()).fit_transform(X_train, y_train)
    print(f"One-liner result shape: {X_final.shape}")

    print("\n" + "=" * 60)
    print("Pipeline execution completed successfully!")
    print("=" * 60)


"""
## Added example of custom configuration:

# Start from a base and modify
def my_custom_config(task_type):
    config = FTconfig.standard(task_type=task_type)
    
    # Your customizations
    config.optimization.metric = OptimizationMetric.F1
    config.processing.imputation_strategy = ImputationStrategy.KNN
    config.anomaly_models = [
        ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)
    ]
    
    return config

# Use it
config = my_custom_config("binary_classification")
pipeline = Featransform(config)
"""














