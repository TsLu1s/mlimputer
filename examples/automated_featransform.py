from featransform.pipeline import Featransform
from featransform.core.models import PipelineConfig, ProcessingConfig, OptimizationConfig
from featransform.core.enums import (
    ImputationStrategy, EncodingStrategy, SelectionStrategy, 
    ModelFamily,
)
from featransform.utils.data_generator import DatasetGenerator
from featransform.core.models import ModelConfig
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=Warning)

###################################################### Main Example

print("\n" + "=" * 60)
print("Basic Featransform Example")
print("=" * 60)

# Generate your own dataset for examples
X, y = DatasetGenerator.generate(
    task='multiclass_classification',       # Options: 'regression', 'binary_classification', 'multiclass_classification'
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

###################################################### Configure Pipeline

# Create configuration
config = PipelineConfig(
    task_type="multiclass_classification", # regression, binary_classification OR multiclass_classification
    # Processing configuration
    processing=ProcessingConfig(
        imputation_strategy=ImputationStrategy.ITERATIVE,  # MEAN, MEDIAN, MODE, CONSTANT, KNN
        encoding_strategy=EncodingStrategy.LABEL, 
        handle_datetime=True,                              # Extract year, month, day, hour, cyclic features
        drop_constant=True,                                # Remove features with single value
        drop_duplicates=True                               # Remove identical feature columns
    ),
    
    # Anomaly detection models
    anomaly_models=[
        ModelConfig(
            model_family=ModelFamily.ISOLATION_FOREST,
            parameters={'n_estimators': 300, 'contamination': 0.002}
        ),
        ModelConfig(
            model_family=ModelFamily.LOCAL_OUTLIER_FACTOR,
            parameters={'n_neighbors': 20, 'contamination': 0.002}
        ),
        ModelConfig(
            model_family=ModelFamily.ONE_CLASS_SVM,
            parameters={'nu': 0.05, 'kernel': 'rbf'}
        ),
        ModelConfig(
            model_family=ModelFamily.ELLIPTIC_ENVELOPE,
            parameters={'contamination': 0.002}
        )
    ],
    
    # Clustering models
    clustering_models=[
        ModelConfig(
            model_family=ModelFamily.KMEANS,
            parameters={'n_clusters': 3}
        ),
        ModelConfig(
            model_family=ModelFamily.BIRCH,
            parameters={'threshold': 0.5, 'branching_factor': 50}
        ),
        ModelConfig(
            model_family=ModelFamily.MINI_BATCH_KMEANS,
            parameters={'n_clusters': 3}
        ),
        ModelConfig(
            model_family=ModelFamily.GAUSSIAN_MIXTURE,
            parameters={'n_components': 3}
        )
    ],
    
    # Dimensionality dimensionality models
    dimensionality_models=[
        ModelConfig(
            model_family=ModelFamily.PCA,
            parameters={'n_components': 0.95}  # Keep 95% variance
        ),
        ModelConfig(
            model_family=ModelFamily.TRUNCATED_SVD,
            parameters={'n_components': 6}
        ),
        ModelConfig(
            model_family=ModelFamily.FAST_ICA,
            parameters={'n_components': 6}
        )
    ],
    # Optimization configuration
    optimization=OptimizationConfig(
        selection_strategy=SelectionStrategy.IMPORTANCE,
        #metric=OptimizationMetric.F1       # Auto Detects as default for specific task
        n_iterations=10,
        validation_split=0.3,
        min_features=10
    ),   
    verbose=True,
    n_jobs=-1,
    random_state=42
)

###################################################### Fit & Transform

# Create and fit pipeline
ft = Featransform(config)
ft.fit(X_train, y_train)

# Transform data
X_train_transformed = ft.transform(X_train)
X_test_transformed = ft.transform(X_test)

print(f"\nOriginal features: {X_train.shape[1]}")
print(f"Transformed features: {X_train_transformed.shape[1]}")

###################################################### Get Results

# Get pipeline summary
ft.optimization_report()


