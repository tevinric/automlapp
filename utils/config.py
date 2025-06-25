import os
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class for the AutoML application"""
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY: str = None
    AZURE_OPENAI_ENDPOINT: str = None
    AZURE_OPENAI_API_VERSION: str = "2024-02-01"
    AZURE_OPENAI_DEPLOYMENT_NAME: str = "gpt-4o"
    
    # Database Configuration
    SQL_SERVER_DRIVER: str = "ODBC Driver 17 for SQL Server"
    
    # Model Storage Configuration
    MODEL_STORAGE_PATH: str = "models"
    EXPERIMENT_LOGS_PATH: str = "logs"
    
    # Application Configuration
    APP_TITLE: str = "AutoML Pipeline Assistant"
    MAX_UPLOAD_SIZE_MB: int = 500
    ALLOWED_FILE_TYPES: list = None
    
    # ML Configuration Defaults
    DEFAULT_CV_FOLDS: int = 5
    DEFAULT_TRAIN_SIZE: float = 0.8
    DEFAULT_RANDOM_SEED: int = 123
    
    # Performance Configuration
    MAX_MODELS_TO_COMPARE: int = 20
    DEFAULT_TUNING_ITERATIONS: int = 20
    
    def __post_init__(self):
        if self.ALLOWED_FILE_TYPES is None:
            self.ALLOWED_FILE_TYPES = ['csv', 'xlsx', 'xls']
        
        # Load from environment variables
        self.load_from_env()
        
        # Create directories
        self.create_directories()
    
    def load_from_env(self):
        """Load configuration from environment variables"""
        self.AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", self.AZURE_OPENAI_API_KEY)
        self.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", self.AZURE_OPENAI_ENDPOINT)
        self.AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", self.AZURE_OPENAI_API_VERSION)
        self.AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", self.AZURE_OPENAI_DEPLOYMENT_NAME)
        
        # Model storage paths
        self.MODEL_STORAGE_PATH = os.getenv("MODEL_STORAGE_PATH", self.MODEL_STORAGE_PATH)
        self.EXPERIMENT_LOGS_PATH = os.getenv("EXPERIMENT_LOGS_PATH", self.EXPERIMENT_LOGS_PATH)
        
        # Parse numeric values
        try:
            self.MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", self.MAX_UPLOAD_SIZE_MB))
            self.DEFAULT_CV_FOLDS = int(os.getenv("DEFAULT_CV_FOLDS", self.DEFAULT_CV_FOLDS))
            self.DEFAULT_RANDOM_SEED = int(os.getenv("DEFAULT_RANDOM_SEED", self.DEFAULT_RANDOM_SEED))
            self.MAX_MODELS_TO_COMPARE = int(os.getenv("MAX_MODELS_TO_COMPARE", self.MAX_MODELS_TO_COMPARE))
            self.DEFAULT_TUNING_ITERATIONS = int(os.getenv("DEFAULT_TUNING_ITERATIONS", self.DEFAULT_TUNING_ITERATIONS))
        except ValueError as e:
            logger.warning(f"Error parsing numeric configuration: {e}")
        
        try:
            self.DEFAULT_TRAIN_SIZE = float(os.getenv("DEFAULT_TRAIN_SIZE", self.DEFAULT_TRAIN_SIZE))
        except ValueError as e:
            logger.warning(f"Error parsing float configuration: {e}")
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.MODEL_STORAGE_PATH,
            self.EXPERIMENT_LOGS_PATH,
            os.path.join(self.MODEL_STORAGE_PATH, "checkpoints"),
            os.path.join(self.EXPERIMENT_LOGS_PATH, "experiments")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        # Check Azure OpenAI configuration
        if not self.AZURE_OPENAI_API_KEY:
            errors.append("AZURE_OPENAI_API_KEY is not set")
        
        if not self.AZURE_OPENAI_ENDPOINT:
            errors.append("AZURE_OPENAI_ENDPOINT is not set")
        
        # Check numeric ranges
        if self.DEFAULT_CV_FOLDS < 2:
            errors.append("DEFAULT_CV_FOLDS must be at least 2")
        
        if not 0.1 <= self.DEFAULT_TRAIN_SIZE <= 0.9:
            errors.append("DEFAULT_TRAIN_SIZE must be between 0.1 and 0.9")
        
        if self.MAX_UPLOAD_SIZE_MB <= 0:
            errors.append("MAX_UPLOAD_SIZE_MB must be positive")
        
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        try:
            config_dict = self.to_dict()
            # Remove sensitive information
            config_dict.pop('AZURE_OPENAI_API_KEY', None)
            
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file"""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            # Create instance with loaded values
            config = cls()
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            logger.info(f"Configuration loaded from {filepath}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return cls()  # Return default configuration

# Global configuration instance
config = Config()

# Utility functions
def get_config() -> Config:
    """Get the global configuration instance"""
    return config

def validate_file_upload(file, max_size_mb: int = None) -> tuple[bool, str]:
    """
    Validate uploaded file
    
    Args:
        file: Uploaded file object
        max_size_mb: Maximum file size in MB
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if max_size_mb is None:
        max_size_mb = config.MAX_UPLOAD_SIZE_MB
    
    # Check file size
    if hasattr(file, 'size') and file.size > max_size_mb * 1024 * 1024:
        return False, f"File size exceeds {max_size_mb}MB limit"
    
    # Check file extension
    if hasattr(file, 'name'):
        file_extension = file.name.split('.')[-1].lower()
        if file_extension not in config.ALLOWED_FILE_TYPES:
            return False, f"File type '{file_extension}' not allowed. Supported types: {', '.join(config.ALLOWED_FILE_TYPES)}"
    
    return True, ""

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    import logging
    from datetime import datetime
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        if not os.path.isabs(log_file):
            log_file = os.path.join(config.EXPERIMENT_LOGS_PATH, log_file)
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logger.info(f"Logging setup complete. Level: {log_level}")

def get_model_info() -> Dict[str, Any]:
    """Get information about available models"""
    return {
        'classification_models': {
            'lr': 'Logistic Regression',
            'knn': 'K-Nearest Neighbors',
            'nb': 'Naive Bayes',
            'dt': 'Decision Tree',
            'svm': 'Support Vector Machine',
            'rbfsvm': 'RBF Support Vector Machine',
            'gpc': 'Gaussian Process Classifier',
            'mlp': 'Multi-Layer Perceptron',
            'ridge': 'Ridge Classifier',
            'rf': 'Random Forest',
            'qda': 'Quadratic Discriminant Analysis',
            'ada': 'AdaBoost',
            'gbc': 'Gradient Boosting Classifier',
            'lda': 'Linear Discriminant Analysis',
            'et': 'Extra Trees',
            'xgboost': 'Extreme Gradient Boosting',
            'lightgbm': 'Light Gradient Boosting',
            'catboost': 'CatBoost Classifier'
        },
        'regression_models': {
            'lr': 'Linear Regression',
            'lasso': 'Lasso Regression',
            'ridge': 'Ridge Regression',
            'en': 'Elastic Net',
            'lar': 'Least Angle Regression',
            'llar': 'Lasso Least Angle Regression',
            'omp': 'Orthogonal Matching Pursuit',
            'br': 'Bayesian Ridge',
            'ard': 'Automatic Relevance Determination',
            'par': 'Passive Aggressive Regressor',
            'ransac': 'Random Sample Consensus',
            'tr': 'TheilSen Regressor',
            'huber': 'Huber Regressor',
            'kr': 'Kernel Ridge',
            'svm': 'Support Vector Regression',
            'knn': 'K-Nearest Neighbors',
            'dt': 'Decision Tree',
            'rf': 'Random Forest',
            'et': 'Extra Trees',
            'ada': 'AdaBoost',
            'gbr': 'Gradient Boosting',
            'mlp': 'Multi-Layer Perceptron',
            'xgboost': 'Extreme Gradient Boosting',
            'lightgbm': 'Light Gradient Boosting',
            'catboost': 'CatBoost Regressor'
        },
        'clustering_models': {
            'kmeans': 'K-Means Clustering',
            'ap': 'Affinity Propagation',
            'meanshift': 'Mean Shift Clustering',
            'sc': 'Spectral Clustering',
            'hclust': 'Hierarchical Clustering',
            'dbscan': 'Density-Based Clustering',
            'optics': 'OPTICS Clustering'
        },
        'anomaly_models': {
            'abod': 'Angle-Based Outlier Detection',
            'iforest': 'Isolation Forest',
            'cluster': 'Clustering-Based Outlier Detection',
            'cof': 'Connectivity-Based Outlier Factor',
            'histogram': 'Histogram-Based Outlier Detection',
            'knn': 'K-Nearest Neighbors Outlier Detection',
            'lof': 'Local Outlier Factor',
            'svm': 'One-Class SVM',
            'pca': 'Principal Component Analysis',
            'mcd': 'Minimum Covariance Determinant'
        }
    }

def get_preprocessing_options() -> Dict[str, Any]:
    """Get available preprocessing options"""
    return {
        'normalization_methods': {
            'zscore': 'Z-Score Normalization (StandardScaler)',
            'minmax': 'Min-Max Normalization',
            'maxabs': 'Max Absolute Normalization',
            'robust': 'Robust Normalization (RobustScaler)'
        },
        'imputation_methods': {
            'mean': 'Mean Imputation',
            'median': 'Median Imputation',
            'mode': 'Mode Imputation',
            'knn': 'K-Nearest Neighbors Imputation',
            'iterative': 'Iterative Imputation'
        },
        'feature_selection_methods': {
            'univariate': 'Univariate Feature Selection',
            'rfe': 'Recursive Feature Elimination',
            'boruta': 'Boruta Feature Selection'
        },
        'imbalance_methods': {
            'smote': 'SMOTE (Synthetic Minority Oversampling)',
            'adasyn': 'ADASYN (Adaptive Synthetic Sampling)',
            'bordersmote': 'BorderlineSMOTE',
            'randomunder': 'Random Under Sampling'
        }
    }

def get_evaluation_metrics() -> Dict[str, Any]:
    """Get available evaluation metrics"""
    return {
        'classification_metrics': {
            'Accuracy': 'Overall Accuracy',
            'AUC': 'Area Under ROC Curve',
            'Recall': 'Recall (Sensitivity)',
            'Precision': 'Precision',
            'F1': 'F1 Score',
            'Kappa': 'Cohen\'s Kappa',
            'MCC': 'Matthews Correlation Coefficient'
        },
        'regression_metrics': {
            'MAE': 'Mean Absolute Error',
            'MSE': 'Mean Squared Error',
            'RMSE': 'Root Mean Squared Error',
            'R2': 'R-Squared Score',
            'RMSLE': 'Root Mean Squared Log Error',
            'MAPE': 'Mean Absolute Percentage Error'
        }
    }

def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are installed"""
    dependencies = {}
    
    # Core ML libraries
    try:
        import pycaret
        dependencies['pycaret'] = True
    except ImportError:
        dependencies['pycaret'] = False
    
    try:
        import sklearn
        dependencies['sklearn'] = True
    except ImportError:
        dependencies['sklearn'] = False
    
    try:
        import pandas
        dependencies['pandas'] = True
    except ImportError:
        dependencies['pandas'] = False
    
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        dependencies['numpy'] = False
    
    # Optional ML libraries
    try:
        import xgboost
        dependencies['xgboost'] = True
    except ImportError:
        dependencies['xgboost'] = False
    
    try:
        import lightgbm
        dependencies['lightgbm'] = True
    except ImportError:
        dependencies['lightgbm'] = False
    
    try:
        import catboost
        dependencies['catboost'] = True
    except ImportError:
        dependencies['catboost'] = False
    
    # Hyperparameter optimization
    try:
        import optuna
        dependencies['optuna'] = True
    except ImportError:
        dependencies['optuna'] = False
    
    try:
        import skopt
        dependencies['skopt'] = True
    except ImportError:
        dependencies['skopt'] = False
    
    # Database connectivity
    try:
        import pyodbc
        dependencies['pyodbc'] = True
    except ImportError:
        dependencies['pyodbc'] = False
    
    try:
        import sqlalchemy
        dependencies['sqlalchemy'] = True
    except ImportError:
        dependencies['sqlalchemy'] = False
    
    # Azure OpenAI
    try:
        import openai
        dependencies['openai'] = True
    except ImportError:
        dependencies['openai'] = False
    
    # Visualization
    try:
        import plotly
        dependencies['plotly'] = True
    except ImportError:
        dependencies['plotly'] = False
    
    try:
        import streamlit
        dependencies['streamlit'] = True
    except ImportError:
        dependencies['streamlit'] = False
    
    return dependencies

def print_dependency_status():
    """Print the status of all dependencies"""
    dependencies = check_dependencies()
    
    print("Dependency Status:")
    print("=" * 50)
    
    for package, installed in dependencies.items():
        status = "✅ Installed" if installed else "❌ Missing"
        print(f"{package:<15}: {status}")
    
    missing = [pkg for pkg, installed in dependencies.items() if not installed]
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install missing packages with: pip install <package_name>")
    else:
        print("\n✅ All dependencies are installed!")

# Initialize logging
setup_logging()

# Validate configuration on import
if not config.validate():
    logger.warning("Configuration validation failed. Some features may not work properly.")
