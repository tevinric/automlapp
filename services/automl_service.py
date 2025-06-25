
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import warnings
import os
import pickle
from datetime import datetime
import joblib

# PyCaret imports
try:
    import pycaret
    from pycaret.classification import *
    from pycaret.regression import *
    from pycaret.clustering import *
    from pycaret.anomaly import *
    from pycaret.nlp import *
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    logging.warning("PyCaret not available. Please install it with: pip install pycaret")

# Additional ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoMLService:
    """Service for automated machine learning using PyCaret"""
    
    def __init__(self):
        self.experiment = None
        self.models = {}
        self.best_model = None
        self.current_setup = None
        
        if not PYCARET_AVAILABLE:
            raise ImportError("PyCaret is required for AutoML functionality")
    
    def preview_preprocessing(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Preview preprocessing steps without running full AutoML
        
        Args:
            data: Input DataFrame
            config: Preprocessing configuration
            
        Returns:
            pd.DataFrame: Preprocessed data preview
        """
        try:
            df = data.copy()
            target_col = config.get('target_column')
            
            # Separate features and target
            if target_col and target_col in df.columns:
                X = df.drop(columns=[target_col])
                y = df[target_col]
            else:
                X = df
                y = None
            
            # Handle missing values
            if config.get('imputation', False):
                method = config.get('imputation_method', 'mean')
                if method in ['mean', 'median']:
                    # Numeric columns only
                    numeric_cols = X.select_dtypes(include=[np.number]).columns
                    if method == 'mean':
                        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
                    else:
                        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
                    
                    # Categorical columns with mode
                    cat_cols = X.select_dtypes(include=['object']).columns
                    for col in cat_cols:
                        X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 'Unknown')
                
                elif method == 'mode':
                    for col in X.columns:
                        X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 'Unknown')
            
            # Normalization
            if config.get('normalize', False):
                method = config.get('normalization_method', 'zscore')
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    if method == 'zscore':
                        scaler = StandardScaler()
                    elif method == 'minmax':
                        scaler = MinMaxScaler()
                    elif method == 'robust':
                        scaler = RobustScaler()
                    elif method == 'maxabs':
                        scaler = MaxAbsScaler()
                    
                    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            
            # Sampling
            if config.get('apply_sampling', False):
                sample_size = config.get('sample_size', 100)
                if sample_size < 100:
                    n_samples = int(len(X) * sample_size / 100)
                    if y is not None:
                        X, _, y, _ = train_test_split(X, y, train_size=n_samples, random_state=42, stratify=y if len(y.unique()) < 20 else None)
                    else:
                        X = X.sample(n=n_samples, random_state=42)
            
            # Combine back with target if exists
            if y is not None:
                result = X.copy()
                result[target_col] = y
            else:
                result = X
            
            logger.info(f"Preprocessing preview completed. Shape: {result.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Error in preprocessing preview: {str(e)}")
            raise e
    
    def setup_experiment(self, data: pd.DataFrame, target_column: str, 
                        problem_type: str, config: Dict[str, Any]) -> Any:
        """
        Setup PyCaret experiment
        
        Args:
            data: Input DataFrame
            target_column: Target column name
            problem_type: Type of ML problem
            config: Configuration dictionary
            
        Returns:
            PyCaret experiment setup
        """
        try:
            # Prepare setup parameters
            setup_params = {
                'data': data,
                'target': target_column,
                'session_id': config.get('session_id', 123),
                'train_size': config.get('train_size', 0.8),
                'silent': config.get('silent', True),
                'profile': config.get('profile', False)
            }
            
            # Add preprocessing parameters
            preprocessing_config = config.get('preprocessing_config', {})
            
            if preprocessing_config.get('normalize'):
                setup_params['normalize'] = True
                setup_params['normalize_method'] = preprocessing_config.get('normalization_method', 'zscore')
            
            if preprocessing_config.get('imputation'):
                setup_params['imputation_type'] = preprocessing_config.get('imputation_method', 'simple')
            
            if preprocessing_config.get('fix_imbalance'):
                setup_params['fix_imbalance'] = True
                setup_params['fix_imbalance_method'] = preprocessing_config.get('imbalance_method', 'smote')
            
            if preprocessing_config.get('remove_outliers'):
                setup_params['remove_outliers'] = True
                setup_params['outliers_threshold'] = preprocessing_config.get('outlier_threshold', 0.05)
            
            if preprocessing_config.get('feature_selection'):
                setup_params['feature_selection'] = True
                setup_params['feature_selection_method'] = preprocessing_config.get('feature_selection_method', 'univariate')
            
            # Setup experiment based on problem type
            if problem_type == 'classification':
                setup_result = setup(**setup_params)
            elif problem_type == 'regression':
                setup_result = setup(**setup_params)
            elif problem_type == 'clustering':
                # Remove target for clustering
                setup_params.pop('target', None)
                setup_result = setup(**setup_params)
            elif problem_type == 'anomaly':
                setup_params.pop('target', None)
                setup_result = setup(**setup_params)
            elif problem_type == 'nlp':
                setup_result = setup(**setup_params)
            else:
                raise ValueError(f"Unsupported problem type: {problem_type}")
            
            self.current_setup = setup_result
            logger.info(f"PyCaret experiment setup completed for {problem_type}")
            return setup_result
            
        except Exception as e:
            logger.error(f"Error setting up experiment: {str(e)}")
            raise e
    
    def train_models(self, data: pd.DataFrame, preprocessing_config: Dict[str, Any], 
                    ml_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train multiple models and return results
        
        Args:
            data: Input DataFrame
            preprocessing_config: Preprocessing configuration
            ml_config: ML configuration
            
        Returns:
            Dict: Training results including leaderboard and models
        """
        try:
            target_column = preprocessing_config.get('target_column')
            problem_type = ml_config.get('problem_type')
            
            # Combine configs
            config = {**ml_config, 'preprocessing_config': preprocessing_config}
            
            # Setup experiment
            setup_result = self.setup_experiment(data, target_column, problem_type, config)
            
            results = {'setup': setup_result}
            
            if problem_type in ['classification', 'regression']:
                # Compare models
                selected_models = ml_config.get('selected_models')
                sort_metric = ml_config.get('sort_metric', 'Accuracy' if problem_type == 'classification' else 'MAE')
                cv_folds = ml_config.get('cv_folds', 5)
                
                if selected_models:
                    leaderboard = compare_models(
                        include=selected_models,
                        sort=sort_metric,
                        fold=cv_folds,
                        n_select=len(selected_models)
                    )
                else:
                    leaderboard = compare_models(
                        sort=sort_metric,
                        fold=cv_folds
                    )
                
                # Convert leaderboard to DataFrame if it's not already
                if hasattr(leaderboard, 'reset_index'):
                    leaderboard_df = leaderboard.reset_index()
                else:
                    # If it's a list of models, create summary
                    model_results = []
                    for model in (leaderboard if isinstance(leaderboard, list) else [leaderboard]):
                        model_name = str(model).split('(')[0]
                        model_results.append({'Model': model_name})
                    leaderboard_df = pd.DataFrame(model_results)
                
                results['leaderboard'] = leaderboard_df
                results['models'] = leaderboard if isinstance(leaderboard, list) else [leaderboard]
                
                # Get best model
                best_model = leaderboard[0] if isinstance(leaderboard, list) else leaderboard
                self.best_model = best_model
                results['best_model'] = best_model
                
            elif problem_type == 'clustering':
                # Create clustering models
                models = ['kmeans', 'ap', 'meanshift', 'sc', 'hclust', 'dbscan', 'optics']
                clustering_results = []
                
                for model_name in models[:3]:  # Limit to first 3 for performance
                    try:
                        model = create_model(model_name)
                        clustering_results.append({
                            'Model': model_name,
                            'Created': True
                        })
                    except Exception as e:
                        logger.warning(f"Could not create {model_name}: {str(e)}")
                
                results['leaderboard'] = pd.DataFrame(clustering_results)
                results['models'] = clustering_results
                
            elif problem_type == 'anomaly':
                # Create anomaly detection models
                models = ['abod', 'iforest', 'cluster', 'cof', 'histogram', 'knn', 'lof', 'svm', 'pca', 'mcd']
                anomaly_results = []
                
                for model_name in models[:3]:  # Limit to first 3 for performance
                    try:
                        model = create_model(model_name)
                        anomaly_results.append({
                            'Model': model_name,
                            'Created': True
                        })
                    except Exception as e:
                        logger.warning(f"Could not create {model_name}: {str(e)}")
                
                results['leaderboard'] = pd.DataFrame(anomaly_results)
                results['models'] = anomaly_results
            
            logger.info(f"Model training completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise e
    
    def analyze_model(self, model_name: str) -> Dict[str, Any]:
        """
        Analyze a specific model in detail
        
        Args:
            model_name: Name of the model to analyze
            
        Returns:
            Dict: Model analysis results
        """
        try:
            if not self.best_model:
                raise ValueError("No trained model available for analysis")
            
            analysis = {}
            
            # Model evaluation
            try:
                # Evaluate model
                evaluate_model(self.best_model)
                analysis['evaluated'] = True
            except Exception as e:
                logger.warning(f"Could not evaluate model: {str(e)}")
                analysis['evaluated'] = False
            
            # Feature importance (if available)
            try:
                if hasattr(self.best_model, 'feature_importances_'):
                    # Get feature names from setup
                    feature_names = get_config('X_train').columns.tolist()
                    importances = self.best_model.feature_importances_
                    
                    feature_importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    analysis['feature_importance'] = feature_importance_df
                elif hasattr(self.best_model, 'coef_'):
                    # For linear models
                    feature_names = get_config('X_train').columns.tolist()
                    coefficients = self.best_model.coef_
                    
                    if len(coefficients.shape) > 1:
                        coefficients = coefficients[0]  # Take first class for multiclass
                    
                    feature_importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': np.abs(coefficients)
                    }).sort_values('importance', ascending=False)
                    
                    analysis['feature_importance'] = feature_importance_df
            except Exception as e:
                logger.warning(f"Could not get feature importance: {str(e)}")
            
            # Predictions
            try:
                predictions = predict_model(self.best_model)
                analysis['predictions'] = predictions
                
                # Calculate residuals for regression
                if 'Label' in predictions.columns and hasattr(predictions, 'Score'):
                    residuals = predictions['Label'] - predictions['Score']
                    analysis['residuals'] = residuals
                    analysis['predictions_series'] = predictions['Score']
            except Exception as e:
                logger.warning(f"Could not generate predictions: {str(e)}")
            
            logger.info("Model analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in model analysis: {str(e)}")
            raise e
    
    def get_model_metrics(self, model) -> Dict[str, float]:
        """
        Get comprehensive metrics for a model
        
        Args:
            model: Trained model object
            
        Returns:
            Dict: Model metrics
        """
        try:
            metrics = {}
            
            # Get predictions
            predictions = predict_model(model)
            
            if 'Label' in predictions.columns:
                y_true = predictions['Label']
                y_pred = predictions['Score'] if 'Score' in predictions.columns else predictions.iloc[:, -1]
                
                # Check if it's classification or regression
                if len(np.unique(y_true)) < 20:  # Likely classification
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    
                    metrics['accuracy'] = accuracy_score(y_true, y_pred)
                    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
                    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
                    metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
                    
                else:  # Likely regression
                    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                    
                    metrics['mae'] = mean_absolute_error(y_true, y_pred)
                    metrics['mse'] = mean_squared_error(y_true, y_pred)
                    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
                    metrics['r2'] = r2_score(y_true, y_pred)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def create_visualizations(self, model, plot_types: List[str] = None) -> Dict[str, Any]:
        """
        Create various visualizations for the model
        
        Args:
            model: Trained model
            plot_types: List of plot types to generate
            
        Returns:
            Dict: Generated plots
        """
        try:
            plots = {}
            
            if not plot_types:
                # Default plots based on problem type
                if hasattr(self.current_setup, 'target_type'):
                    if self.current_setup.target_type == 'Binary' or self.current_setup.target_type == 'Multiclass':
                        plot_types = ['confusion_matrix', 'auc', 'class_report', 'feature_importance']
                    else:
                        plot_types = ['residuals', 'prediction_error', 'feature_importance']
                else:
                    plot_types = ['feature_importance']
            
            for plot_type in plot_types:
                try:
                    if plot_type == 'confusion_matrix':
                        plot_model(model, plot='confusion_matrix', save=True)
                        plots['confusion_matrix'] = 'Generated'
                    elif plot_type == 'auc':
                        plot_model(model, plot='auc', save=True)
                        plots['auc'] = 'Generated'
                    elif plot_type == 'class_report':
                        plot_model(model, plot='class_report', save=True)
                        plots['class_report'] = 'Generated'
                    elif plot_type == 'feature_importance':
                        plot_model(model, plot='feature', save=True)
                        plots['feature_importance'] = 'Generated'
                    elif plot_type == 'residuals':
                        plot_model(model, plot='residuals', save=True)
                        plots['residuals'] = 'Generated'
                    elif plot_type == 'prediction_error':
                        plot_model(model, plot='prediction_error', save=True)
                        plots['prediction_error'] = 'Generated'
                except Exception as e:
                    logger.warning(f"Could not generate {plot_type} plot: {str(e)}")
            
            return plots
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            return {}
    
    def save_model(self, model, model_name: str, save_path: str = None) -> str:
        """
        Save trained model to disk
        
        Args:
            model: Trained model object
            model_name: Name for the saved model
            save_path: Path to save the model
            
        Returns:
            str: Path where model was saved
        """
        try:
            if not save_path:
                save_path = f"models/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save using PyCaret's save_model function
            saved_path = save_model(model, model_name)
            
            logger.info(f"Model saved successfully at: {saved_path}")
            return saved_path
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise e
    
    def load_model(self, model_path: str):
        """
        Load a saved model
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model object
        """
        try:
            # Load using PyCaret's load_model function
            model = load_model(model_path)
            
            logger.info(f"Model loaded successfully from: {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e
    
    def get_model_interpretation(self, model) -> Dict[str, Any]:
        """
        Get model interpretation using SHAP or other explainability tools
        
        Args:
            model: Trained model
            
        Returns:
            Dict: Model interpretation results
        """
        try:
            interpretation = {}
            
            # Try to use PyCaret's interpret_model function
            try:
                interpret_model(model)
                interpretation['shap_available'] = True
            except Exception as e:
                logger.warning(f"SHAP interpretation not available: {str(e)}")
                interpretation['shap_available'] = False
            
            # Get feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_names = get_config('X_train').columns.tolist()
                importances = model.feature_importances_
                
                interpretation['feature_importance'] = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
            
            return interpretation
            
        except Exception as e:
            logger.error(f"Error in model interpretation: {str(e)}")
            return {}
    
    def get_available_models(self, problem_type: str) -> List[str]:
        """
        Get list of available models for a given problem type
        
        Args:
            problem_type: Type of ML problem
            
        Returns:
            List: Available model names
        """
        try:
            if problem_type == 'classification':
                return ['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 
                       'ridge', 'rf', 'qda', 'ada', 'gbc', 'lda', 'et', 'xgboost', 
                       'lightgbm', 'catboost']
            elif problem_type == 'regression':
                return ['lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 
                       'ard', 'par', 'ransac', 'tr', 'huber', 'kr', 'svm', 'knn', 
                       'dt', 'rf', 'et', 'ada', 'gbr', 'mlp', 'xgboost', 'lightgbm', 
                       'catboost']
            elif problem_type == 'clustering':
                return ['kmeans', 'ap', 'meanshift', 'sc', 'hclust', 'dbscan', 'optics']
            elif problem_type == 'anomaly':
                return ['abod', 'iforest', 'cluster', 'cof', 'histogram', 'knn', 'lof', 
                       'svm', 'pca', 'mcd']
            elif problem_type == 'nlp':
                return ['lda', 'lsi', 'hdp', 'rp', 'nmf']
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return []
