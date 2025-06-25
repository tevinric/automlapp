import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import warnings
import json
import pickle
import joblib
from datetime import datetime
import os

# PyCaret imports
try:
    import pycaret
    from pycaret.classification import *
    from pycaret.regression import *
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    logging.warning("PyCaret not available")

# Hyperparameter optimization libraries
try:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
    from sklearn.metrics import make_scorer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical, Integer
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelService:
    """Service for model fine-tuning and hyperparameter optimization"""
    
    def __init__(self):
        self.current_model = None
        self.best_params = None
        self.tuning_history = []
        
        if not PYCARET_AVAILABLE:
            logger.warning("PyCaret not available. Model service functionality will be limited.")
    
    def tune_hyperparameters(self, model_name: str, search_library: str = "scikit-learn",
                           search_algorithm: str = "random", n_iter: int = 20,
                           custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Tune hyperparameters for a given model
        
        Args:
            model_name: Name of the model to tune
            search_library: Library to use for search
            search_algorithm: Search algorithm to use
            n_iter: Number of iterations
            custom_params: Custom parameter grid
            
        Returns:
            Dict: Tuning results
        """
        try:
            if not PYCARET_AVAILABLE:
                return {"error": "PyCaret not available for hyperparameter tuning"}
            
            # Get default parameter grid for the model
            param_grid = self._get_default_param_grid(model_name)
            
            # Update with custom parameters if provided
            if custom_params:
                param_grid.update(custom_params)
            
            logger.info(f"Starting hyperparameter tuning for {model_name}")
            logger.info(f"Using {search_library} with {search_algorithm} search")
            logger.info(f"Parameter grid: {param_grid}")
            
            # Perform tuning based on selected library and algorithm
            if search_library == "scikit-learn":
                results = self._tune_with_sklearn(model_name, param_grid, search_algorithm, n_iter)
            elif search_library == "scikit-optimize":
                results = self._tune_with_skopt(model_name, param_grid, n_iter)
            elif search_library == "optuna":
                results = self._tune_with_optuna(model_name, param_grid, n_iter)
            elif search_library == "tune-sklearn":
                results = self._tune_with_tune_sklearn(model_name, param_grid, search_algorithm, n_iter)
            else:
                # Fallback to PyCaret's built-in tuning
                results = self._tune_with_pycaret(model_name, param_grid, n_iter)
            
            self.best_params = results.get('best_params', {})
            logger.info(f"Hyperparameter tuning completed. Best score: {results.get('best_score', 'N/A')}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {str(e)}")
            return {"error": str(e)}
    
    def _get_default_param_grid(self, model_name: str) -> Dict[str, List]:
        """Get default parameter grid for a model"""
        
        param_grids = {
            # Classification models
            'rf': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6, 7],
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 70, 100],
                'subsample': [0.8, 0.9, 1.0]
            },
            'catboost': {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5, 7, 9]
            },
            'svm': {
                'C': [0.1, 1, 10, 100, 1000],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            },
            'dt': {
                'max_depth': [3, 5, 7, 10, 15, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10],
                'criterion': ['gini', 'entropy']
            },
            'nb': {
                'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
                'fit_prior': [True, False]
            },
            'lr': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'solver': ['liblinear', 'saga', 'lbfgs']
            },
            'ada': {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.5, 1.0, 2.0],
                'algorithm': ['SAMME', 'SAMME.R']
            },
            'gbc': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'max_depth': [3, 4, 5, 6, 7],
                'subsample': [0.8, 0.9, 1.0],
                'max_features': ['auto', 'sqrt', 'log2']
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100), (200, 100)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate': ['constant', 'invscaling', 'adaptive']
            },
            
            # Regression models (similar parameters but adapted for regression)
            'lasso': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'max_iter': [1000, 2000, 5000]
            },
            'ridge': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            },
            'en': {  # Elastic Net
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'max_iter': [1000, 2000, 5000]
            },
            'gbr': {  # Gradient Boosting Regressor
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'max_depth': [3, 4, 5, 6, 7],
                'subsample': [0.8, 0.9, 1.0],
                'max_features': ['auto', 'sqrt', 'log2']
            }
        }
        
        return param_grids.get(model_name, {})
    
    def _tune_with_sklearn(self, model_name: str, param_grid: Dict[str, List],
                          search_algorithm: str, n_iter: int) -> Dict[str, Any]:
        """Tune using scikit-learn's GridSearchCV or RandomizedSearchCV"""
        try:
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn not available")
            
            # Create model using PyCaret
            model = create_model(model_name)
            
            # Get training data
            X_train = get_config('X_train')
            y_train = get_config('y_train')
            
            # Choose search method
            if search_algorithm == "grid":
                search = GridSearchCV(
                    model, 
                    param_grid, 
                    cv=5, 
                    scoring='accuracy' if hasattr(y_train, 'nunique') and y_train.nunique() < 20 else 'neg_mean_squared_error',
                    n_jobs=-1
                )
            else:  # random search
                search = RandomizedSearchCV(
                    model, 
                    param_grid, 
                    n_iter=n_iter,
                    cv=5, 
                    scoring='accuracy' if hasattr(y_train, 'nunique') and y_train.nunique() < 20 else 'neg_mean_squared_error',
                    n_jobs=-1,
                    random_state=42
                )
            
            # Fit the search
            search.fit(X_train, y_train)
            
            # Prepare results
            results = {
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'cv_results': search.cv_results_,
                'search_method': f"sklearn_{search_algorithm}",
                'n_iterations': len(search.cv_results_['params'])
            }
            
            # Create tuning history
            if hasattr(search, 'cv_results_'):
                scores = search.cv_results_['mean_test_score']
                results['tuning_history'] = pd.DataFrame({
                    'iteration': range(len(scores)),
                    'score': scores
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in sklearn tuning: {str(e)}")
            return {"error": str(e)}
    
    def _tune_with_skopt(self, model_name: str, param_grid: Dict[str, List], n_iter: int) -> Dict[str, Any]:
        """Tune using scikit-optimize (Bayesian optimization)"""
        try:
            if not SKOPT_AVAILABLE:
                raise ImportError("scikit-optimize not available")
            
            # Create model using PyCaret
            model = create_model(model_name)
            
            # Get training data
            X_train = get_config('X_train')
            y_train = get_config('y_train')
            
            # Convert parameter grid to skopt format
            search_spaces = {}
            for param, values in param_grid.items():
                if isinstance(values[0], (int, np.integer)):
                    search_spaces[param] = Integer(min(values), max(values))
                elif isinstance(values[0], (float, np.floating)):
                    search_spaces[param] = Real(min(values), max(values))
                else:
                    search_spaces[param] = Categorical(values)
            
            # Bayesian search
            search = BayesSearchCV(
                model,
                search_spaces,
                n_iter=n_iter,
                cv=5,
                scoring='accuracy' if hasattr(y_train, 'nunique') and y_train.nunique() < 20 else 'neg_mean_squared_error',
                n_jobs=-1,
                random_state=42
            )
            
            # Fit the search
            search.fit(X_train, y_train)
            
            # Prepare results
            results = {
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'search_method': 'skopt_bayesian',
                'n_iterations': n_iter
            }
            
            # Create tuning history from optimizer results
            if hasattr(search, 'optimizer_results_'):
                func_vals = search.optimizer_results_[0].func_vals
                results['tuning_history'] = pd.DataFrame({
                    'iteration': range(len(func_vals)),
                    'score': -np.array(func_vals)  # Convert back from negative
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in skopt tuning: {str(e)}")
            return {"error": str(e)}
    
    def _tune_with_optuna(self, model_name: str, param_grid: Dict[str, List], n_iter: int) -> Dict[str, Any]:
        """Tune using Optuna"""
        try:
            if not OPTUNA_AVAILABLE:
                raise ImportError("Optuna not available")
            
            # Get training data
            X_train = get_config('X_train')
            y_train = get_config('y_train')
            
            # Define objective function
            def objective(trial):
                # Suggest parameters
                params = {}
                for param, values in param_grid.items():
                    if isinstance(values[0], (int, np.integer)):
                        params[param] = trial.suggest_int(param, min(values), max(values))
                    elif isinstance(values[0], (float, np.floating)):
                        params[param] = trial.suggest_float(param, min(values), max(values))
                    else:
                        params[param] = trial.suggest_categorical(param, values)
                
                # Create model with suggested parameters
                model = create_model(model_name, **params)
                
                # Cross-validation score
                scores = cross_val_score(
                    model, X_train, y_train, cv=5,
                    scoring='accuracy' if hasattr(y_train, 'nunique') and y_train.nunique() < 20 else 'neg_mean_squared_error'
                )
                
                return scores.mean()
            
            # Create study and optimize
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            study.optimize(objective, n_trials=n_iter)
            
            # Prepare results
            results = {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'search_method': 'optuna_tpe',
                'n_iterations': n_iter
            }
            
            # Create tuning history
            trials_df = study.trials_dataframe()
            if not trials_df.empty:
                results['tuning_history'] = pd.DataFrame({
                    'iteration': range(len(trials_df)),
                    'score': trials_df['value'].values
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Optuna tuning: {str(e)}")
            return {"error": str(e)}
    
    def _tune_with_tune_sklearn(self, model_name: str, param_grid: Dict[str, List],
                               search_algorithm: str, n_iter: int) -> Dict[str, Any]:
        """Tune using tune-sklearn (Ray Tune integration)"""
        try:
            # This would require Ray Tune installation
            # For now, fallback to regular sklearn approach
            logger.warning("tune-sklearn not implemented, falling back to sklearn")
            return self._tune_with_sklearn(model_name, param_grid, search_algorithm, n_iter)
            
        except Exception as e:
            logger.error(f"Error in tune-sklearn: {str(e)}")
            return {"error": str(e)}
    
    def _tune_with_pycaret(self, model_name: str, param_grid: Dict[str, List], n_iter: int) -> Dict[str, Any]:
        """Tune using PyCaret's built-in tuning"""
        try:
            # Create base model
            model = create_model(model_name)
            
            # Tune model
            tuned_model = tune_model(
                model,
                custom_grid=param_grid if param_grid else None,
                n_iter=n_iter,
                optimize='Accuracy' if hasattr(get_config('y_train'), 'nunique') and get_config('y_train').nunique() < 20 else 'MAE'
            )
            
            # Get tuning results
            results = {
                'tuned_model': tuned_model,
                'search_method': 'pycaret_randomized',
                'n_iterations': n_iter
            }
            
            # Try to extract best parameters (might not be directly available)
            if hasattr(tuned_model, 'get_params'):
                results['best_params'] = tuned_model.get_params()
            
            return results
            
        except Exception as e:
            logger.error(f"Error in PyCaret tuning: {str(e)}")
            return {"error": str(e)}
    
    def evaluate_tuned_model(self, model, X_test=None, y_test=None) -> Dict[str, Any]:
        """
        Evaluate the tuned model
        
        Args:
            model: Tuned model object
            X_test: Test features (optional, will use holdout if not provided)
            y_test: Test target (optional, will use holdout if not provided)
            
        Returns:
            Dict: Evaluation results
        """
        try:
            if X_test is None or y_test is None:
                # Use PyCaret's holdout set
                predictions = predict_model(model)
                return {
                    'predictions': predictions,
                    'evaluation_method': 'holdout'
                }
            else:
                # Use provided test set
                predictions = model.predict(X_test)
                
                # Calculate metrics
                if hasattr(y_test, 'nunique') and y_test.nunique() < 20:  # Classification
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    
                    metrics = {
                        'accuracy': accuracy_score(y_test, predictions),
                        'precision': precision_score(y_test, predictions, average='weighted'),
                        'recall': recall_score(y_test, predictions, average='weighted'),
                        'f1': f1_score(y_test, predictions, average='weighted')
                    }
                else:  # Regression
                    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                    
                    metrics = {
                        'mae': mean_absolute_error(y_test, predictions),
                        'mse': mean_squared_error(y_test, predictions),
                        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                        'r2': r2_score(y_test, predictions)
                    }
                
                return {
                    'predictions': predictions,
                    'metrics': metrics,
                    'evaluation_method': 'custom_test_set'
                }
                
        except Exception as e:
            logger.error(f"Error evaluating tuned model: {str(e)}")
            return {"error": str(e)}
    
    def compare_models_performance(self, original_model, tuned_model) -> Dict[str, Any]:
        """
        Compare performance between original and tuned models
        
        Args:
            original_model: Original model
            tuned_model: Tuned model
            
        Returns:
            Dict: Comparison results
        """
        try:
            # Evaluate both models
            original_results = evaluate_model(original_model, fold=5)
            tuned_results = evaluate_model(tuned_model, fold=5)
            
            comparison = {
                'original_performance': original_results,
                'tuned_performance': tuned_results,
                'improvement': {}
            }
            
            # Calculate improvement (this is simplified - actual implementation would depend on PyCaret's output format)
            if hasattr(original_results, 'mean') and hasattr(tuned_results, 'mean'):
                for metric in original_results.columns:
                    original_score = original_results[metric].mean()
                    tuned_score = tuned_results[metric].mean()
                    
                    # Calculate percentage improvement
                    if original_score != 0:
                        improvement = ((tuned_score - original_score) / abs(original_score)) * 100
                        comparison['improvement'][metric] = improvement
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            return {"error": str(e)}
    
    def save_tuned_model(self, model, model_name: str, tuning_results: Dict[str, Any], 
                        save_path: str = None) -> str:
        """
        Save tuned model and its results
        
        Args:
            model: Tuned model object
            model_name: Name for the saved model
            tuning_results: Results from hyperparameter tuning
            save_path: Path to save the model
            
        Returns:
            str: Path where model was saved
        """
        try:
            if not save_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = f"models/tuned_{model_name}_{timestamp}"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save model using PyCaret
            model_path = save_model(model, f"tuned_{model_name}")
            
            # Save tuning results
            results_path = f"{save_path}_tuning_results.json"
            with open(results_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = {}
                for key, value in tuning_results.items():
                    if isinstance(value, np.ndarray):
                        json_results[key] = value.tolist()
                    elif isinstance(value, pd.DataFrame):
                        json_results[key] = value.to_dict()
                    elif hasattr(value, '__dict__'):
                        # Skip complex objects that can't be serialized
                        continue
                    else:
                        json_results[key] = value
                
                json.dump(json_results, f, indent=2, default=str)
            
            logger.info(f"Tuned model and results saved to: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error saving tuned model: {str(e)}")
            raise e
    
    def load_tuned_model(self, model_path: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a saved tuned model and its results
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Tuple: (model, tuning_results)
        """
        try:
            # Load model using PyCaret
            model = load_model(model_path)
            
            # Load tuning results
            results_path = f"{model_path}_tuning_results.json"
            tuning_results = {}
            
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    tuning_results = json.load(f)
            
            logger.info(f"Tuned model loaded from: {model_path}")
            return model, tuning_results
            
        except Exception as e:
            logger.error(f"Error loading tuned model: {str(e)}")
            raise e
    
    def get_tuning_recommendations(self, model_name: str, current_performance: float,
                                 target_performance: float = None) -> Dict[str, Any]:
        """
        Get recommendations for further tuning
        
        Args:
            model_name: Name of the model
            current_performance: Current model performance
            target_performance: Target performance (optional)
            
        Returns:
            Dict: Tuning recommendations
        """
        try:
            recommendations = {
                'model': model_name,
                'current_performance': current_performance,
                'recommendations': []
            }
            
            # General recommendations based on model type
            if model_name in ['rf', 'et']:
                recommendations['recommendations'].extend([
                    "Try increasing n_estimators for potentially better performance (at cost of training time)",
                    "Experiment with max_features to control overfitting",
                    "Tune min_samples_split and min_samples_leaf for regularization"
                ])
            
            elif model_name in ['xgboost', 'lightgbm', 'catboost']:
                recommendations['recommendations'].extend([
                    "Fine-tune learning_rate and n_estimators together",
                    "Adjust max_depth to control model complexity",
                    "Try different regularization parameters"
                ])
            
            elif model_name == 'svm':
                recommendations['recommendations'].extend([
                    "Grid search over C and gamma parameters",
                    "Try different kernels (RBF, polynomial, sigmoid)",
                    "Consider feature scaling if not already done"
                ])
            
            elif model_name in ['lr', 'lasso', 'ridge']:
                recommendations['recommendations'].extend([
                    "Tune regularization parameter (C for logistic, alpha for lasso/ridge)",
                    "Try different solvers for optimization",
                    "Consider feature polynomial features or interactions"
                ])
            
            # Performance-based recommendations
            if target_performance and current_performance < target_performance:
                gap = target_performance - current_performance
                if gap > 0.1:  # Large gap
                    recommendations['recommendations'].append(
                        "Consider ensemble methods or more complex models"
                    )
                    recommendations['recommendations'].append(
                        "Feature engineering might be needed to reach target performance"
                    )
                else:  # Small gap
                    recommendations['recommendations'].append(
                        "Fine-tune current model hyperparameters more extensively"
                    )
                    recommendations['recommendations'].append(
                        "Try cross-validation with different random seeds"
                    )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting tuning recommendations: {str(e)}")
            return {"error": str(e)}
