import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import json
import requests
from datetime import datetime
import warnings

# Azure OpenAI
try:
    import openai
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI package not available. Please install it with: pip install openai")

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService:
    """Service for integrating with Azure OpenAI GPT-4o for ML assistance"""
    
    def __init__(self):
        self.client = None
        self.model_name = "gpt-4o"  # Azure OpenAI deployment name
        self.max_tokens = 4000
        self.temperature = 0.7
        
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI package not available. LLM features will be limited.")
            return
        
        # Initialize Azure OpenAI client
        try:
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            logger.info("Azure OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            self.client = None
    
    def _make_api_call(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Make API call to Azure OpenAI
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters
            
        Returns:
            str: API response content
        """
        try:
            if not self.client:
                return "LLM service not available. Please check your Azure OpenAI configuration."
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                top_p=kwargs.get('top_p', 0.95),
                frequency_penalty=kwargs.get('frequency_penalty', 0),
                presence_penalty=kwargs.get('presence_penalty', 0)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error making API call: {str(e)}")
            return f"Error communicating with AI service: {str(e)}"
    
    def analyze_data(self, data: pd.DataFrame) -> str:
        """
        Analyze dataset and provide insights
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            str: AI analysis of the data
        """
        try:
            # Prepare data summary
            data_summary = self._prepare_data_summary(data)
            
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert data scientist assistant. Analyze the provided dataset summary and provide insights about:
                    1. Data quality and potential issues
                    2. Patterns and distributions in the data
                    3. Recommendations for preprocessing
                    4. Potential ML problems this data could solve
                    5. Key insights and observations
                    
                    Be specific, actionable, and explain technical concepts clearly."""
                },
                {
                    "role": "user",
                    "content": f"Please analyze this dataset:\n\n{data_summary}"
                }
            ]
            
            return self._make_api_call(messages)
            
        except Exception as e:
            logger.error(f"Error in data analysis: {str(e)}")
            return f"Error analyzing data: {str(e)}"
    
    def get_preprocessing_advice(self, data: pd.DataFrame, target_column: str) -> str:
        """
        Get preprocessing recommendations
        
        Args:
            data: DataFrame to analyze
            target_column: Target column name
            
        Returns:
            str: Preprocessing recommendations
        """
        try:
            data_summary = self._prepare_data_summary(data)
            target_info = self._analyze_target_column(data, target_column)
            
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert in data preprocessing for machine learning. Provide specific recommendations for:
                    1. Handling missing values (which methods are best for this data)
                    2. Feature scaling/normalization (which method to choose)
                    3. Handling categorical variables
                    4. Dealing with outliers
                    5. Feature selection strategies
                    6. Class imbalance (if applicable)
                    7. Data sampling considerations
                    
                    Explain why each recommendation is suitable for this specific dataset."""
                },
                {
                    "role": "user",
                    "content": f"Dataset summary:\n{data_summary}\n\nTarget column analysis:\n{target_info}\n\nWhat preprocessing steps do you recommend?"
                }
            ]
            
            return self._make_api_call(messages)
            
        except Exception as e:
            logger.error(f"Error getting preprocessing advice: {str(e)}")
            return f"Error getting preprocessing advice: {str(e)}"
    
    def get_ml_config_advice(self, data: pd.DataFrame, preprocessing_config: Dict[str, Any], 
                           ml_config: Dict[str, Any]) -> str:
        """
        Get ML configuration recommendations
        
        Args:
            data: DataFrame
            preprocessing_config: Preprocessing configuration
            ml_config: ML configuration
            
        Returns:
            str: ML configuration advice
        """
        try:
            data_summary = self._prepare_data_summary(data)
            
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert machine learning engineer. Analyze the dataset and configurations to provide advice on:
                    1. Is the chosen problem type appropriate?
                    2. Recommended models for this type of data
                    3. Cross-validation strategy
                    4. Evaluation metrics selection
                    5. Potential challenges and solutions
                    6. Expected performance ranges
                    
                    Consider the data characteristics, size, and business context."""
                },
                {
                    "role": "user",
                    "content": f"""Dataset summary:\n{data_summary}
                    
                    Preprocessing config:\n{json.dumps(preprocessing_config, indent=2)}
                    
                    ML config:\n{json.dumps(ml_config, indent=2)}
                    
                    Please provide recommendations for the ML configuration."""
                }
            ]
            
            return self._make_api_call(messages)
            
        except Exception as e:
            logger.error(f"Error getting ML config advice: {str(e)}")
            return f"Error getting ML config advice: {str(e)}"
    
    def analyze_model_results(self, results: Dict[str, Any]) -> str:
        """
        Analyze model training results
        
        Args:
            results: Training results dictionary
            
        Returns:
            str: Analysis of model results
        """
        try:
            # Prepare results summary
            results_summary = self._prepare_results_summary(results)
            
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert ML model evaluator. Analyze the model training results and provide:
                    1. Performance interpretation (what do the metrics mean?)
                    2. Model comparison and ranking explanation
                    3. Best model recommendation with reasoning
                    4. Potential overfitting/underfitting insights
                    5. Recommendations for improvement
                    6. Business impact assessment
                    
                    Make technical concepts accessible and provide actionable insights."""
                },
                {
                    "role": "user",
                    "content": f"Model training results:\n{results_summary}\n\nPlease analyze these results and recommend the best approach."
                }
            ]
            
            return self._make_api_call(messages)
            
        except Exception as e:
            logger.error(f"Error analyzing model results: {str(e)}")
            return f"Error analyzing model results: {str(e)}"
    
    def interpret_model_results(self, analysis: Dict[str, Any]) -> str:
        """
        Interpret detailed model analysis results
        
        Args:
            analysis: Model analysis dictionary
            
        Returns:
            str: Interpretation of model results
        """
        try:
            analysis_summary = self._prepare_analysis_summary(analysis)
            
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert in model interpretability. Analyze the model results and explain:
                    1. Feature importance interpretation (what drives predictions?)
                    2. Model behavior insights
                    3. Potential biases or issues
                    4. Business implications of key features
                    5. Recommendations for feature engineering
                    6. Model reliability assessment
                    
                    Focus on practical insights that can drive business decisions."""
                },
                {
                    "role": "user",
                    "content": f"Model analysis results:\n{analysis_summary}\n\nPlease interpret these results and explain what they mean for the business."
                }
            ]
            
            return self._make_api_call(messages)
            
        except Exception as e:
            logger.error(f"Error interpreting model results: {str(e)}")
            return f"Error interpreting model results: {str(e)}"
    
    def get_hyperparameter_recommendations(self, model_name: str) -> str:
        """
        Get hyperparameter tuning recommendations
        
        Args:
            model_name: Name of the model to tune
            
        Returns:
            str: Hyperparameter recommendations
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert in hyperparameter optimization. For the given model, provide:
                    1. Most important hyperparameters to tune
                    2. Recommended value ranges for each parameter
                    3. What each hyperparameter controls
                    4. Tuning strategy (grid search, random search, Bayesian optimization)
                    5. Expected impact on performance
                    6. Computational considerations
                    
                    Prioritize parameters by their impact on model performance."""
                },
                {
                    "role": "user",
                    "content": f"Model: {model_name}\n\nWhat hyperparameters should I tune and what values should I try?"
                }
            ]
            
            return self._make_api_call(messages)
            
        except Exception as e:
            logger.error(f"Error getting hyperparameter recommendations: {str(e)}")
            return f"Error getting hyperparameter recommendations: {str(e)}"
    
    def analyze_tuning_results(self, results: Dict[str, Any]) -> str:
        """
        Analyze hyperparameter tuning results
        
        Args:
            results: Tuning results dictionary
            
        Returns:
            str: Analysis of tuning results
        """
        try:
            tuning_summary = json.dumps(results, indent=2, default=str)
            
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert in hyperparameter optimization analysis. Analyze the tuning results and provide:
                    1. Performance improvement assessment
                    2. Parameter sensitivity analysis
                    3. Optimal parameter combination explanation
                    4. Further tuning recommendations
                    5. Model stability insights
                    6. Production deployment considerations
                    
                    Focus on practical implications and next steps."""
                },
                {
                    "role": "user",
                    "content": f"Hyperparameter tuning results:\n{tuning_summary}\n\nPlease analyze these results and provide insights."
                }
            ]
            
            return self._make_api_call(messages)
            
        except Exception as e:
            logger.error(f"Error analyzing tuning results: {str(e)}")
            return f"Error analyzing tuning results: {str(e)}"
    
    def chat_with_context(self, user_question: str, context: Dict[str, Any]) -> str:
        """
        Chat with context about the ML pipeline
        
        Args:
            user_question: User's question
            context: Context information about the current state
            
        Returns:
            str: AI response
        """
        try:
            context_summary = json.dumps(context, indent=2, default=str)
            
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert ML assistant helping users with their machine learning pipeline. 
                    You have access to information about their current data, preprocessing, and model configurations.
                    
                    Provide helpful, accurate, and actionable advice. If you need more information to answer properly, ask clarifying questions.
                    Always consider the specific context of their project when providing recommendations.
                    
                    Be conversational but professional, and explain technical concepts clearly."""
                },
                {
                    "role": "user",
                    "content": f"Current project context:\n{context_summary}\n\nUser question: {user_question}"
                }
            ]
            
            return self._make_api_call(messages)
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return f"Error processing your question: {str(e)}"
    
    def _prepare_data_summary(self, data: pd.DataFrame) -> str:
        """Prepare a comprehensive data summary for LLM analysis"""
        try:
            summary = f"""Dataset Overview:
- Shape: {data.shape[0]} rows, {data.shape[1]} columns
- Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB

Data Types:
{data.dtypes.value_counts().to_dict()}

Missing Values:
{data.isnull().sum().to_dict()}

Numeric Columns Summary:
{data.describe().to_string() if len(data.select_dtypes(include=[np.number]).columns) > 0 else 'No numeric columns'}

Categorical Columns:
"""
            
            # Add categorical column info
            cat_cols = data.select_dtypes(include=['object']).columns
            for col in cat_cols[:5]:  # Limit to first 5 categorical columns
                unique_count = data[col].nunique()
                summary += f"- {col}: {unique_count} unique values"
                if unique_count <= 10:
                    summary += f" ({data[col].value_counts().head().to_dict()})"
                summary += "\n"
            
            return summary
            
        except Exception as e:
            return f"Error preparing data summary: {str(e)}"
    
    def _analyze_target_column(self, data: pd.DataFrame, target_column: str) -> str:
        """Analyze target column characteristics"""
        try:
            if target_column not in data.columns:
                return f"Target column '{target_column}' not found in data"
            
            target = data[target_column]
            
            analysis = f"""Target Column Analysis ({target_column}):
- Data type: {target.dtype}
- Non-null count: {target.count()}
- Null count: {target.isnull().sum()}
- Unique values: {target.nunique()}
"""
            
            if target.dtype in ['int64', 'float64']:
                analysis += f"""- Min: {target.min()}
- Max: {target.max()}
- Mean: {target.mean():.4f}
- Std: {target.std():.4f}
- Skewness: {target.skew():.4f}
"""
            else:
                value_counts = target.value_counts().head(10)
                analysis += f"- Value distribution:\n{value_counts.to_dict()}"
            
            return analysis
            
        except Exception as e:
            return f"Error analyzing target column: {str(e)}"
    
    def _prepare_results_summary(self, results: Dict[str, Any]) -> str:
        """Prepare model results summary for LLM analysis"""
        try:
            summary = "Model Training Results:\n"
            
            if 'leaderboard' in results:
                leaderboard = results['leaderboard']
                summary += f"\nLeaderboard (top 5 models):\n"
                summary += leaderboard.head().to_string()
                
                if len(leaderboard) > 0:
                    best_model = leaderboard.iloc[0]
                    summary += f"\n\nBest Model: {best_model.iloc[0]}"
                    if len(best_model) > 1:
                        summary += f" with score: {best_model.iloc[1]:.4f}"
            
            if 'models' in results:
                summary += f"\n\nTotal models trained: {len(results['models'])}"
            
            return summary
            
        except Exception as e:
            return f"Error preparing results summary: {str(e)}"
    
    def _prepare_analysis_summary(self, analysis: Dict[str, Any]) -> str:
        """Prepare model analysis summary for LLM interpretation"""
        try:
            summary = "Model Analysis Results:\n"
            
            if 'feature_importance' in analysis:
                feature_imp = analysis['feature_importance']
                summary += f"\nTop 10 Important Features:\n"
                summary += feature_imp.head(10).to_string()
            
            if 'predictions' in analysis:
                predictions = analysis['predictions']
                summary += f"\n\nPredictions shape: {predictions.shape}"
                if 'Label' in predictions.columns and 'Score' in predictions.columns:
                    # Calculate basic metrics
                    from sklearn.metrics import mean_absolute_error, r2_score
                    try:
                        mae = mean_absolute_error(predictions['Label'], predictions['Score'])
                        r2 = r2_score(predictions['Label'], predictions['Score'])
                        summary += f"\nMAE: {mae:.4f}, R²: {r2:.4f}"
                    except:
                        pass
            
            if 'residuals' in analysis:
                residuals = analysis['residuals']
                summary += f"\n\nResiduals statistics:"
                summary += f"\nMean: {residuals.mean():.4f}"
                summary += f"\nStd: {residuals.std():.4f}"
                summary += f"\nMin: {residuals.min():.4f}"
                summary += f"\nMax: {residuals.max():.4f}"
            
            return summary
            
        except Exception as e:
            return f"Error preparing analysis summary: {str(e)}"
    
    def get_feature_engineering_suggestions(self, data: pd.DataFrame, target_column: str) -> str:
        """
        Get feature engineering suggestions
        
        Args:
            data: DataFrame to analyze
            target_column: Target column name
            
        Returns:
            str: Feature engineering suggestions
        """
        try:
            data_summary = self._prepare_data_summary(data)
            
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert in feature engineering. Analyze the dataset and suggest:
                    1. New features that could be created from existing ones
                    2. Feature transformations (log, sqrt, polynomial, etc.)
                    3. Feature interactions that might be valuable
                    4. Time-based features (if applicable)
                    5. Categorical feature encoding strategies
                    6. Text feature extraction methods (if applicable)
                    
                    Focus on features that are likely to improve model performance for the given target variable."""
                },
                {
                    "role": "user",
                    "content": f"Dataset:\n{data_summary}\n\nTarget: {target_column}\n\nWhat new features should I create?"
                }
            ]
            
            return self._make_api_call(messages)
            
        except Exception as e:
            logger.error(f"Error getting feature engineering suggestions: {str(e)}")
            return f"Error getting feature engineering suggestions: {str(e)}"
