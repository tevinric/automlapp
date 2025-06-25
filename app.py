import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
import sys
import traceback
from io import StringIO
import base64

# Import backend services
from services.data_ingestion import DataIngestionService
from services.automl_service import AutoMLService
from services.llm_service import LLMService
from services.model_service import ModelService
from utils.config import Config

# Page configuration
st.set_page_config(
    page_title="AutoML Pipeline Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize services
@st.cache_resource
def initialize_services():
    """Initialize all backend services"""
    try:
        data_service = DataIngestionService()
        automl_service = AutoMLService()
        llm_service = LLMService()
        model_service = ModelService()
        return data_service, automl_service, llm_service, model_service
    except Exception as e:
        st.error(f"Error initializing services: {str(e)}")
        return None, None, None, None

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'ml_results' not in st.session_state:
        st.session_state.ml_results = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'preprocessing_config' not in st.session_state:
        st.session_state.preprocessing_config = {}
    if 'ml_config' not in st.session_state:
        st.session_state.ml_config = {}

def main():
    """Main application function"""
    st.title("🤖 AutoML Pipeline Assistant")
    st.markdown("---")
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize services
    data_service, automl_service, llm_service, model_service = initialize_services()
    
    if not all([data_service, automl_service, llm_service, model_service]):
        st.error("Failed to initialize services. Please check your configuration.")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Data Ingestion", "Data Preprocessing", "AutoML Configuration", 
         "Model Training", "Results Analysis", "Model Fine-tuning", "LLM Assistant"]
    )
    
    # Data Ingestion Page
    if page == "Data Ingestion":
        data_ingestion_page(data_service, llm_service)
    
    # Data Preprocessing Page
    elif page == "Data Preprocessing":
        data_preprocessing_page(automl_service, llm_service)
    
    # AutoML Configuration Page
    elif page == "AutoML Configuration":
        automl_configuration_page(automl_service, llm_service)
    
    # Model Training Page
    elif page == "Model Training":
        model_training_page(automl_service, llm_service)
    
    # Results Analysis Page
    elif page == "Results Analysis":
        results_analysis_page(automl_service, llm_service)
    
    # Model Fine-tuning Page
    elif page == "Model Fine-tuning":
        model_finetuning_page(model_service, llm_service)
    
    # LLM Assistant Page
    elif page == "LLM Assistant":
        llm_assistant_page(llm_service)

def data_ingestion_page(data_service, llm_service):
    """Data ingestion interface"""
    st.header("📊 Data Ingestion")
    
    # Data source selection
    data_source = st.selectbox(
        "Select Data Source:",
        ["Upload CSV/Excel", "SQL Server", "Databricks", "Sample Data"]
    )
    
    if data_source == "Upload CSV/Excel":
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=['csv', 'xlsx', 'xls']
        )
        
        if uploaded_file is not None:
            try:
                with st.spinner("Loading data..."):
                    data = data_service.load_file(uploaded_file)
                    st.session_state.data = data
                    
                st.success(f"Data loaded successfully! Shape: {data.shape}")
                display_data_overview(data, llm_service)
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    elif data_source == "SQL Server":
        sql_connection_form(data_service)
    
    elif data_source == "Databricks":
        databricks_connection_form(data_service)
    
    elif data_source == "Sample Data":
        sample_data_selection(data_service)

def sql_connection_form(data_service):
    """SQL Server connection form"""
    st.subheader("SQL Server Connection")
    
    with st.form("sql_connection"):
        server = st.text_input("Server")
        database = st.text_input("Database")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        query = st.text_area("SQL Query", height=100)
        
        if st.form_submit_button("Connect and Execute"):
            try:
                with st.spinner("Connecting to SQL Server..."):
                    data = data_service.load_from_sql(server, database, username, password, query)
                    st.session_state.data = data
                    st.success(f"Data loaded successfully! Shape: {data.shape}")
                    display_data_overview(data, None)
            except Exception as e:
                st.error(f"Error connecting to SQL Server: {str(e)}")

def databricks_connection_form(data_service):
    """Databricks connection form"""
    st.subheader("Databricks Connection")
    
    with st.form("databricks_connection"):
        server_hostname = st.text_input("Server Hostname")
        http_path = st.text_input("HTTP Path")
        access_token = st.text_input("Access Token", type="password")
        query = st.text_area("SQL Query", height=100)
        
        if st.form_submit_button("Connect and Execute"):
            try:
                with st.spinner("Connecting to Databricks..."):
                    data = data_service.load_from_databricks(server_hostname, http_path, access_token, query)
                    st.session_state.data = data
                    st.success(f"Data loaded successfully! Shape: {data.shape}")
                    display_data_overview(data, None)
            except Exception as e:
                st.error(f"Error connecting to Databricks: {str(e)}")

def sample_data_selection(data_service):
    """Sample data selection"""
    st.subheader("Sample Datasets")
    
    sample_datasets = {
        "Iris Classification": "iris",
        "California Housing Regression": "california_housing",
        "Wine Classification": "wine",
        "Diabetes Regression": "diabetes",
        "Synthetic Housing Regression": "boston"
    }
    
    selected_dataset = st.selectbox("Choose a sample dataset:", list(sample_datasets.keys()))
    
    if st.button("Load Sample Data"):
        try:
            with st.spinner("Loading sample data..."):
                data = data_service.load_sample_data(sample_datasets[selected_dataset])
                st.session_state.data = data
                st.success(f"Sample data loaded successfully! Shape: {data.shape}")
                display_data_overview(data, None)
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")

def display_data_overview(data, llm_service):
    """Display data overview and statistics"""
    st.subheader("Data Overview")
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", data.shape[0])
    with col2:
        st.metric("Columns", data.shape[1])
    with col3:
        st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(data.head(10))
    
    # Data types and statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Types")
        dtype_df = pd.DataFrame({
            'Column': data.columns,
            'Type': data.dtypes.astype(str),
            'Non-Null Count': data.count(),
            'Null Count': data.isnull().sum()
        })
        st.dataframe(dtype_df)
    
    with col2:
        st.subheader("Statistical Summary")
        st.dataframe(data.describe())
    
    # Missing values visualization
    if data.isnull().sum().sum() > 0:
        st.subheader("Missing Values")
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        fig = px.bar(x=missing_data.index, y=missing_data.values, 
                     title="Missing Values by Column")
        fig.update_xaxis(title="Columns")
        fig.update_yaxis(title="Missing Count")
        st.plotly_chart(fig, use_container_width=True)
    
    # LLM Data Analysis
    if llm_service and st.button("🤖 Get AI Data Analysis"):
        with st.spinner("Analyzing data with AI..."):
            try:
                analysis = llm_service.analyze_data(data)
                st.subheader("AI Data Analysis")
                st.markdown(analysis)
            except Exception as e:
                st.error(f"Error in AI analysis: {str(e)}")

def data_preprocessing_page(automl_service, llm_service):
    """Data preprocessing configuration"""
    st.header("🔧 Data Preprocessing")
    
    if st.session_state.data is None:
        st.warning("Please load data first in the Data Ingestion section.")
        return
    
    data = st.session_state.data
    
    # Target variable selection
    st.subheader("Target Variable")
    target_column = st.selectbox("Select target column:", data.columns.tolist())
    
    # Preprocessing options
    st.subheader("Preprocessing Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Normalization
        normalize = st.checkbox("Normalize Features", value=True)
        normalization_method = st.selectbox(
            "Normalization Method:",
            ["zscore", "minmax", "maxabs", "robust"],
            disabled=not normalize
        )
        
        # Handle missing values
        imputation = st.checkbox("Handle Missing Values", value=True)
        imputation_method = st.selectbox(
            "Imputation Method:",
            ["mean", "median", "mode", "knn", "iterative"],
            disabled=not imputation
        )
        
        # Feature selection
        feature_selection = st.checkbox("Feature Selection")
        feature_selection_method = st.selectbox(
            "Feature Selection Method:",
            ["univariate", "rfe", "boruta"],
            disabled=not feature_selection
        )
    
    with col2:
        # Handle imbalanced data
        fix_imbalance = st.checkbox("Fix Imbalanced Data")
        imbalance_method = st.selectbox(
            "Imbalance Handling Method:",
            ["smote", "adasyn", "bordersmote", "randomunder"],
            disabled=not fix_imbalance
        )
        
        # Outlier removal
        remove_outliers = st.checkbox("Remove Outliers")
        outlier_threshold = st.slider(
            "Outlier Threshold:", 0.01, 0.1, 0.05,
            disabled=not remove_outliers
        )
        
        # Sampling
        apply_sampling = st.checkbox("Apply Sampling")
        sample_size = st.slider(
            "Sample Size (%):", 10, 100, 100,
            disabled=not apply_sampling
        )
    
    # Store preprocessing configuration
    preprocessing_config = {
        'target_column': target_column,
        'normalize': normalize,
        'normalization_method': normalization_method if normalize else None,
        'imputation': imputation,
        'imputation_method': imputation_method if imputation else None,
        'feature_selection': feature_selection,
        'feature_selection_method': feature_selection_method if feature_selection else None,
        'fix_imbalance': fix_imbalance,
        'imbalance_method': imbalance_method if fix_imbalance else None,
        'remove_outliers': remove_outliers,
        'outlier_threshold': outlier_threshold if remove_outliers else None,
        'apply_sampling': apply_sampling,
        'sample_size': sample_size if apply_sampling else 100
    }
    
    st.session_state.preprocessing_config = preprocessing_config
    
    # Preview preprocessing
    if st.button("Preview Preprocessing"):
        with st.spinner("Applying preprocessing..."):
            try:
                processed_data = automl_service.preview_preprocessing(data, preprocessing_config)
                
                st.subheader("Preprocessing Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Data Shape:**", data.shape)
                    st.write("**Processed Data Shape:**", processed_data.shape)
                
                with col2:
                    st.write("**Original Missing Values:**", data.isnull().sum().sum())
                    st.write("**Processed Missing Values:**", processed_data.isnull().sum().sum())
                
                st.subheader("Processed Data Preview")
                st.dataframe(processed_data.head())
                
            except Exception as e:
                st.error(f"Error in preprocessing: {str(e)}")
    
    # LLM Preprocessing Advice
    if st.button("🤖 Get AI Preprocessing Advice"):
        with st.spinner("Getting AI advice..."):
            try:
                advice = llm_service.get_preprocessing_advice(data, target_column)
                st.subheader("AI Preprocessing Recommendations")
                st.markdown(advice)
            except Exception as e:
                st.error(f"Error getting AI advice: {str(e)}")

def automl_configuration_page(automl_service, llm_service):
    """AutoML configuration interface"""
    st.header("⚙️ AutoML Configuration")
    
    if st.session_state.data is None:
        st.warning("Please load data first in the Data Ingestion section.")
        return
    
    if not st.session_state.preprocessing_config:
        st.warning("Please configure preprocessing first.")
        return
    
    # ML Problem Type
    st.subheader("Machine Learning Problem Type")
    problem_type = st.selectbox(
        "Select ML Problem Type:",
        ["classification", "regression", "clustering", "anomaly", "nlp"]
    )
    
    # Model selection
    st.subheader("Model Configuration")
    
    if problem_type in ["classification", "regression"]:
        # Cross-validation
        cv_folds = st.slider("Cross-validation Folds:", 3, 10, 5)
        
        # Train-test split
        train_size = st.slider("Training Data Size (%):", 60, 90, 80)
        
        # Model selection
        if problem_type == "classification":
            available_models = [
                'lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 
                'ridge', 'rf', 'qda', 'ada', 'gbc', 'lda', 'et', 'xgboost', 
                'lightgbm', 'catboost'
            ]
        else:
            available_models = [
                'lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 
                'ard', 'par', 'ransac', 'tr', 'huber', 'kr', 'svm', 'knn', 
                'dt', 'rf', 'et', 'ada', 'gbr', 'mlp', 'xgboost', 'lightgbm', 'catboost'
            ]
        
        selected_models = st.multiselect(
            "Select Models to Include:",
            available_models,
            default=available_models[:5]
        )
        
        # Evaluation metrics
        if problem_type == "classification":
            sort_metric = st.selectbox(
                "Primary Evaluation Metric:",
                ["Accuracy", "AUC", "Recall", "Precision", "F1", "Kappa", "MCC"]
            )
        else:
            sort_metric = st.selectbox(
                "Primary Evaluation Metric:",
                ["MAE", "MSE", "RMSE", "R2", "RMSLE", "MAPE"]
            )
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        session_id = st.number_input("Random Seed:", value=123)
        silent = st.checkbox("Silent Mode", value=True)
        profile = st.checkbox("Enable Profiling", value=False)
    
    # Store ML configuration
    ml_config = {
        'problem_type': problem_type,
        'cv_folds': cv_folds if problem_type in ["classification", "regression"] else 5,
        'train_size': train_size / 100 if problem_type in ["classification", "regression"] else 0.8,
        'selected_models': selected_models if problem_type in ["classification", "regression"] else None,
        'sort_metric': sort_metric if problem_type in ["classification", "regression"] else None,
        'session_id': session_id,
        'silent': silent,
        'profile': profile
    }
    
    st.session_state.ml_config = ml_config
    
    # Configuration summary
    st.subheader("Configuration Summary")
    st.json(ml_config)
    
    # LLM Configuration Advice
    if st.button("🤖 Get AI Configuration Advice"):
        with st.spinner("Getting AI advice..."):
            try:
                advice = llm_service.get_ml_config_advice(
                    st.session_state.data, 
                    st.session_state.preprocessing_config,
                    ml_config
                )
                st.subheader("AI Configuration Recommendations")
                st.markdown(advice)
            except Exception as e:
                st.error(f"Error getting AI advice: {str(e)}")

def model_training_page(automl_service, llm_service):
    """Model training interface"""
    st.header("🚀 Model Training")
    
    if not all([st.session_state.data is not None, 
                st.session_state.preprocessing_config,
                st.session_state.ml_config]):
        st.warning("Please complete data ingestion, preprocessing, and AutoML configuration first.")
        return
    
    st.subheader("Training Configuration")
    
    # Display current configuration
    with st.expander("Current Configuration"):
        st.write("**Preprocessing Config:**")
        st.json(st.session_state.preprocessing_config)
        st.write("**ML Config:**")
        st.json(st.session_state.ml_config)
    
    # Training button
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training models... This may take a while."):
            try:
                # Train models
                results = automl_service.train_models(
                    st.session_state.data,
                    st.session_state.preprocessing_config,
                    st.session_state.ml_config
                )
                
                st.session_state.ml_results = results
                st.success("✅ Training completed successfully!")
                
                # Display results
                display_training_results(results, llm_service)
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.error("Full error traceback:")
                st.text(traceback.format_exc())

def display_training_results(results, llm_service):
    """Display training results"""
    st.subheader("🏆 Model Comparison Results")
    
    if 'leaderboard' in results:
        st.dataframe(results['leaderboard'])
        
        # Best model highlight
        best_model = results['leaderboard'].iloc[0]
        st.success(f"🥇 Best Model: {best_model.name} with score: {best_model.iloc[1]:.4f}")
        
        # Visualization
        fig = px.bar(
            results['leaderboard'].head(10), 
            x='Model', 
            y=results['leaderboard'].columns[1],
            title="Top 10 Models Performance"
        )
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # LLM Analysis
    if st.button("🤖 Get AI Model Analysis"):
        with st.spinner("Analyzing results with AI..."):
            try:
                analysis = llm_service.analyze_model_results(results)
                st.subheader("AI Model Analysis")
                st.markdown(analysis)
            except Exception as e:
                st.error(f"Error in AI analysis: {str(e)}")

def results_analysis_page(automl_service, llm_service):
    """Results analysis interface"""
    st.header("📊 Results Analysis")
    
    if st.session_state.ml_results is None:
        st.warning("Please train models first.")
        return
    
    results = st.session_state.ml_results
    
    # Model selection for detailed analysis
    if 'leaderboard' in results:
        model_names = results['leaderboard']['Model'].tolist()
        selected_model_name = st.selectbox("Select model for detailed analysis:", model_names)
        
        if st.button("Analyze Selected Model"):
            with st.spinner("Analyzing selected model..."):
                try:
                    model_analysis = automl_service.analyze_model(selected_model_name)
                    display_model_analysis(model_analysis, llm_service)
                except Exception as e:
                    st.error(f"Error analyzing model: {str(e)}")

def display_model_analysis(analysis, llm_service):
    """Display detailed model analysis"""
    st.subheader("Detailed Model Analysis")
    
    # Feature importance
    if 'feature_importance' in analysis:
        st.subheader("Feature Importance")
        fig = px.bar(
            analysis['feature_importance'],
            x='importance',
            y='feature',
            orientation='h',
            title="Feature Importance"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrix (for classification)
    if 'confusion_matrix' in analysis:
        st.subheader("Confusion Matrix")
        fig = px.imshow(
            analysis['confusion_matrix'],
            title="Confusion Matrix",
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Residuals plot (for regression)
    if 'residuals' in analysis:
        st.subheader("Residuals Plot")
        fig = px.scatter(
            x=analysis['predictions'],
            y=analysis['residuals'],
            title="Residuals vs Predictions"
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    # LLM Interpretation
    if st.button("🤖 Get AI Model Interpretation"):
        with st.spinner("Getting AI interpretation..."):
            try:
                interpretation = llm_service.interpret_model_results(analysis)
                st.subheader("AI Model Interpretation")
                st.markdown(interpretation)
            except Exception as e:
                st.error(f"Error getting interpretation: {str(e)}")

def model_finetuning_page(model_service, llm_service):
    """Model fine-tuning interface"""
    st.header("🔧 Model Fine-tuning")
    
    if st.session_state.ml_results is None:
        st.warning("Please train models first.")
        return
    
    # Model selection
    model_names = st.session_state.ml_results['leaderboard']['Model'].tolist()
    selected_model = st.selectbox("Select model to fine-tune:", model_names)
    
    if st.button("🤖 Get Hyperparameter Recommendations"):
        with st.spinner("Getting AI recommendations..."):
            try:
                recommendations = llm_service.get_hyperparameter_recommendations(selected_model)
                st.subheader("AI Hyperparameter Recommendations")
                st.markdown(recommendations)
            except Exception as e:
                st.error(f"Error getting recommendations: {str(e)}")
    
    # Hyperparameter tuning interface
    st.subheader("Hyperparameter Tuning")
    
    # Grid search parameters
    search_library = st.selectbox(
        "Search Library:",
        ["scikit-learn", "scikit-optimize", "tune-sklearn", "optuna"]
    )
    
    search_algorithm = st.selectbox(
        "Search Algorithm:",
        ["random", "grid", "bayesian"]
    )
    
    n_iter = st.slider("Number of Iterations:", 10, 100, 20)
    
    # Custom hyperparameters (simplified interface)
    st.subheader("Custom Hyperparameters")
    custom_params = st.text_area(
        "Enter custom parameters (JSON format):",
        value='{"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7]}'
    )
    
    if st.button("Start Fine-tuning"):
        with st.spinner("Fine-tuning model..."):
            try:
                # Parse custom parameters
                params = json.loads(custom_params) if custom_params else {}
                
                tuning_results = model_service.tune_hyperparameters(
                    selected_model,
                    search_library,
                    search_algorithm,
                    n_iter,
                    params
                )
                
                st.success("Fine-tuning completed!")
                display_tuning_results(tuning_results, llm_service)
                
            except Exception as e:
                st.error(f"Error in fine-tuning: {str(e)}")

def display_tuning_results(results, llm_service):
    """Display hyperparameter tuning results"""
    st.subheader("Fine-tuning Results")
    
    if 'best_params' in results:
        st.write("**Best Parameters:**")
        st.json(results['best_params'])
    
    if 'best_score' in results:
        st.metric("Best Score", f"{results['best_score']:.4f}")
    
    if 'tuning_history' in results:
        st.subheader("Tuning History")
        fig = px.line(
            results['tuning_history'],
            title="Hyperparameter Tuning Progress"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # LLM Analysis
    if st.button("🤖 Analyze Fine-tuning Results"):
        with st.spinner("Analyzing results..."):
            try:
                analysis = llm_service.analyze_tuning_results(results)
                st.subheader("AI Fine-tuning Analysis")
                st.markdown(analysis)
            except Exception as e:
                st.error(f"Error in analysis: {str(e)}")

def llm_assistant_page(llm_service):
    """LLM assistant chat interface"""
    st.header("🤖 AI Assistant")
    
    # Chat interface
    st.subheader("Ask the AI Assistant")
    
    # Display chat history
    for i, (role, message) in enumerate(st.session_state.chat_history):
        if role == "user":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**AI:** {message}")
    
    # Chat input
    user_question = st.text_input(
        "Ask about your data, models, or anything related to your ML pipeline:",
        key="chat_input"
    )
    
    if st.button("Send") and user_question:
        # Add user message to history
        st.session_state.chat_history.append(("user", user_question))
        
        with st.spinner("AI is thinking..."):
            try:
                # Get context from session state
                context = {
                    'data_shape': st.session_state.data.shape if st.session_state.data is not None else None,
                    'preprocessing_config': st.session_state.preprocessing_config,
                    'ml_config': st.session_state.ml_config,
                    'has_results': st.session_state.ml_results is not None
                }
                
                response = llm_service.chat_with_context(user_question, context)
                
                # Add AI response to history
                st.session_state.chat_history.append(("assistant", response))
                
                # Rerun to update the display
                st.rerun()
                
            except Exception as e:
                st.error(f"Error getting AI response: {str(e)}")
    
    # Clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

if __name__ == "__main__":
    main()
