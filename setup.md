# AutoML Pipeline Assistant - Complete Setup Guide

This guide will walk you through the complete setup process for the AutoML Pipeline Assistant application.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Azure OpenAI Setup](#azure-openai-setup)
5. [Database Setup](#database-setup)
6. [Running the Application](#running-the-application)
7. [Usage Guide](#usage-guide)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Configuration](#advanced-configuration)

## Prerequisites

### System Requirements
- Python 3.8 or higher
- At least 8GB RAM (16GB recommended)
- 5GB free disk space
- Internet connection for package installation and Azure OpenAI

### Software Dependencies
- Git (for cloning repositories)
- Azure CLI (optional, for Azure services)
- SQL Server ODBC Driver (for SQL Server connectivity)
- Databricks CLI (optional, for Databricks connectivity)

## Installation

### Step 1: Create Project Directory
```bash
mkdir automl-pipeline-assistant
cd automl-pipeline-assistant
```

### Step 2: Set Up Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Create Project Structure
```bash
mkdir -p services utils models logs
```

### Step 4: Create Python Files
Create the following files in your project directory with the provided code:

1. **Main Application File**: `app.py` (from the first artifact)
2. **Services Directory**:
   - `services/__init__.py` (empty file)
   - `services/data_ingestion.py`
   - `services/automl_service.py`
   - `services/llm_service.py`
   - `services/model_service.py`
3. **Utils Directory**:
   - `utils/__init__.py` (empty file)
   - `utils/config.py`
4. **Requirements File**: `requirements.txt`

### Step 5: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# If you encounter issues, install core packages first:
pip install streamlit pandas numpy scikit-learn
pip install pycaret
pip install openai
pip install plotly
```

### Step 6: Verify Installation
```bash
python -c "from utils.config import print_dependency_status; print_dependency_status()"
```

## Configuration

### Step 1: Environment Variables
Create a `.env` file in your project root:

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

# Application Configuration
MODEL_STORAGE_PATH=models
EXPERIMENT_LOGS_PATH=logs
MAX_UPLOAD_SIZE_MB=500

# ML Configuration
DEFAULT_CV_FOLDS=5
DEFAULT_TRAIN_SIZE=0.8
DEFAULT_RANDOM_SEED=123
DEFAULT_TUNING_ITERATIONS=20
MAX_MODELS_TO_COMPARE=15
```

### Step 2: Load Environment Variables
Add this to your Python code or create a `load_env.py` file:

```python
from dotenv import load_dotenv
load_dotenv()
```

## Azure OpenAI Setup

### Step 1: Create Azure OpenAI Resource
1. Go to [Azure Portal](https://portal.azure.com)
2. Create a new Azure OpenAI resource
3. Wait for deployment to complete
4. Note down the endpoint URL and API key

### Step 2: Deploy GPT-4o Model
1. Go to Azure OpenAI Studio
2. Navigate to "Deployments"
3. Create a new deployment with:
   - Model: `gpt-4o`
   - Deployment name: `gpt-4o` (or your preferred name)
   - Version: Latest available

### Step 3: Configure API Access
1. Copy the API key from the Azure OpenAI resource
2. Copy the endpoint URL
3. Update your `.env` file with these values

### Step 4: Test Connection
```python
from services.llm_service import LLMService
llm = LLMService()
response = llm.chat_with_context("Hello, are you working?", {})
print(response)
```

## Database Setup

### SQL Server Setup
1. **Install SQL Server ODBC Driver**:
   - Windows: Download from Microsoft
   - macOS: `brew install msodbcsql17`
   - Linux: Follow Microsoft's installation guide

2. **Test Connection**:
   ```python
   from services.data_ingestion import DataIngestionService
   data_service = DataIngestionService()
   # Test with your credentials
   ```

### Databricks Setup
1. **Install Databricks CLI**:
   ```bash
   pip install databricks-cli
   ```

2. **Configure Access Token**:
   - Go to Databricks workspace
   - Generate a personal access token
   - Configure in your application

## Running the Application

### Step 1: Start the Application
```bash
streamlit run app.py
```

### Step 2: Access the Application
- Open your browser to `http://localhost:8501`
- The application should load with the navigation sidebar

### Step 3: Test Basic Functionality
1. **Data Ingestion**: Try uploading a CSV file or using sample data
2. **Preprocessing**: Configure preprocessing options
3. **Model Training**: Run a simple classification or regression task
4. **LLM Integration**: Test the AI assistant

## Usage Guide

### 1. Data Ingestion
- **Upload Files**: Support for CSV, Excel files
- **SQL Server**: Connect using connection string
- **Databricks**: Connect using access token and endpoint
- **Sample Data**: Use built-in datasets for testing

### 2. Data Preprocessing
- **Missing Values**: Choose imputation methods
- **Normalization**: Select scaling methods
- **Feature Selection**: Choose selection algorithms
- **Imbalanced Data**: Handle class imbalance
- **Outlier Removal**: Configure outlier detection

### 3. AutoML Configuration
- **Problem Type**: Classification, regression, clustering, anomaly detection
- **Model Selection**: Choose specific models or use all
- **Evaluation Metrics**: Select appropriate metrics
- **Cross-Validation**: Configure CV strategy

### 4. Model Training
- **Start Training**: Begin the AutoML process
- **Monitor Progress**: View real-time updates
- **Compare Models**: Analyze model performance
- **Select Best Model**: Choose optimal model

### 5. Results Analysis
- **Performance Metrics**: Detailed model evaluation
- **Feature Importance**: Understand key features
- **Visualizations**: Charts and plots
- **AI Analysis**: Get LLM insights

### 6. Model Fine-tuning
- **Hyperparameter Optimization**: Multiple search strategies
- **Parameter Grids**: Custom or default parameters
- **Optimization Libraries**: sklearn, optuna, scikit-optimize
- **Performance Tracking**: Monitor improvement

### 7. LLM Assistant
- **Ask Questions**: Get AI-powered insights
- **Context Awareness**: AI understands your project
- **Recommendations**: Get expert advice
- **Chat History**: Maintain conversation context

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Issue: ModuleNotFoundError
# Solution: Install missing packages
pip install package_name

# Or reinstall all requirements
pip install -r requirements.txt --force-reinstall
```

#### 2. Azure OpenAI Connection Errors
```bash
# Check environment variables
python -c "import os; print(os.getenv('AZURE_OPENAI_API_KEY'))"

# Verify endpoint format
# Should be: https://your-resource-name.openai.azure.com/
```

#### 3. Database Connection Issues
```bash
# SQL Server: Check ODBC driver installation
odbcinst -j

# Test connection parameters
```

#### 4. Memory Issues
```bash
# Reduce model comparison count
export MAX_MODELS_TO_COMPARE=5

# Use smaller datasets for testing
```

#### 5. Streamlit Issues
```bash
# Clear Streamlit cache
streamlit cache clear

# Run with specific port
streamlit run app.py --server.port 8502
```

### Performance Optimization

#### 1. Hardware Requirements
- **Minimum**: 8GB RAM, 4 CPU cores
- **Recommended**: 16GB RAM, 8 CPU cores
- **Optimal**: 32GB RAM, 16 CPU cores

#### 2. Configuration Tuning
```python
# Reduce model training time
DEFAULT_CV_FOLDS = 3
MAX_MODELS_TO_COMPARE = 10
DEFAULT_TUNING_ITERATIONS = 10
```

#### 3. Dataset Size Limits
- **Small**: < 10,000 rows (fast processing)
- **Medium**: 10,000 - 100,000 rows (moderate processing)
- **Large**: > 100,000 rows (longer processing time)

## Advanced Configuration

### 1. Custom Model Integration
```python
# Add custom models to automl_service.py
def create_custom_model(self, model_name, **params):
    # Your custom model implementation
    pass
```

### 2. Additional Data Sources
```python
# Extend data_ingestion.py
def load_from_api(self, api_endpoint, params):
    # API data loading implementation
    pass
```

### 3. Custom Preprocessing
```python
# Add custom preprocessing steps
def custom_preprocessing(self, data, config):
    # Your custom preprocessing logic
    pass
```

### 4. Deployment Configuration
```bash
# For production deployment
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
```

### 5. Monitoring and Logging
```python
# Enhanced logging configuration
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
```

## Security Considerations

### 1. API Key Management
- Never commit API keys to version control
- Use environment variables or Azure Key Vault
- Rotate keys regularly

### 2. Database Security
- Use connection strings with minimal permissions
- Enable encryption in transit
- Implement connection pooling

### 3. File Upload Security
- Validate file types and sizes
- Scan for malicious content
- Implement upload limits

### 4. Access Control
- Implement authentication if needed
- Use HTTPS in production
- Monitor access logs

## Testing

### 1. Unit Tests
```bash
# Run tests
python -m pytest tests/

# With coverage
python -m pytest --cov=services tests/
```

### 2. Integration Tests
```bash
# Test data ingestion
python -c "from services.data_ingestion import DataIngestionService; ds = DataIngestionService(); print('Data service OK')"

# Test AutoML
python -c "from services.automl_service import AutoMLService; ams = AutoMLService(); print('AutoML service OK')"

# Test LLM service
python -c "from services.llm_service import LLMService; llm = LLMService(); print('LLM service OK')"
```

### 3. End-to-End Testing
1. Start the application
2. Upload test data
3. Configure preprocessing
4. Train a simple model
5. Analyze results
6. Test LLM integration

## Support and Maintenance

### 1. Regular Updates
```bash
# Update packages
pip install --upgrade -r requirements.txt

# Check for security vulnerabilities
pip audit
```

### 2. Backup Strategy
- Regular backup of models directory
- Export important configurations
- Version control your customizations

### 3. Monitoring
- Monitor application performance
- Check error logs regularly
- Monitor Azure OpenAI usage and costs

## Getting Help

### 1. Documentation
- PyCaret: [https://pycaret.org/](https://pycaret.org/)
- Streamlit: [https://docs.streamlit.io/](https://docs.streamlit.io/)
- Azure OpenAI: [https://docs.microsoft.com/azure/cognitive-services/openai/](https://docs.microsoft.com/azure/cognitive-services/openai/)

### 2. Community Support
- PyCaret Slack community
- Streamlit community forum
- Azure OpenAI documentation and support

### 3. Troubleshooting Steps
1. Check logs for error messages
2. Verify environment variables
3. Test individual components
4. Check dependency versions
5. Review configuration settings

---

**Congratulations!** You now have a fully functional AutoML Pipeline Assistant with AI-powered insights and recommendations. The application provides a complete end-to-end machine learning workflow with an intuitive interface and powerful automation capabilities.
