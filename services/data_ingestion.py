import pandas as pd
import numpy as np
import pyodbc
import sqlalchemy
from sqlalchemy import create_engine
import os
from typing import Optional, Dict, Any
import logging
from io import StringIO, BytesIO
import requests
from sklearn.datasets import load_iris, load_wine, load_diabetes, make_regression, fetch_california_housing
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestionService:
    """Service for handling various data ingestion methods"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls']
        
    def load_file(self, uploaded_file) -> pd.DataFrame:
        """
        Load data from uploaded file (CSV or Excel)
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Try different encodings and separators
                content = uploaded_file.read()
                
                # Try UTF-8 first
                try:
                    df = pd.read_csv(StringIO(content.decode('utf-8')))
                except UnicodeDecodeError:
                    # Try latin-1 if UTF-8 fails
                    try:
                        df = pd.read_csv(StringIO(content.decode('latin-1')))
                    except UnicodeDecodeError:
                        # Try cp1252 as last resort
                        df = pd.read_csv(StringIO(content.decode('cp1252')))
                
                # Auto-detect separator if needed
                if df.shape[1] == 1 and ';' in df.columns[0]:
                    uploaded_file.seek(0)
                    content = uploaded_file.read()
                    df = pd.read_csv(StringIO(content.decode('utf-8')), sep=';')
                    
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            logger.info(f"Successfully loaded file with shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            raise e
    
    def load_from_sql(self, server: str, database: str, username: str, 
                      password: str, query: str) -> pd.DataFrame:
        """
        Load data from SQL Server
        
        Args:
            server: SQL Server hostname
            database: Database name
            username: Username
            password: Password
            query: SQL query to execute
            
        Returns:
            pd.DataFrame: Query results
        """
        try:
            # Create connection string
            connection_string = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={server};"
                f"DATABASE={database};"
                f"UID={username};"
                f"PWD={password};"
                f"Trusted_Connection=no;"
            )
            
            # Alternative using SQLAlchemy
            engine_string = (
                f"mssql+pyodbc://{username}:{password}@{server}/{database}"
                f"?driver=ODBC+Driver+17+for+SQL+Server"
            )
            
            try:
                # Try SQLAlchemy first
                engine = create_engine(engine_string)
                df = pd.read_sql(query, engine)
            except Exception:
                # Fallback to pyodbc
                conn = pyodbc.connect(connection_string)
                df = pd.read_sql(query, conn)
                conn.close()
            
            logger.info(f"Successfully loaded data from SQL Server with shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error connecting to SQL Server: {str(e)}")
            raise e
    
    def load_from_databricks(self, server_hostname: str, http_path: str, 
                           access_token: str, query: str) -> pd.DataFrame:
        """
        Load data from Databricks
        
        Args:
            server_hostname: Databricks server hostname
            http_path: HTTP path for the cluster/warehouse
            access_token: Access token for authentication
            query: SQL query to execute
            
        Returns:
            pd.DataFrame: Query results
        """
        try:
            # Install databricks-sql-connector if not available
            try:
                from databricks import sql
            except ImportError:
                raise ImportError("databricks-sql-connector not installed. Please install it with: pip install databricks-sql-connector")
            
            # Create connection
            connection = sql.connect(
                server_hostname=server_hostname,
                http_path=http_path,
                access_token=access_token
            )
            
            # Execute query
            cursor = connection.cursor()
            cursor.execute(query)
            
            # Fetch results
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=columns)
            
            # Close connections
            cursor.close()
            connection.close()
            
            logger.info(f"Successfully loaded data from Databricks with shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error connecting to Databricks: {str(e)}")
            raise e
    
    def load_sample_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Load sample datasets for testing
        
        Args:
            dataset_name: Name of the sample dataset
            
        Returns:
            pd.DataFrame: Sample dataset
        """
        try:
            if dataset_name == 'iris':
                data = load_iris()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                df['target_names'] = [data.target_names[i] for i in data.target]
                
            elif dataset_name == 'california_housing':
                data = fetch_california_housing()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                
            elif dataset_name == 'boston':
                # Boston dataset removed due to ethical concerns - using synthetic regression data
                X, y = make_regression(
                    n_samples=506, 
                    n_features=13, 
                    n_informative=10,
                    noise=0.1, 
                    random_state=42
                )
                feature_names = [
                    'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 
                    'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat'
                ]
                df = pd.DataFrame(X, columns=feature_names)
                df['target'] = y
                
            elif dataset_name == 'wine':
                data = load_wine()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                df['target_names'] = [data.target_names[i] for i in data.target]
                
            elif dataset_name == 'diabetes':
                data = load_diabetes()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                
            else:
                raise ValueError(f"Unknown sample dataset: {dataset_name}")
            
            logger.info(f"Successfully loaded sample dataset '{dataset_name}' with shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading sample dataset: {str(e)}")
            raise e
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate loaded data and return quality metrics
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dict: Validation results
        """
        try:
            validation_results = {
                'shape': df.shape,
                'total_cells': df.shape[0] * df.shape[1],
                'missing_values': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
                'duplicate_rows': df.duplicated().sum(),
                'data_types': df.dtypes.value_counts().to_dict(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
                'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
            }
            
            # Check for high cardinality categorical columns
            high_cardinality_cols = []
            for col in validation_results['categorical_columns']:
                if df[col].nunique() > 50:
                    high_cardinality_cols.append({
                        'column': col,
                        'unique_values': df[col].nunique()
                    })
            validation_results['high_cardinality_columns'] = high_cardinality_cols
            
            # Check for potential issues
            issues = []
            if validation_results['missing_percentage'] > 50:
                issues.append("High percentage of missing values (>50%)")
            if validation_results['duplicate_rows'] > df.shape[0] * 0.1:
                issues.append("High number of duplicate rows (>10%)")
            if len(high_cardinality_cols) > 0:
                issues.append("High cardinality categorical columns detected")
            
            validation_results['potential_issues'] = issues
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            raise e
    
    def get_data_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data profile
        
        Args:
            df: DataFrame to profile
            
        Returns:
            Dict: Data profile information
        """
        try:
            profile = {
                'basic_info': {
                    'rows': df.shape[0],
                    'columns': df.shape[1],
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
                },
                'column_info': {},
                'data_quality': self.validate_data(df)
            }
            
            # Detailed column analysis
            for col in df.columns:
                col_info = {
                    'dtype': str(df[col].dtype),
                    'non_null_count': df[col].count(),
                    'null_count': df[col].isnull().sum(),
                    'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                    'unique_count': df[col].nunique(),
                    'unique_percentage': (df[col].nunique() / len(df)) * 100
                }
                
                if df[col].dtype in ['int64', 'float64']:
                    col_info.update({
                        'mean': df[col].mean(),
                        'median': df[col].median(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'q25': df[col].quantile(0.25),
                        'q75': df[col].quantile(0.75),
                        'skewness': df[col].skew(),
                        'kurtosis': df[col].kurtosis()
                    })
                
                elif df[col].dtype == 'object':
                    try:
                        col_info.update({
                            'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                            'most_frequent_count': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0,
                            'average_length': df[col].astype(str).str.len().mean()
                        })
                    except:
                        pass
                
                profile['column_info'][col] = col_info
            
            return profile
            
        except Exception as e:
            logger.error(f"Error generating data profile: {str(e)}")
            raise e
    
    def suggest_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Suggest optimal data types for columns
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict: Suggested data types for each column
        """
        try:
            suggestions = {}
            
            for col in df.columns:
                current_dtype = df[col].dtype
                
                if current_dtype == 'object':
                    # Try to convert to numeric
                    try:
                        pd.to_numeric(df[col], errors='raise')
                        # Check if it can be integer
                        if df[col].dropna().astype(float).apply(float.is_integer).all():
                            suggestions[col] = 'int64'
                        else:
                            suggestions[col] = 'float64'
                    except:
                        # Try to convert to datetime
                        try:
                            pd.to_datetime(df[col], errors='raise')
                            suggestions[col] = 'datetime64[ns]'
                        except:
                            # Check if it's categorical
                            if df[col].nunique() / len(df) < 0.1:  # Less than 10% unique values
                                suggestions[col] = 'category'
                            else:
                                suggestions[col] = 'object'
                
                elif current_dtype in ['int64', 'float64']:
                    # Check if we can downcast
                    if current_dtype == 'int64':
                        min_val, max_val = df[col].min(), df[col].max()
                        if min_val >= 0:
                            if max_val <= 255:
                                suggestions[col] = 'uint8'
                            elif max_val <= 65535:
                                suggestions[col] = 'uint16'
                            elif max_val <= 4294967295:
                                suggestions[col] = 'uint32'
                            else:
                                suggestions[col] = 'uint64'
                        else:
                            if min_val >= -128 and max_val <= 127:
                                suggestions[col] = 'int8'
                            elif min_val >= -32768 and max_val <= 32767:
                                suggestions[col] = 'int16'
                            elif min_val >= -2147483648 and max_val <= 2147483647:
                                suggestions[col] = 'int32'
                            else:
                                suggestions[col] = 'int64'
                    else:
                        # Float optimization
                        if df[col].min() >= np.finfo(np.float32).min and df[col].max() <= np.finfo(np.float32).max:
                            suggestions[col] = 'float32'
                        else:
                            suggestions[col] = 'float64'
                else:
                    suggestions[col] = str(current_dtype)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error suggesting data types: {str(e)}")
            raise e
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean column names to be ML-friendly
        
        Args:
            df: DataFrame with potentially problematic column names
            
        Returns:
            pd.DataFrame: DataFrame with cleaned column names
        """
        try:
            df_clean = df.copy()
            
            # Clean column names
            df_clean.columns = (df_clean.columns
                               .str.strip()  # Remove leading/trailing spaces
                               .str.lower()  # Convert to lowercase
                               .str.replace(' ', '_')  # Replace spaces with underscores
                               .str.replace('[^a-zA-Z0-9_]', '', regex=True)  # Remove special characters
                               .str.replace(r'_+', '_', regex=True)  # Replace multiple underscores with single
                               .str.strip('_'))  # Remove leading/trailing underscores
            
            # Ensure column names don't start with numbers
            df_clean.columns = ['col_' + col if col[0].isdigit() else col for col in df_clean.columns]
            
            # Handle duplicate column names
            cols = pd.Series(df_clean.columns)
            for dup in cols[cols.duplicated()].unique():
                cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup 
                                                                 for i in range(sum(cols == dup))]
            df_clean.columns = cols
            
            logger.info("Column names cleaned successfully")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cleaning column names: {str(e)}")
            raise e
