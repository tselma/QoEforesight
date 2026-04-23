"""
QoE-Foresight: Google Colab Data Loader and Preprocessing for Public Datasets
==============================================================================

This module provides comprehensive data loading and preprocessing capabilities
for public QoE datasets optimized for Google Colab environment and top Q1 
journal publication quality research.

Supported Datasets:
- ITU Extracted Features
- Waterloo University Streaming Datasets
- MAWI QoS Features
- LIVE Netflix II Dataset
- Combined Features with MOS scores
- Drift Detection Ground Truth

Author: Manus AI
Date: June 11, 2025
Version: 2.0 (Public Dataset Integration)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
import logging
import ast
import pickle
import os
from typing import Dict, List, Tuple, Optional, Any
from google.colab import drive
import zipfile
import requests
from io import StringIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class PublicDatasetConfig:
    """Configuration for public dataset integration."""
    
    def __init__(self):
        # Google Drive paths for public datasets
        self.drive_path = "/content/drive/MyDrive/Colab Notebooks/PublicData/1ext"
        
        # Dataset file mappings
        self.dataset_files = {
            'itu_features': 'ITU_extracted_features.csv',
            'combined_features': 'combined_features.csv',
            'cleaned_features': 'cleaned_features_ready_for_training.csv',
            'waterloo_streaming': 'waterloo_streaminglog.csv',
            'waterloo_videos': 'waterloo_servervideos_fromjson.csv',
            'waterloo_data': 'waterloo_data.csv',
            'waterloo_video': 'waterloo_video.csv',
            'mawi_qos': 'mawi_qos_features.csv',
            'live_netflix': 'LIVE_NFLX_II_pkl_mat.csv',
            'qos_metrics': 'qos_metrics.xlsx',
            'qoe_metrics': 'qoe_metrics.csv',
            'final_merged': 'final_merged_dataset.csv',
            'lstm_drift': 'lstm_drift_results.csv',
            'drift_detection': 'drift_detection_results.csv',
            'rl_results': 'all_drifts_rl_results.csv',
            'predictions': 'predictions.csv'
        }
        
        # Feature mappings for different datasets
        self.feature_mappings = {
            'qoe_features': [
                'playout_bitrate', 'frame_rate', 'rebuffer_duration', 
                'throughput_trace', 'buffer_occupancy', 'stall_events',
                'playback_resolution'
            ],
            'device_features': [
                'battery', 'cpu', 'gpu', 'temperature'
            ],
            'network_features': [
                'throughput_trace', 'inter_arrival_delay', 'jitter',
                'packet_loss', 'latency'
            ],
            'content_features': [
                'content_type', 'region_id', 'adaptive_resolution_flag',
                'encoding_profile', 'bitrate_kbps', 'framerate_encoding'
            ]
        }
        
        # Target variables
        self.target_variables = ['mos', 'qoe_score', 'quality_rating']
        
        # Preprocessing parameters
        self.preprocessing_config = {
            'missing_value_threshold': 0.3,
            'outlier_threshold': 3.0,
            'correlation_threshold': 0.95,
            'variance_threshold': 0.01
        }

class PublicDatasetLoader:
    """Comprehensive data loader for public QoE datasets."""
    
    def __init__(self, config: PublicDatasetConfig):
        self.config = config
        self.datasets = {}
        self.processed_datasets = {}
        self.feature_importance = {}
        self.data_statistics = {}
        
        # Mount Google Drive
        self._mount_drive()
        
        logger.info("Public dataset loader initialized")
    
    def _mount_drive(self):
        """Mount Google Drive for dataset access."""
        try:
            drive.mount('/content/drive')
            logger.info("Google Drive mounted successfully")
        except Exception as e:
            logger.warning(f"Drive mounting failed: {e}")
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all available public datasets."""
        logger.info("Loading all public datasets...")
        
        for dataset_name, filename in self.config.dataset_files.items():
            try:
                self.datasets[dataset_name] = self._load_single_dataset(filename)
                logger.info(f"Loaded {dataset_name}: {self.datasets[dataset_name].shape}")
            except Exception as e:
                logger.warning(f"Failed to load {dataset_name}: {e}")
        
        return self.datasets
    
    def _load_single_dataset(self, filename: str) -> pd.DataFrame:
        """Load a single dataset file."""
        filepath = os.path.join(self.config.drive_path, filename)
        
        if filename.endswith('.csv'):
            # Handle special cases for complex CSV files
            if 'ITU_extracted' in filename or 'combined_features' in filename:
                return self._load_complex_csv(filepath)
            else:
                return pd.read_csv(filepath)
        elif filename.endswith('.xlsx'):
            return pd.read_excel(filepath)
        elif filename.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    
    def _load_complex_csv(self, filepath: str) -> pd.DataFrame:
        """Load complex CSV files with array-like string data."""
        df = pd.read_csv(filepath)
        
        # Process array-like string columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Try to parse array-like strings
                    sample_val = str(df[col].iloc[0])
                    if sample_val.startswith('[') and 'np.float64' in sample_val:
                        df[col] = df[col].apply(self._parse_numpy_array)
                except:
                    continue
        
        return df
    
    def _parse_numpy_array(self, array_str: str) -> List[float]:
        """Parse numpy array string to list of floats."""
        try:
            # Extract numeric values from numpy array string
            import re
            numbers = re.findall(r'np\.float64\(([\d.]+)\)', str(array_str))
            return [float(x) for x in numbers]
        except:
            return []
    
    def get_itu_dataset(self) -> pd.DataFrame:
        """Get processed ITU dataset for QoE prediction."""
        if 'itu_features' not in self.datasets:
            self.datasets['itu_features'] = self._load_single_dataset('ITU_extracted_features.csv')
        
        return self._process_itu_dataset(self.datasets['itu_features'])
    
    def _process_itu_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process ITU dataset for QoE prediction."""
        processed_df = df.copy()
        
        # Expand array columns to individual features
        array_columns = ['playout_bitrate', 'frame_rate', 'rebuffer_duration', 
                        'throughput_trace', 'buffer_occupancy']
        
        for col in array_columns:
            if col in processed_df.columns:
                # Extract statistical features from arrays
                processed_df[f'{col}_mean'] = processed_df[col].apply(
                    lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else 0
                )
                processed_df[f'{col}_std'] = processed_df[col].apply(
                    lambda x: np.std(x) if isinstance(x, list) and len(x) > 0 else 0
                )
                processed_df[f'{col}_max'] = processed_df[col].apply(
                    lambda x: np.max(x) if isinstance(x, list) and len(x) > 0 else 0
                )
                processed_df[f'{col}_min'] = processed_df[col].apply(
                    lambda x: np.min(x) if isinstance(x, list) and len(x) > 0 else 0
                )
        
        # Remove original array columns
        processed_df = processed_df.drop(columns=array_columns, errors='ignore')
        
        return processed_df
    
    def get_combined_dataset(self) -> pd.DataFrame:
        """Get combined dataset with MOS scores."""
        if 'combined_features' not in self.datasets:
            self.datasets['combined_features'] = self._load_single_dataset('combined_features.csv')
        
        return self._process_combined_dataset(self.datasets['combined_features'])
    
    def _process_combined_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process combined dataset with MOS scores."""
        processed_df = df.copy()
        
        # Extract MOS scores as target variable
        if 'mos' in processed_df.columns:
            processed_df['qoe_target'] = processed_df['mos']
        
        # Process content and device features
        categorical_features = ['content', 'encoding_profile', 'device', 'content_type']
        for feature in categorical_features:
            if feature in processed_df.columns:
                le = LabelEncoder()
                processed_df[f'{feature}_encoded'] = le.fit_transform(
                    processed_df[feature].astype(str)
                )
        
        return processed_df
    
    def get_waterloo_dataset(self) -> pd.DataFrame:
        """Get combined Waterloo datasets."""
        waterloo_datasets = []
        
        for dataset_name in ['waterloo_streaming', 'waterloo_data', 'waterloo_video']:
            if dataset_name not in self.datasets:
                try:
                    filename = self.config.dataset_files[dataset_name]
                    self.datasets[dataset_name] = self._load_single_dataset(filename)
                    waterloo_datasets.append(self.datasets[dataset_name])
                except Exception as e:
                    logger.warning(f"Failed to load {dataset_name}: {e}")
        
        if waterloo_datasets:
            return self._merge_waterloo_datasets(waterloo_datasets)
        else:
            return pd.DataFrame()
    
    def _merge_waterloo_datasets(self, datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple Waterloo datasets."""
        if not datasets:
            return pd.DataFrame()
        
        merged_df = datasets[0]
        for df in datasets[1:]:
            # Find common columns for merging
            common_cols = set(merged_df.columns) & set(df.columns)
            if common_cols:
                merge_col = list(common_cols)[0]
                merged_df = pd.merge(merged_df, df, on=merge_col, how='outer')
            else:
                # Concatenate if no common columns
                merged_df = pd.concat([merged_df, df], axis=1)
        
        return merged_df
    
    def get_mawi_qos_dataset(self) -> pd.DataFrame:
        """Get MAWI QoS dataset for network analysis."""
        if 'mawi_qos' not in self.datasets:
            self.datasets['mawi_qos'] = self._load_single_dataset('mawi_qos_features.csv')
        
        return self.datasets['mawi_qos']
    
    def get_live_netflix_dataset(self) -> pd.DataFrame:
        """Get LIVE Netflix II dataset."""
        if 'live_netflix' not in self.datasets:
            self.datasets['live_netflix'] = self._load_single_dataset('LIVE_NFLX_II_pkl_mat.csv')
        
        return self.datasets['live_netflix']
    
    def get_drift_detection_datasets(self) -> Dict[str, pd.DataFrame]:
        """Get drift detection datasets for validation."""
        drift_datasets = {}
        
        drift_files = ['lstm_drift', 'drift_detection', 'rl_results']
        for dataset_name in drift_files:
            if dataset_name not in self.datasets:
                try:
                    filename = self.config.dataset_files[dataset_name]
                    self.datasets[dataset_name] = self._load_single_dataset(filename)
                    drift_datasets[dataset_name] = self.datasets[dataset_name]
                except Exception as e:
                    logger.warning(f"Failed to load {dataset_name}: {e}")
        
        return drift_datasets
    
    def get_final_merged_dataset(self) -> pd.DataFrame:
        """Get the final merged dataset for comprehensive evaluation."""
        if 'final_merged' not in self.datasets:
            self.datasets['final_merged'] = self._load_single_dataset('final_merged_dataset.csv')
        
        return self.datasets['final_merged']

class PublicDatasetPreprocessor:
    """Advanced preprocessing for public QoE datasets."""
    
    def __init__(self, config: PublicDatasetConfig):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
        self.imputers = {}
        
        logger.info("Public dataset preprocessor initialized")
    
    def preprocess_for_qoe_prediction(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Preprocess dataset for QoE prediction task."""
        logger.info("Preprocessing dataset for QoE prediction...")
        
        # Create a copy for processing
        processed_df = df.copy()
        
        # Identify target variable
        target_col = None
        for col in self.config.target_variables:
            if col in processed_df.columns:
                target_col = col
                break
        
        if target_col is None:
            # Create synthetic QoE target if not available
            logger.warning("No target variable found, creating synthetic QoE scores")
            processed_df['qoe_target'] = self._create_synthetic_qoe(processed_df)
            target_col = 'qoe_target'
        
        # Separate features and target
        y = processed_df[target_col].values
        X_df = processed_df.drop(columns=[target_col])
        
        # Handle missing values
        X_df = self._handle_missing_values(X_df)
        
        # Encode categorical variables
        X_df = self._encode_categorical_features(X_df)
        
        # Remove highly correlated features
        X_df = self._remove_correlated_features(X_df)
        
        # Scale numerical features
        X_df = self._scale_features(X_df)
        
        # Feature selection
        X_df = self._select_features(X_df, y)
        
        feature_names = list(X_df.columns)
        X = X_df.values
        
        logger.info(f"Preprocessing complete: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, feature_names
    
    def _create_synthetic_qoe(self, df: pd.DataFrame) -> np.ndarray:
        """Create synthetic QoE scores based on available features."""
        # Use heuristic based on common QoE factors
        qoe_score = np.random.normal(3.5, 1.0, len(df))  # Base score
        
        # Adjust based on available features
        if 'stall_events' in df.columns:
            qoe_score -= df['stall_events'] * 0.5
        
        if 'rebuffer_duration_mean' in df.columns:
            qoe_score -= df['rebuffer_duration_mean'] * 0.1
        
        if 'throughput_trace_mean' in df.columns:
            qoe_score += np.log1p(df['throughput_trace_mean']) * 0.2
        
        # Clip to valid range
        qoe_score = np.clip(qoe_score, 1.0, 5.0)
        
        return qoe_score
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Remove columns with too many missing values
        missing_ratio = df.isnull().sum() / len(df)
        cols_to_drop = missing_ratio[missing_ratio > self.config.preprocessing_config['missing_value_threshold']].index
        df = df.drop(columns=cols_to_drop)
        
        # Impute remaining missing values
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(numerical_cols) > 0:
            num_imputer = SimpleImputer(strategy='median')
            df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
            self.imputers['numerical'] = num_imputer
        
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
            self.imputers['categorical'] = cat_imputer
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if df[col].nunique() < 50:  # Only encode if not too many categories
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
                df = df.drop(columns=[col])
        
        return df
    
    def _remove_correlated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = [column for column in upper_triangle.columns 
                      if any(upper_triangle[column] > self.config.preprocessing_config['correlation_threshold'])]
            
            df = df.drop(columns=to_drop)
            logger.info(f"Removed {len(to_drop)} highly correlated features")
        
        return df
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            self.scalers['standard'] = scaler
        
        return df
    
    def _select_features(self, df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Select most important features."""
        from sklearn.feature_selection import SelectKBest, f_regression
        
        if df.shape[1] > 50:  # Only apply if too many features
            k = min(50, df.shape[1])  # Select top 50 features
            selector = SelectKBest(score_func=f_regression, k=k)
            
            X_selected = selector.fit_transform(df.values, y)
            selected_features = df.columns[selector.get_support()]
            
            df = pd.DataFrame(X_selected, columns=selected_features, index=df.index)
            self.feature_selectors['kbest'] = selector
            
            logger.info(f"Selected {k} most important features")
        
        return df
    
    def preprocess_for_drift_detection(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[int]]:
        """Preprocess dataset for drift detection."""
        logger.info("Preprocessing dataset for drift detection...")
        
        # Extract features
        X, _, feature_names = self.preprocess_for_qoe_prediction(df)
        
        # Create drift points (for simulation)
        drift_points = self._identify_drift_points(X)
        
        return X, drift_points
    
    def _identify_drift_points(self, X: np.ndarray) -> List[int]:
        """Identify potential drift points in the data."""
        # Simple drift detection based on statistical changes
        drift_points = []
        window_size = 100
        
        if len(X) > window_size * 2:
            for i in range(window_size, len(X) - window_size, window_size):
                # Compare statistical properties of consecutive windows
                window1 = X[i-window_size:i]
                window2 = X[i:i+window_size]
                
                # Use KS test for drift detection
                from scipy.stats import ks_2samp
                
                for feature_idx in range(X.shape[1]):
                    statistic, p_value = ks_2samp(window1[:, feature_idx], window2[:, feature_idx])
                    if p_value < 0.01:  # Significant difference
                        drift_points.append(i)
                        break
        
        return drift_points
    
    def create_train_test_splits(self, X: np.ndarray, y: np.ndarray, 
                                test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create train/test splits for evaluation."""
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=None)

class PublicDatasetVisualizer:
    """Visualization utilities for public datasets."""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    def plot_dataset_overview(self, datasets: Dict[str, pd.DataFrame]) -> None:
        """Plot overview of all loaded datasets."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Public Datasets Overview', fontsize=16, fontweight='bold')
        
        # Dataset sizes
        dataset_names = list(datasets.keys())
        dataset_sizes = [len(df) for df in datasets.values()]
        
        axes[0, 0].bar(range(len(dataset_names)), dataset_sizes, color=self.colors[:len(dataset_names)])
        axes[0, 0].set_title('Dataset Sizes')
        axes[0, 0].set_xlabel('Datasets')
        axes[0, 0].set_ylabel('Number of Samples')
        axes[0, 0].set_xticks(range(len(dataset_names)))
        axes[0, 0].set_xticklabels(dataset_names, rotation=45, ha='right')
        
        # Feature counts
        feature_counts = [len(df.columns) for df in datasets.values()]
        axes[0, 1].bar(range(len(dataset_names)), feature_counts, color=self.colors[:len(dataset_names)])
        axes[0, 1].set_title('Feature Counts')
        axes[0, 1].set_xlabel('Datasets')
        axes[0, 1].set_ylabel('Number of Features')
        axes[0, 1].set_xticks(range(len(dataset_names)))
        axes[0, 1].set_xticklabels(dataset_names, rotation=45, ha='right')
        
        # Missing value analysis
        missing_percentages = []
        for df in datasets.values():
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            missing_percentages.append(missing_pct)
        
        axes[1, 0].bar(range(len(dataset_names)), missing_percentages, color=self.colors[:len(dataset_names)])
        axes[1, 0].set_title('Missing Values (%)')
        axes[1, 0].set_xlabel('Datasets')
        axes[1, 0].set_ylabel('Missing Values (%)')
        axes[1, 0].set_xticks(range(len(dataset_names)))
        axes[1, 0].set_xticklabels(dataset_names, rotation=45, ha='right')
        
        # Data types distribution
        if datasets:
            sample_df = list(datasets.values())[0]
            dtype_counts = sample_df.dtypes.value_counts()
            axes[1, 1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
            axes[1, 1].set_title('Data Types Distribution (Sample)')
        
        plt.tight_layout()
        plt.show()
    
    def plot_qoe_distribution(self, df: pd.DataFrame, target_col: str = 'mos') -> None:
        """Plot QoE score distribution."""
        if target_col not in df.columns:
            logger.warning(f"Target column {target_col} not found")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('QoE Score Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Histogram
        axes[0].hist(df[target_col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title('QoE Score Distribution')
        axes[0].set_xlabel('QoE Score')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(df[target_col].dropna())
        axes[1].set_title('QoE Score Box Plot')
        axes[1].set_ylabel('QoE Score')
        axes[1].grid(True, alpha=0.3)
        
        # QQ plot
        from scipy import stats
        stats.probplot(df[target_col].dropna(), dist="norm", plot=axes[2])
        axes[2].set_title('QoE Score Q-Q Plot')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_correlations(self, df: pd.DataFrame, target_col: str = 'mos') -> None:
        """Plot feature correlations with target variable."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if target_col in numerical_cols and len(numerical_cols) > 1:
            correlations = df[numerical_cols].corr()[target_col].drop(target_col).sort_values(key=abs, ascending=False)
            
            plt.figure(figsize=(12, 8))
            colors = ['red' if x < 0 else 'blue' for x in correlations.values]
            bars = plt.barh(range(len(correlations)), correlations.values, color=colors, alpha=0.7)
            
            plt.yticks(range(len(correlations)), correlations.index)
            plt.xlabel('Correlation with QoE Score')
            plt.title('Feature Correlations with QoE Score', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, correlations.values)):
                plt.text(value + 0.01 if value >= 0 else value - 0.01, i, 
                        f'{value:.3f}', va='center', ha='left' if value >= 0 else 'right')
            
            plt.tight_layout()
            plt.show()

# Example usage and testing
if __name__ == "__main__":
    # Initialize configuration and loader
    config = PublicDatasetConfig()
    loader = PublicDatasetLoader(config)
    preprocessor = PublicDatasetPreprocessor(config)
    visualizer = PublicDatasetVisualizer()
    
    print("🚀 QoE-Foresight Public Dataset Integration")
    print("=" * 50)
    
    # Load datasets
    print("📊 Loading public datasets...")
    datasets = loader.load_all_datasets()
    
    if datasets:
        print(f"✅ Successfully loaded {len(datasets)} datasets")
        
        # Show dataset overview
        visualizer.plot_dataset_overview(datasets)
        
        # Process ITU dataset for QoE prediction
        print("\n🔧 Processing ITU dataset for QoE prediction...")
        itu_df = loader.get_itu_dataset()
        if not itu_df.empty:
            X, y, feature_names = preprocessor.preprocess_for_qoe_prediction(itu_df)
            print(f"✅ Processed dataset: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Create train/test splits
            X_train, X_test, y_train, y_test = preprocessor.create_train_test_splits(X, y)
            print(f"📈 Train set: {X_train.shape[0]} samples")
            print(f"📊 Test set: {X_test.shape[0]} samples")
        
        # Process combined dataset
        print("\n🔧 Processing combined dataset with MOS scores...")
        combined_df = loader.get_combined_dataset()
        if not combined_df.empty:
            visualizer.plot_qoe_distribution(combined_df, 'mos')
            visualizer.plot_feature_correlations(combined_df, 'mos')
        
        print("\n🎯 Public dataset integration complete!")
        print("Ready for QoE-Foresight framework integration")
    
    else:
        print("❌ No datasets loaded. Please check Google Drive paths.")

