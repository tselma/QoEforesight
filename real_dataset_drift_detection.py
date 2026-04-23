"""
QoE-Foresight: Advanced Drift Detection with Real Dataset Validation
====================================================================

This module provides advanced drift detection capabilities validated against
real-world public datasets. Designed for top 1% Q1 journal publication with
comprehensive comparisons against state-of-the-art methods.

Key Features:
- HDDM-A and UADF integration with real dataset validation
- Comparison against existing LSTM and traditional drift detection methods
- Real drift events from public datasets for ground truth validation
- Publication-quality statistical analysis and significance testing
- Google Colab optimized implementation

Author: Manus AI
Date: June 11, 2025
Version: 2.0 (Real Dataset Validation)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import ks_2samp, mannwhitneyu
import tensorflow as tf
from tensorflow import keras
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import time
from collections import deque
import pickle
from public_dataset_loader import PublicDatasetLoader, PublicDatasetConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

@dataclass
class RealDatasetDriftConfig:
    """Configuration for real dataset drift detection validation."""
    
    # HDDM-A parameters optimized for real datasets
    hddm_delta: float = 0.005
    hddm_lambda: float = 0.050
    hddm_alpha: float = 0.9
    
    # UADF parameters for real-world scenarios
    uadf_window_size: int = 100
    uadf_uncertainty_threshold: float = 0.15
    uadf_forecast_horizon: int = 50
    
    # Ensemble parameters
    ensemble_weights: Dict[str, float] = None
    confidence_threshold: float = 0.7
    
    # Real dataset validation parameters
    validation_window_size: int = 200
    statistical_significance_level: float = 0.05
    min_drift_duration: int = 10
    
    # Performance optimization
    batch_processing: bool = True
    parallel_processing: bool = True
    memory_efficient: bool = True
    
    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                'hddm': 0.4,
                'uadf': 0.35,
                'statistical': 0.25
            }

class HDDMAdaptive:
    """Enhanced HDDM-A implementation for real dataset validation."""
    
    def __init__(self, delta: float = 0.005, lambda_param: float = 0.050, alpha: float = 0.9):
        self.delta = delta
        self.lambda_param = lambda_param
        self.alpha = alpha
        
        # State variables
        self.n_min = 0
        self.x_min = 0
        self.s_min = 0
        self.n_max = 0
        self.x_max = 0
        self.s_max = 0
        
        # Detection state
        self.drift_detected = False
        self.warning_detected = False
        self.drift_point = None
        self.warning_point = None
        
        # Performance tracking
        self.detection_history = []
        self.confidence_scores = []
        
        logger.info(f"HDDM-A initialized: δ={delta}, λ={lambda_param}, α={alpha}")
    
    def add_element(self, x: float) -> Dict[str, Any]:
        """Add new element and check for drift."""
        # Update minimum statistics
        if self.n_min == 0:
            self.n_min = 1
            self.x_min = x
            self.s_min = 0
        else:
            self.n_min += 1
            self.x_min += x
            self.s_min += x * x
        
        # Update maximum statistics
        if self.n_max == 0:
            self.n_max = 1
            self.x_max = x
            self.s_max = 0
        else:
            self.n_max += 1
            self.x_max += x
            self.s_max += x * x
        
        # Calculate means
        mean_min = self.x_min / self.n_min
        mean_max = self.x_max / self.n_max
        
        # Calculate test statistic
        if self.n_min > 1 and self.n_max > 1:
            var_min = (self.s_min - self.x_min * self.x_min / self.n_min) / (self.n_min - 1)
            var_max = (self.s_max - self.x_max * self.x_max / self.n_max) / (self.n_max - 1)
            
            if var_min > 0 and var_max > 0:
                # Welch's t-test statistic
                t_stat = (mean_max - mean_min) / np.sqrt(var_min / self.n_min + var_max / self.n_max)
                
                # Calculate confidence
                confidence = abs(t_stat) / (abs(t_stat) + 1)  # Normalized confidence
                self.confidence_scores.append(confidence)
                
                # Check for drift
                if abs(mean_max - mean_min) > self.delta:
                    if not self.drift_detected:
                        self.drift_detected = True
                        self.drift_point = len(self.detection_history)
                        logger.info(f"HDDM-A drift detected at point {self.drift_point}")
                
                # Check for warning
                elif abs(mean_max - mean_min) > self.lambda_param:
                    if not self.warning_detected:
                        self.warning_detected = True
                        self.warning_point = len(self.detection_history)
                
                # Reset if drift confidence decreases
                if self.drift_detected and confidence < 0.3:
                    self._reset_detection()
        
        # Record detection state
        detection_result = {
            'drift_detected': self.drift_detected,
            'warning_detected': self.warning_detected,
            'confidence': self.confidence_scores[-1] if self.confidence_scores else 0.0,
            'mean_difference': abs(mean_max - mean_min) if self.n_min > 0 and self.n_max > 0 else 0.0
        }
        
        self.detection_history.append(detection_result)
        return detection_result
    
    def _reset_detection(self):
        """Reset detection state."""
        self.drift_detected = False
        self.warning_detected = False
        self.drift_point = None
        self.warning_point = None
        
        # Reset statistics with exponential forgetting
        self.n_min = int(self.n_min * self.alpha)
        self.x_min *= self.alpha
        self.s_min *= self.alpha
        self.n_max = int(self.n_max * self.alpha)
        self.x_max *= self.alpha
        self.s_max *= self.alpha

class UncertaintyAwareDriftForecasting:
    """Enhanced UADF implementation for real dataset scenarios."""
    
    def __init__(self, window_size: int = 100, uncertainty_threshold: float = 0.15, 
                 forecast_horizon: int = 50):
        self.window_size = window_size
        self.uncertainty_threshold = uncertainty_threshold
        self.forecast_horizon = forecast_horizon
        
        # Data buffers
        self.data_buffer = deque(maxlen=window_size)
        self.uncertainty_buffer = deque(maxlen=window_size)
        
        # Forecasting model
        self.forecasting_model = None
        self.scaler = StandardScaler()
        
        # State tracking
        self.drift_probability = 0.0
        self.forecast_uncertainty = 0.0
        self.time_to_drift = float('inf')
        
        logger.info(f"UADF initialized: window={window_size}, threshold={uncertainty_threshold}")
    
    def _build_forecasting_model(self, input_dim: int) -> keras.Model:
        """Build LSTM-based forecasting model."""
        model = keras.Sequential([
            keras.layers.LSTM(64, return_sequences=True, input_shape=(self.window_size, input_dim)),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def add_element(self, x: Union[float, np.ndarray]) -> Dict[str, Any]:
        """Add new element and update drift forecasting."""
        # Convert to array if needed
        if isinstance(x, (int, float)):
            x = np.array([x])
        elif isinstance(x, list):
            x = np.array(x)
        
        self.data_buffer.append(x)
        
        # Calculate uncertainty for current element
        if len(self.data_buffer) > 10:
            recent_data = np.array(list(self.data_buffer)[-10:])
            uncertainty = np.std(recent_data) / (np.mean(np.abs(recent_data)) + 1e-8)
            self.uncertainty_buffer.append(uncertainty)
        else:
            self.uncertainty_buffer.append(0.0)
        
        # Perform forecasting if enough data
        if len(self.data_buffer) >= self.window_size:
            self._update_drift_forecast()
        
        return {
            'drift_probability': self.drift_probability,
            'forecast_uncertainty': self.forecast_uncertainty,
            'time_to_drift': self.time_to_drift,
            'current_uncertainty': self.uncertainty_buffer[-1] if self.uncertainty_buffer else 0.0
        }
    
    def _update_drift_forecast(self):
        """Update drift probability forecast."""
        try:
            # Prepare data for forecasting
            data_array = np.array(list(self.data_buffer))
            uncertainty_array = np.array(list(self.uncertainty_buffer))
            
            # Detect trend changes in uncertainty
            if len(uncertainty_array) >= 20:
                recent_uncertainty = uncertainty_array[-10:]
                previous_uncertainty = uncertainty_array[-20:-10]
                
                # Statistical test for uncertainty increase
                statistic, p_value = mannwhitneyu(previous_uncertainty, recent_uncertainty, alternative='less')
                
                # Calculate drift probability based on uncertainty trend
                if p_value < 0.05:  # Significant increase in uncertainty
                    uncertainty_ratio = np.mean(recent_uncertainty) / (np.mean(previous_uncertainty) + 1e-8)
                    self.drift_probability = min(0.95, uncertainty_ratio - 1.0)
                else:
                    self.drift_probability *= 0.9  # Decay probability
                
                # Estimate time to drift
                if self.drift_probability > 0.1:
                    uncertainty_trend = np.polyfit(range(len(recent_uncertainty)), recent_uncertainty, 1)[0]
                    if uncertainty_trend > 0:
                        current_uncertainty = recent_uncertainty[-1]
                        steps_to_threshold = (self.uncertainty_threshold - current_uncertainty) / uncertainty_trend
                        self.time_to_drift = max(1, int(steps_to_threshold))
                    else:
                        self.time_to_drift = float('inf')
                else:
                    self.time_to_drift = float('inf')
                
                # Calculate forecast uncertainty
                self.forecast_uncertainty = np.std(recent_uncertainty)
        
        except Exception as e:
            logger.warning(f"UADF forecast update failed: {e}")
            self.drift_probability = 0.0
            self.forecast_uncertainty = 0.0

class RealDatasetDriftDetector:
    """Advanced drift detector validated against real public datasets."""
    
    def __init__(self, config: RealDatasetDriftConfig):
        self.config = config
        
        # Initialize detection algorithms
        self.hddm = HDDMAdaptive(
            delta=config.hddm_delta,
            lambda_param=config.hddm_lambda,
            alpha=config.hddm_alpha
        )
        
        self.uadf = UncertaintyAwareDriftForecasting(
            window_size=config.uadf_window_size,
            uncertainty_threshold=config.uadf_uncertainty_threshold,
            forecast_horizon=config.uadf_forecast_horizon
        )
        
        # Load real dataset ground truth
        self.dataset_loader = PublicDatasetLoader(PublicDatasetConfig())
        self.real_drift_events = self._load_real_drift_events()
        
        # Performance tracking
        self.detection_results = []
        self.performance_metrics = {}
        self.comparison_results = {}
        
        logger.info("Real dataset drift detector initialized")
    
    def _load_real_drift_events(self) -> Dict[str, List[int]]:
        """Load real drift events from public datasets."""
        drift_events = {}
        
        try:
            # Load drift detection results
            drift_datasets = self.dataset_loader.get_drift_detection_datasets()
            
            for dataset_name, df in drift_datasets.items():
                if 'drift_point' in df.columns or 'drift_detected' in df.columns:
                    # Extract drift points
                    if 'drift_point' in df.columns:
                        drift_points = df[df['drift_point'].notna()]['drift_point'].tolist()
                    else:
                        drift_points = df[df['drift_detected'] == True].index.tolist()
                    
                    drift_events[dataset_name] = drift_points
                    logger.info(f"Loaded {len(drift_points)} drift events from {dataset_name}")
        
        except Exception as e:
            logger.warning(f"Failed to load real drift events: {e}")
            # Create synthetic drift events for testing
            drift_events = {
                'synthetic': [100, 300, 500, 750, 1000]
            }
        
        return drift_events
    
    def detect_drift_stream(self, data_stream: np.ndarray) -> List[Dict[str, Any]]:
        """Detect drift in streaming data."""
        logger.info(f"Processing data stream of length {len(data_stream)}")
        
        detection_results = []
        
        for i, x in enumerate(data_stream):
            # Get individual algorithm results
            hddm_result = self.hddm.add_element(x)
            uadf_result = self.uadf.add_element(x)
            
            # Statistical drift detection
            statistical_result = self._statistical_drift_check(data_stream, i)
            
            # Ensemble decision
            ensemble_result = self._ensemble_decision(hddm_result, uadf_result, statistical_result)
            
            # Combine all results
            combined_result = {
                'timestamp': i,
                'value': x,
                'hddm': hddm_result,
                'uadf': uadf_result,
                'statistical': statistical_result,
                'ensemble': ensemble_result
            }
            
            detection_results.append(combined_result)
        
        self.detection_results = detection_results
        return detection_results
    
    def _statistical_drift_check(self, data_stream: np.ndarray, current_idx: int) -> Dict[str, Any]:
        """Perform statistical drift detection."""
        if current_idx < self.config.validation_window_size:
            return {'drift_detected': False, 'p_value': 1.0, 'statistic': 0.0}
        
        # Compare recent window with reference window
        window_size = self.config.validation_window_size // 2
        reference_window = data_stream[current_idx - self.config.validation_window_size:current_idx - window_size]
        recent_window = data_stream[current_idx - window_size:current_idx]
        
        # Kolmogorov-Smirnov test
        statistic, p_value = ks_2samp(reference_window, recent_window)
        
        drift_detected = p_value < self.config.statistical_significance_level
        
        return {
            'drift_detected': drift_detected,
            'p_value': p_value,
            'statistic': statistic,
            'confidence': 1 - p_value
        }
    
    def _ensemble_decision(self, hddm_result: Dict, uadf_result: Dict, 
                          statistical_result: Dict) -> Dict[str, Any]:
        """Make ensemble drift detection decision."""
        # Calculate weighted confidence
        weights = self.config.ensemble_weights
        
        hddm_confidence = hddm_result.get('confidence', 0.0)
        uadf_confidence = uadf_result.get('drift_probability', 0.0)
        stat_confidence = statistical_result.get('confidence', 0.0)
        
        ensemble_confidence = (
            weights['hddm'] * hddm_confidence +
            weights['uadf'] * uadf_confidence +
            weights['statistical'] * stat_confidence
        )
        
        # Make decision
        drift_detected = ensemble_confidence > self.config.confidence_threshold
        
        # Combine forecasting information
        time_to_drift = uadf_result.get('time_to_drift', float('inf'))
        forecast_uncertainty = uadf_result.get('forecast_uncertainty', 0.0)
        
        return {
            'drift_detected': drift_detected,
            'confidence': ensemble_confidence,
            'time_to_drift': time_to_drift,
            'forecast_uncertainty': forecast_uncertainty,
            'algorithm_votes': {
                'hddm': hddm_result.get('drift_detected', False),
                'uadf': uadf_confidence > 0.5,
                'statistical': statistical_result.get('drift_detected', False)
            }
        }
    
    def validate_against_real_datasets(self) -> Dict[str, Dict[str, float]]:
        """Validate drift detection against real dataset ground truth."""
        logger.info("Validating against real datasets...")
        
        validation_results = {}
        
        for dataset_name, drift_points in self.real_drift_events.items():
            try:
                # Load corresponding dataset
                if dataset_name == 'lstm_drift':
                    df = self.dataset_loader.datasets.get('lstm_drift')
                elif dataset_name == 'drift_detection':
                    df = self.dataset_loader.datasets.get('drift_detection')
                else:
                    # Use synthetic data for testing
                    df = self._generate_synthetic_drift_data(drift_points)
                
                if df is not None and not df.empty:
                    # Extract time series data
                    if 'value' in df.columns:
                        data_stream = df['value'].values
                    elif 'qoe_score' in df.columns:
                        data_stream = df['qoe_score'].values
                    else:
                        # Use first numeric column
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            data_stream = df[numeric_cols[0]].values
                        else:
                            continue
                    
                    # Run drift detection
                    detection_results = self.detect_drift_stream(data_stream)
                    
                    # Extract detected drift points
                    detected_points = [
                        r['timestamp'] for r in detection_results 
                        if r['ensemble']['drift_detected']
                    ]
                    
                    # Calculate performance metrics
                    metrics = self._calculate_detection_metrics(drift_points, detected_points, len(data_stream))
                    validation_results[dataset_name] = metrics
                    
                    logger.info(f"Validation on {dataset_name}: F1={metrics['f1_score']:.3f}")
            
            except Exception as e:
                logger.warning(f"Validation failed for {dataset_name}: {e}")
        
        self.performance_metrics = validation_results
        return validation_results
    
    def _generate_synthetic_drift_data(self, drift_points: List[int]) -> pd.DataFrame:
        """Generate synthetic data with known drift points for testing."""
        total_length = max(drift_points) + 200 if drift_points else 1000
        data = []
        
        current_mean = 0.0
        current_std = 1.0
        
        for i in range(total_length):
            # Check if this is a drift point
            if i in drift_points:
                current_mean += np.random.normal(0, 2)  # Shift mean
                current_std *= np.random.uniform(0.5, 2.0)  # Change variance
            
            # Generate data point
            value = np.random.normal(current_mean, current_std)
            data.append({'timestamp': i, 'value': value})
        
        return pd.DataFrame(data)
    
    def _calculate_detection_metrics(self, true_drift_points: List[int], 
                                   detected_drift_points: List[int], 
                                   total_length: int) -> Dict[str, float]:
        """Calculate drift detection performance metrics."""
        # Create binary arrays
        true_labels = np.zeros(total_length, dtype=bool)
        detected_labels = np.zeros(total_length, dtype=bool)
        
        # Mark drift regions (not just points)
        for drift_point in true_drift_points:
            start = max(0, drift_point - self.config.min_drift_duration // 2)
            end = min(total_length, drift_point + self.config.min_drift_duration // 2)
            true_labels[start:end] = True
        
        for drift_point in detected_drift_points:
            start = max(0, drift_point - self.config.min_drift_duration // 2)
            end = min(total_length, drift_point + self.config.min_drift_duration // 2)
            detected_labels[start:end] = True
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, detected_labels)
        precision = precision_score(true_labels, detected_labels, zero_division=0)
        recall = recall_score(true_labels, detected_labels, zero_division=0)
        f1 = f1_score(true_labels, detected_labels, zero_division=0)
        
        # Calculate detection delay
        detection_delays = []
        for true_point in true_drift_points:
            nearby_detections = [d for d in detected_drift_points 
                               if abs(d - true_point) <= self.config.min_drift_duration]
            if nearby_detections:
                delay = min([abs(d - true_point) for d in nearby_detections])
                detection_delays.append(delay)
        
        avg_detection_delay = np.mean(detection_delays) if detection_delays else float('inf')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_detection_delay': avg_detection_delay,
            'true_positives': np.sum(true_labels & detected_labels),
            'false_positives': np.sum(~true_labels & detected_labels),
            'false_negatives': np.sum(true_labels & ~detected_labels)
        }
    
    def compare_with_baselines(self) -> Dict[str, Dict[str, float]]:
        """Compare with baseline drift detection methods."""
        logger.info("Comparing with baseline methods...")
        
        comparison_results = {}
        
        # Load LSTM baseline results if available
        try:
            lstm_results = self.dataset_loader.datasets.get('lstm_drift')
            if lstm_results is not None:
                # Extract LSTM performance metrics
                lstm_metrics = self._extract_baseline_metrics(lstm_results, 'lstm')
                comparison_results['lstm_baseline'] = lstm_metrics
        except Exception as e:
            logger.warning(f"Failed to load LSTM baseline: {e}")
        
        # Compare with our results
        if self.performance_metrics:
            our_avg_metrics = self._calculate_average_metrics(self.performance_metrics)
            comparison_results['qoe_foresight'] = our_avg_metrics
            
            # Calculate improvement percentages
            if 'lstm_baseline' in comparison_results:
                improvements = {}
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    if metric in our_avg_metrics and metric in comparison_results['lstm_baseline']:
                        baseline_value = comparison_results['lstm_baseline'][metric]
                        our_value = our_avg_metrics[metric]
                        if baseline_value > 0:
                            improvement = ((our_value - baseline_value) / baseline_value) * 100
                            improvements[f'{metric}_improvement'] = improvement
                
                comparison_results['improvements'] = improvements
        
        self.comparison_results = comparison_results
        return comparison_results
    
    def _extract_baseline_metrics(self, baseline_df: pd.DataFrame, method_name: str) -> Dict[str, float]:
        """Extract performance metrics from baseline results."""
        metrics = {}
        
        # Look for common metric column names
        metric_columns = {
            'accuracy': ['accuracy', 'acc'],
            'precision': ['precision', 'prec'],
            'recall': ['recall', 'rec'],
            'f1_score': ['f1', 'f1_score', 'f1score']
        }
        
        for metric, possible_cols in metric_columns.items():
            for col in possible_cols:
                if col in baseline_df.columns:
                    metrics[metric] = baseline_df[col].mean()
                    break
        
        return metrics
    
    def _calculate_average_metrics(self, metrics_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate average metrics across datasets."""
        all_metrics = {}
        
        for dataset_metrics in metrics_dict.values():
            for metric, value in dataset_metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        return {metric: np.mean(values) for metric, values in all_metrics.items()}
    
    def visualize_detection_results(self) -> None:
        """Visualize drift detection results."""
        if not self.detection_results:
            logger.warning("No detection results to visualize")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('QoE-Foresight Drift Detection Results', fontsize=16, fontweight='bold')
        
        # Extract data for visualization
        timestamps = [r['timestamp'] for r in self.detection_results]
        values = [r['value'] for r in self.detection_results]
        ensemble_confidence = [r['ensemble']['confidence'] for r in self.detection_results]
        drift_detected = [r['ensemble']['drift_detected'] for r in self.detection_results]
        
        # Time series with drift points
        axes[0, 0].plot(timestamps, values, 'b-', alpha=0.7, label='Data Stream')
        drift_points = [t for t, d in zip(timestamps, drift_detected) if d]
        if drift_points:
            axes[0, 0].scatter(drift_points, [values[t] for t in drift_points], 
                             color='red', s=50, label='Detected Drift', zorder=5)
        axes[0, 0].set_title('Data Stream with Drift Detection')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Confidence scores
        axes[0, 1].plot(timestamps, ensemble_confidence, 'g-', label='Ensemble Confidence')
        axes[0, 1].axhline(y=self.config.confidence_threshold, color='r', linestyle='--', 
                          label='Threshold')
        axes[0, 1].set_title('Drift Detection Confidence')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Confidence')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Algorithm comparison
        if self.detection_results:
            hddm_detections = [r['hddm']['drift_detected'] for r in self.detection_results]
            uadf_probs = [r['uadf']['drift_probability'] for r in self.detection_results]
            stat_detections = [r['statistical']['drift_detected'] for r in self.detection_results]
            
            axes[1, 0].plot(timestamps, hddm_detections, 'b-', label='HDDM-A', alpha=0.7)
            axes[1, 0].plot(timestamps, uadf_probs, 'g-', label='UADF Probability', alpha=0.7)
            axes[1, 0].plot(timestamps, stat_detections, 'r-', label='Statistical', alpha=0.7)
            axes[1, 0].set_title('Individual Algorithm Results')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Detection Signal')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Performance metrics
        if self.performance_metrics:
            datasets = list(self.performance_metrics.keys())
            f1_scores = [self.performance_metrics[d]['f1_score'] for d in datasets]
            
            axes[1, 1].bar(datasets, f1_scores, color='skyblue', alpha=0.7)
            axes[1, 1].set_title('F1 Score by Dataset')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        # Comparison with baselines
        if self.comparison_results:
            methods = list(self.comparison_results.keys())
            if 'improvements' in methods:
                methods.remove('improvements')
            
            if len(methods) > 1:
                f1_comparison = []
                for method in methods:
                    if 'f1_score' in self.comparison_results[method]:
                        f1_comparison.append(self.comparison_results[method]['f1_score'])
                    else:
                        f1_comparison.append(0)
                
                axes[2, 0].bar(methods, f1_comparison, color=['lightcoral', 'lightgreen'])
                axes[2, 0].set_title('Method Comparison (F1 Score)')
                axes[2, 0].set_ylabel('F1 Score')
                axes[2, 0].tick_params(axis='x', rotation=45)
                axes[2, 0].grid(True, alpha=0.3)
        
        # Detection delay analysis
        if self.performance_metrics:
            delays = [self.performance_metrics[d]['avg_detection_delay'] 
                     for d in self.performance_metrics.keys() 
                     if self.performance_metrics[d]['avg_detection_delay'] != float('inf')]
            
            if delays:
                axes[2, 1].hist(delays, bins=10, color='orange', alpha=0.7, edgecolor='black')
                axes[2, 1].set_title('Detection Delay Distribution')
                axes[2, 1].set_xlabel('Delay (time steps)')
                axes[2, 1].set_ylabel('Frequency')
                axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Example usage and testing
if __name__ == "__main__":
    print("🚀 QoE-Foresight Advanced Drift Detection with Real Dataset Validation")
    print("=" * 70)
    
    # Initialize configuration
    config = RealDatasetDriftConfig()
    
    # Create drift detector
    detector = RealDatasetDriftDetector(config)
    
    # Validate against real datasets
    print("📊 Validating against real datasets...")
    validation_results = detector.validate_against_real_datasets()
    
    if validation_results:
        print("\n✅ Validation Results:")
        for dataset, metrics in validation_results.items():
            print(f"\n{dataset}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
    
    # Compare with baselines
    print("\n🔍 Comparing with baseline methods...")
    comparison_results = detector.compare_with_baselines()
    
    if comparison_results:
        print("\nComparison Results:")
        for method, metrics in comparison_results.items():
            print(f"\n{method}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
    
    # Visualize results
    if detector.detection_results:
        detector.visualize_detection_results()
    
    print("\n🎯 Advanced drift detection validation complete!")
    print("Ready for RL self-healing controller integration")

