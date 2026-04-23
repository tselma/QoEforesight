"""
QoE-Foresight: Advanced Drift Detection Engine
Implements HDDM-A, Uncertainty-Aware Dynamic Filter (UADF), and QoE Deviation Quantifier

This module provides sophisticated drift detection capabilities for proactive QoE management,
combining statistical methods with machine learning for robust drift identification.
"""

import os
import json
import time
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from collections import deque, defaultdict
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DriftType(Enum):
    """Types of concept drift"""
    NONE = "none"
    ABRUPT = "abrupt"
    GRADUAL = "gradual"
    INCREMENTAL = "incremental"
    RECURRING = "recurring"

class DriftSeverity(Enum):
    """Severity levels of drift"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DriftEvent:
    """Represents a detected drift event"""
    timestamp: float
    drift_type: DriftType
    severity: DriftSeverity
    confidence: float
    affected_features: List[str]
    trigger_source: str  # network/device/application
    magnitude: float
    persistence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DriftDetectionConfig:
    """Configuration for drift detection components"""
    # HDDM-A Configuration
    hddm_drift_threshold: float = 0.005
    hddm_warning_threshold: float = 0.002
    hddm_lambda: float = 0.05
    hddm_two_sided: bool = True
    
    # UADF Configuration
    uadf_confidence_level: float = 0.95
    uadf_window_size: int = 50
    uadf_adaptation_rate: float = 0.1
    uadf_sensitivity_factor: float = 1.5
    
    # QoE Prediction Configuration
    qoe_sequence_length: int = 15
    qoe_prediction_horizon: int = 5
    qoe_lstm_units: int = 128
    qoe_dropout_rate: float = 0.25
    qoe_learning_rate: float = 0.001
    
    # Deviation Quantifier Configuration
    deviation_window_size: int = 30
    deviation_threshold_multiplier: float = 2.0
    persistence_threshold: int = 3
    
    # General Configuration
    feature_importance_threshold: float = 0.1
    drift_buffer_size: int = 1000
    enable_ensemble: bool = True
    ensemble_voting_threshold: float = 0.6

class HoeffdingDriftDetectionMethod:
    """
    Implementation of Hoeffding Drift Detection Method with Adaptation (HDDM-A)
    for detecting concept drift in streaming data
    """
    
    def __init__(self, config: DriftDetectionConfig):
        self.config = config
        self.drift_threshold = config.hddm_drift_threshold
        self.warning_threshold = config.hddm_warning_threshold
        self.lambda_param = config.hddm_lambda
        self.two_sided = config.hddm_two_sided
        
        # Internal state
        self.n_samples = 0
        self.sum_errors = 0.0
        self.sum_squared_errors = 0.0
        self.mean_error = 0.0
        self.variance_error = 0.0
        self.std_error = 0.0
        
        # Drift detection state
        self.in_warning = False
        self.in_drift = False
        self.warning_start = None
        self.drift_start = None
        
        # Statistics tracking
        self.error_history = deque(maxlen=1000)
        self.drift_history = []
        
        logger.info("HDDM-A drift detector initialized")
    
    def _update_statistics(self, error: float):
        """Update running statistics with new error"""
        self.n_samples += 1
        self.sum_errors += error
        self.sum_squared_errors += error * error
        
        # Update mean and variance
        self.mean_error = self.sum_errors / self.n_samples
        if self.n_samples > 1:
            self.variance_error = (self.sum_squared_errors - self.n_samples * self.mean_error * self.mean_error) / (self.n_samples - 1)
            self.std_error = np.sqrt(max(0, self.variance_error))
        
        self.error_history.append(error)
    
    def _compute_hoeffding_bound(self, n: int, confidence: float) -> float:
        """Compute Hoeffding bound for given sample size and confidence"""
        if n <= 1:
            return float('inf')
        
        # Hoeffding bound: sqrt(ln(1/delta) / (2*n))
        # where delta = 1 - confidence
        delta = 1.0 - confidence
        bound = np.sqrt(np.log(1.0 / delta) / (2.0 * n))
        return bound
    
    def _detect_drift_condition(self) -> Tuple[bool, bool, float]:
        """Detect drift and warning conditions"""
        if self.n_samples < 30:  # Need minimum samples
            return False, False, 0.0
        
        # Compute current statistics
        current_mean = self.mean_error
        current_std = self.std_error
        
        if current_std == 0:
            return False, False, 0.0
        
        # Compute adaptive thresholds using Hoeffding bound
        warning_bound = self._compute_hoeffding_bound(self.n_samples, 1 - self.warning_threshold)
        drift_bound = self._compute_hoeffding_bound(self.n_samples, 1 - self.drift_threshold)
        
        # Compute test statistic (normalized error change)
        if len(self.error_history) >= 10:
            recent_errors = list(self.error_history)[-10:]
            recent_mean = np.mean(recent_errors)
            
            # Normalized difference
            test_statistic = abs(recent_mean - current_mean) / (current_std + 1e-8)
            
            # Check conditions
            warning_detected = test_statistic > warning_bound * self.config.uadf_sensitivity_factor
            drift_detected = test_statistic > drift_bound * self.config.uadf_sensitivity_factor
            
            return drift_detected, warning_detected, test_statistic
        
        return False, False, 0.0
    
    def update(self, prediction_error: float) -> Tuple[bool, bool, Dict[str, Any]]:
        """
        Update detector with new prediction error
        
        Returns:
            (drift_detected, warning_detected, detection_info)
        """
        # Update statistics
        self._update_statistics(prediction_error)
        
        # Detect drift conditions
        drift_detected, warning_detected, test_statistic = self._detect_drift_condition()
        
        # Update state
        current_time = time.time()
        
        if drift_detected and not self.in_drift:
            self.in_drift = True
            self.drift_start = current_time
            self.in_warning = False  # Clear warning state
            
            drift_info = {
                "type": "drift",
                "timestamp": current_time,
                "test_statistic": test_statistic,
                "n_samples": self.n_samples,
                "mean_error": self.mean_error,
                "std_error": self.std_error
            }
            self.drift_history.append(drift_info)
            
            logger.warning(f"Drift detected at sample {self.n_samples}, test_statistic: {test_statistic:.4f}")
            
        elif warning_detected and not self.in_warning and not self.in_drift:
            self.in_warning = True
            self.warning_start = current_time
            
            logger.info(f"Warning detected at sample {self.n_samples}, test_statistic: {test_statistic:.4f}")
        
        elif not warning_detected and not drift_detected:
            # Reset states if conditions are no longer met
            if self.in_warning:
                self.in_warning = False
                self.warning_start = None
            
            # Note: drift state is typically reset externally after handling
        
        detection_info = {
            "test_statistic": test_statistic,
            "n_samples": self.n_samples,
            "mean_error": self.mean_error,
            "std_error": self.std_error,
            "in_warning": self.in_warning,
            "in_drift": self.in_drift,
            "warning_duration": current_time - self.warning_start if self.warning_start else 0,
            "drift_duration": current_time - self.drift_start if self.drift_start else 0
        }
        
        return drift_detected, warning_detected, detection_info
    
    def reset(self):
        """Reset detector state"""
        self.n_samples = 0
        self.sum_errors = 0.0
        self.sum_squared_errors = 0.0
        self.mean_error = 0.0
        self.variance_error = 0.0
        self.std_error = 0.0
        self.in_warning = False
        self.in_drift = False
        self.warning_start = None
        self.drift_start = None
        self.error_history.clear()
        
        logger.info("HDDM-A detector reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics"""
        return {
            "n_samples": self.n_samples,
            "mean_error": self.mean_error,
            "std_error": self.std_error,
            "in_warning": self.in_warning,
            "in_drift": self.in_drift,
            "total_drifts": len(self.drift_history),
            "drift_history": self.drift_history[-10:]  # Last 10 drifts
        }

class UncertaintyAwareDynamicFilter:
    """
    Uncertainty-Aware Dynamic Filter (UADF) for adaptive drift detection
    Adjusts sensitivity based on prediction confidence and contextual factors
    """
    
    def __init__(self, config: DriftDetectionConfig):
        self.config = config
        self.confidence_level = config.uadf_confidence_level
        self.window_size = config.uadf_window_size
        self.adaptation_rate = config.uadf_adaptation_rate
        self.sensitivity_factor = config.uadf_sensitivity_factor
        
        # Dynamic thresholds
        self.current_threshold = 0.5
        self.threshold_history = deque(maxlen=100)
        
        # Uncertainty estimation
        self.prediction_history = deque(maxlen=self.window_size)
        self.confidence_history = deque(maxlen=self.window_size)
        self.error_history = deque(maxlen=self.window_size)
        
        # Context tracking
        self.context_features = deque(maxlen=self.window_size)
        self.feature_importance = {}
        
        # Isolation Forest for anomaly detection
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_fitted = False
        
        logger.info("UADF initialized")
    
    def _estimate_prediction_uncertainty(self, predictions: np.ndarray, 
                                       confidence_scores: Optional[np.ndarray] = None) -> float:
        """Estimate uncertainty in predictions"""
        if len(predictions) < 2:
            return 1.0  # High uncertainty for insufficient data
        
        # Variance-based uncertainty
        prediction_variance = np.var(predictions)
        
        # Confidence-based uncertainty (if available)
        if confidence_scores is not None and len(confidence_scores) > 0:
            avg_confidence = np.mean(confidence_scores)
            confidence_uncertainty = 1.0 - avg_confidence
        else:
            confidence_uncertainty = 0.5  # Neutral uncertainty
        
        # Trend-based uncertainty
        if len(predictions) >= 5:
            # Compute trend stability
            trend_changes = np.diff(np.sign(np.diff(predictions)))
            trend_instability = np.sum(np.abs(trend_changes)) / len(trend_changes)
        else:
            trend_instability = 0.5
        
        # Combined uncertainty
        combined_uncertainty = (
            0.4 * prediction_variance + 
            0.4 * confidence_uncertainty + 
            0.2 * trend_instability
        )
        
        return np.clip(combined_uncertainty, 0.0, 1.0)
    
    def _compute_contextual_factors(self, features: np.ndarray) -> Dict[str, float]:
        """Compute contextual factors affecting drift sensitivity"""
        if len(features) == 0:
            return {"stability": 0.5, "complexity": 0.5, "novelty": 0.5}
        
        # Feature stability (low variance indicates stability)
        feature_variance = np.var(features, axis=0) if features.ndim > 1 else np.var(features)
        stability = 1.0 / (1.0 + np.mean(feature_variance))
        
        # Feature complexity (number of significant features)
        if features.ndim > 1:
            significant_features = np.sum(np.std(features, axis=0) > 0.1)
            complexity = significant_features / features.shape[1]
        else:
            complexity = 0.5
        
        # Novelty detection using anomaly detector
        if self.anomaly_fitted and features.ndim > 1:
            try:
                anomaly_scores = self.anomaly_detector.decision_function(features)
                novelty = 1.0 - np.mean(anomaly_scores)  # Higher score = less novel
            except:
                novelty = 0.5
        else:
            novelty = 0.5
        
        return {
            "stability": np.clip(stability, 0.0, 1.0),
            "complexity": np.clip(complexity, 0.0, 1.0),
            "novelty": np.clip(novelty, 0.0, 1.0)
        }
    
    def _adapt_threshold(self, uncertainty: float, context: Dict[str, float], 
                        error_magnitude: float) -> float:
        """Adapt detection threshold based on uncertainty and context"""
        base_threshold = self.current_threshold
        
        # Uncertainty adjustment (higher uncertainty -> lower threshold)
        uncertainty_factor = 1.0 - uncertainty * 0.5
        
        # Context adjustments
        stability_factor = 1.0 + (context["stability"] - 0.5) * 0.3
        complexity_factor = 1.0 - (context["complexity"] - 0.5) * 0.2
        novelty_factor = 1.0 - (context["novelty"] - 0.5) * 0.4
        
        # Error magnitude adjustment
        error_factor = 1.0 + np.tanh(error_magnitude) * 0.2
        
        # Combined adaptation
        adaptation_factor = (
            uncertainty_factor * 
            stability_factor * 
            complexity_factor * 
            novelty_factor * 
            error_factor
        )
        
        # Apply adaptation with learning rate
        new_threshold = base_threshold * (1 - self.adaptation_rate) + \
                       (base_threshold * adaptation_factor) * self.adaptation_rate
        
        # Clamp threshold to reasonable range
        new_threshold = np.clip(new_threshold, 0.1, 2.0)
        
        return new_threshold
    
    def update(self, prediction: float, actual: float, confidence: Optional[float] = None,
               features: Optional[np.ndarray] = None) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Update filter with new prediction and actual value
        
        Returns:
            (drift_detected, adapted_threshold, filter_info)
        """
        error = abs(prediction - actual)
        
        # Store in history
        self.prediction_history.append(prediction)
        self.error_history.append(error)
        if confidence is not None:
            self.confidence_history.append(confidence)
        if features is not None:
            self.context_features.append(features)
        
        # Estimate uncertainty
        predictions_array = np.array(list(self.prediction_history))
        confidences_array = np.array(list(self.confidence_history)) if self.confidence_history else None
        uncertainty = self._estimate_prediction_uncertainty(predictions_array, confidences_array)
        
        # Compute contextual factors
        if self.context_features and len(self.context_features) > 5:
            features_array = np.array(list(self.context_features))
            context = self._compute_contextual_factors(features_array)
            
            # Fit anomaly detector if not already fitted
            if not self.anomaly_fitted and len(self.context_features) >= 20:
                try:
                    self.anomaly_detector.fit(features_array)
                    self.anomaly_fitted = True
                    logger.info("UADF anomaly detector fitted")
                except:
                    logger.warning("Failed to fit UADF anomaly detector")
        else:
            context = {"stability": 0.5, "complexity": 0.5, "novelty": 0.5}
        
        # Adapt threshold
        adapted_threshold = self._adapt_threshold(uncertainty, context, error)
        self.current_threshold = adapted_threshold
        self.threshold_history.append(adapted_threshold)
        
        # Detect drift
        drift_detected = error > adapted_threshold
        
        filter_info = {
            "uncertainty": uncertainty,
            "context": context,
            "adapted_threshold": adapted_threshold,
            "error": error,
            "confidence": confidence,
            "drift_detected": drift_detected
        }
        
        return drift_detected, adapted_threshold, filter_info
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get filter statistics"""
        return {
            "current_threshold": self.current_threshold,
            "avg_threshold": np.mean(list(self.threshold_history)) if self.threshold_history else 0,
            "threshold_std": np.std(list(self.threshold_history)) if self.threshold_history else 0,
            "anomaly_fitted": self.anomaly_fitted,
            "window_fill": len(self.prediction_history) / self.window_size,
            "recent_errors": list(self.error_history)[-10:] if self.error_history else []
        }

class QoEPredictionModel:
    """
    LSTM-based QoE prediction model with attention mechanism
    Provides base QoE predictions and confidence estimates
    """
    
    def __init__(self, config: DriftDetectionConfig, feature_dim: int):
        self.config = config
        self.feature_dim = feature_dim
        self.sequence_length = config.qoe_sequence_length
        self.prediction_horizon = config.qoe_prediction_horizon
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Training history
        self.training_history = []
        self.prediction_history = deque(maxlen=1000)
        
        # Build model
        self._build_model()
        
        logger.info(f"QoE prediction model initialized with feature_dim={feature_dim}")
    
    def _build_model(self):
        """Build LSTM model with attention mechanism"""
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.feature_dim))
        
        # LSTM layers with return sequences for attention
        lstm1 = LSTM(self.config.qoe_lstm_units, return_sequences=True, dropout=self.config.qoe_dropout_rate)(inputs)
        lstm2 = LSTM(self.config.qoe_lstm_units // 2, return_sequences=True, dropout=self.config.qoe_dropout_rate)(lstm1)
        
        # Multi-head attention
        attention = MultiHeadAttention(num_heads=4, key_dim=self.config.qoe_lstm_units // 8)(lstm2, lstm2)
        
        # Global average pooling
        pooled = tf.keras.layers.GlobalAveragePooling1D()(attention)
        
        # Dense layers
        dense1 = Dense(64, activation='relu')(pooled)
        dropout1 = Dropout(self.config.qoe_dropout_rate)(dense1)
        dense2 = Dense(32, activation='relu')(dropout1)
        dropout2 = Dropout(self.config.qoe_dropout_rate)(dense2)
        
        # Output layers (prediction + confidence)
        qoe_output = Dense(1, activation='linear', name='qoe_prediction')(dropout2)
        confidence_output = Dense(1, activation='sigmoid', name='confidence')(dropout2)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=[qoe_output, confidence_output])
        
        # Compile with custom loss
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.qoe_learning_rate),
            loss={
                'qoe_prediction': 'mse',
                'confidence': 'binary_crossentropy'
            },
            loss_weights={'qoe_prediction': 1.0, 'confidence': 0.1},
            metrics={'qoe_prediction': ['mae'], 'confidence': ['accuracy']}
        )
    
    def _prepare_sequences(self, features: np.ndarray, targets: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare feature sequences for training/prediction"""
        if len(features) < self.sequence_length:
            return np.array([]), None if targets is None else np.array([])
        
        X = []
        y = [] if targets is not None else None
        
        for i in range(len(features) - self.sequence_length + 1):
            X.append(features[i:i + self.sequence_length])
            if targets is not None:
                y.append(targets[i + self.sequence_length - 1])
        
        X = np.array(X)
        y = np.array(y) if y is not None else None
        
        return X, y
    
    def fit(self, features: np.ndarray, targets: np.ndarray, validation_split: float = 0.2) -> Dict[str, Any]:
        """Train the QoE prediction model"""
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Prepare sequences
        X, y = self._prepare_sequences(features_scaled, targets)
        
        if len(X) == 0:
            raise ValueError("Insufficient data for sequence preparation")
        
        # Prepare confidence targets (high confidence for low error)
        y_confidence = np.ones(len(y))  # Start with high confidence
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
        
        # Train model
        history = self.model.fit(
            X, {'qoe_prediction': y, 'confidence': y_confidence},
            validation_split=validation_split,
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_fitted = True
        self.training_history.append(history.history)
        
        logger.info(f"QoE model trained on {len(X)} sequences")
        
        return history.history
    
    def predict(self, features: np.ndarray, return_confidence: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make QoE predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Prepare sequences
        X, _ = self._prepare_sequences(features_scaled)
        
        if len(X) == 0:
            if return_confidence:
                return np.array([]), np.array([])
            return np.array([])
        
        # Make predictions
        predictions = self.model.predict(X, verbose=0)
        qoe_pred = predictions[0].flatten()
        confidence = predictions[1].flatten()
        
        # Store in history
        for i, (qoe, conf) in enumerate(zip(qoe_pred, confidence)):
            self.prediction_history.append({
                'timestamp': time.time(),
                'qoe_prediction': qoe,
                'confidence': conf
            })
        
        if return_confidence:
            return qoe_pred, confidence
        return qoe_pred
    
    def update_online(self, features: np.ndarray, target: float) -> Dict[str, Any]:
        """Online update of the model with new data"""
        if not self.is_fitted:
            return {"status": "model_not_fitted"}
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Prepare sequence (use last sequence_length-1 + new features)
        if len(self.prediction_history) >= self.sequence_length - 1:
            # Get recent features from history (this is simplified - in practice, store feature history)
            X = features_scaled.reshape(1, 1, -1)  # Simplified for demo
            y_qoe = np.array([target])
            y_conf = np.array([1.0])  # High confidence for observed data
            
            # Perform one step of training
            loss = self.model.train_on_batch(
                X, {'qoe_prediction': y_qoe, 'confidence': y_conf}
            )
            
            return {"status": "updated", "loss": loss}
        
        return {"status": "insufficient_history"}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get model statistics"""
        recent_predictions = list(self.prediction_history)[-100:]
        
        stats = {
            "is_fitted": self.is_fitted,
            "total_predictions": len(self.prediction_history),
            "recent_predictions": len(recent_predictions),
            "training_epochs": len(self.training_history)
        }
        
        if recent_predictions:
            qoe_values = [p['qoe_prediction'] for p in recent_predictions]
            conf_values = [p['confidence'] for p in recent_predictions]
            
            stats.update({
                "avg_qoe_prediction": np.mean(qoe_values),
                "std_qoe_prediction": np.std(qoe_values),
                "avg_confidence": np.mean(conf_values),
                "std_confidence": np.std(conf_values)
            })
        
        return stats

class QoEDeviationQuantifier:
    """
    Quantifies QoE deviation magnitude and persistence
    Measures sustained divergences between predicted and observed QoE
    """
    
    def __init__(self, config: DriftDetectionConfig):
        self.config = config
        self.window_size = config.deviation_window_size
        self.threshold_multiplier = config.deviation_threshold_multiplier
        self.persistence_threshold = config.persistence_threshold
        
        # Deviation tracking
        self.residual_history = deque(maxlen=self.window_size)
        self.deviation_events = []
        self.current_deviation = None
        
        # Statistics
        self.baseline_stats = {"mean": 0.0, "std": 1.0}
        self.adaptive_threshold = 1.0
        
        logger.info("QoE deviation quantifier initialized")
    
    def _update_baseline_stats(self):
        """Update baseline statistics from residual history"""
        if len(self.residual_history) >= 10:
            residuals = np.array(list(self.residual_history))
            self.baseline_stats["mean"] = np.mean(residuals)
            self.baseline_stats["std"] = np.std(residuals)
            
            # Update adaptive threshold
            self.adaptive_threshold = self.baseline_stats["std"] * self.threshold_multiplier
    
    def _detect_sustained_deviation(self, current_residual: float) -> Tuple[bool, Dict[str, Any]]:
        """Detect sustained deviation from baseline"""
        if len(self.residual_history) < self.persistence_threshold:
            return False, {}
        
        # Check if current residual exceeds threshold
        threshold_exceeded = abs(current_residual) > self.adaptive_threshold
        
        if not threshold_exceeded:
            return False, {}
        
        # Check persistence - how many recent residuals exceed threshold
        recent_residuals = list(self.residual_history)[-self.persistence_threshold:]
        persistent_count = sum(1 for r in recent_residuals if abs(r) > self.adaptive_threshold)
        
        is_persistent = persistent_count >= self.persistence_threshold
        
        deviation_info = {
            "magnitude": abs(current_residual),
            "threshold": self.adaptive_threshold,
            "persistence_count": persistent_count,
            "persistence_ratio": persistent_count / self.persistence_threshold,
            "is_persistent": is_persistent
        }
        
        return is_persistent, deviation_info
    
    def _classify_deviation_type(self, residuals: List[float]) -> DriftType:
        """Classify the type of deviation based on residual pattern"""
        if len(residuals) < 5:
            return DriftType.NONE
        
        residuals_array = np.array(residuals)
        
        # Check for abrupt change (sudden jump)
        diff = np.diff(residuals_array)
        max_jump = np.max(np.abs(diff))
        mean_change = np.mean(np.abs(diff))
        
        if max_jump > 3 * mean_change:
            return DriftType.ABRUPT
        
        # Check for gradual trend
        if len(residuals) >= 10:
            # Linear trend test
            x = np.arange(len(residuals))
            slope, _, r_value, _, _ = stats.linregress(x, residuals_array)
            
            if abs(r_value) > 0.7 and abs(slope) > 0.1:
                return DriftType.GRADUAL
        
        # Check for incremental changes
        increasing_trend = np.sum(diff > 0) > len(diff) * 0.7
        decreasing_trend = np.sum(diff < 0) > len(diff) * 0.7
        
        if increasing_trend or decreasing_trend:
            return DriftType.INCREMENTAL
        
        # Default to gradual if persistent but no clear pattern
        return DriftType.GRADUAL
    
    def _compute_severity(self, magnitude: float, persistence: float) -> DriftSeverity:
        """Compute deviation severity based on magnitude and persistence"""
        # Normalize magnitude relative to baseline
        normalized_magnitude = magnitude / (self.baseline_stats["std"] + 1e-8)
        
        # Combine magnitude and persistence
        severity_score = 0.7 * normalized_magnitude + 0.3 * persistence
        
        if severity_score < 1.5:
            return DriftSeverity.LOW
        elif severity_score < 3.0:
            return DriftSeverity.MEDIUM
        elif severity_score < 5.0:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL
    
    def update(self, predicted_qoe: float, actual_qoe: float, 
               features: Optional[np.ndarray] = None) -> Tuple[bool, Optional[DriftEvent]]:
        """
        Update quantifier with new QoE prediction and actual value
        
        Returns:
            (deviation_detected, drift_event)
        """
        residual = actual_qoe - predicted_qoe
        self.residual_history.append(residual)
        
        # Update baseline statistics
        self._update_baseline_stats()
        
        # Detect sustained deviation
        is_persistent, deviation_info = self._detect_sustained_deviation(residual)
        
        current_time = time.time()
        
        if is_persistent:
            # Check if this is a new deviation or continuation
            if self.current_deviation is None:
                # New deviation detected
                recent_residuals = list(self.residual_history)[-self.persistence_threshold:]
                drift_type = self._classify_deviation_type(recent_residuals)
                severity = self._compute_severity(
                    deviation_info["magnitude"], 
                    deviation_info["persistence_ratio"]
                )
                
                # Create drift event
                drift_event = DriftEvent(
                    timestamp=current_time,
                    drift_type=drift_type,
                    severity=severity,
                    confidence=deviation_info["persistence_ratio"],
                    affected_features=["qoe"],  # Could be expanded with feature analysis
                    trigger_source="qoe_deviation",
                    magnitude=deviation_info["magnitude"],
                    persistence=deviation_info["persistence_ratio"],
                    metadata={
                        "baseline_mean": self.baseline_stats["mean"],
                        "baseline_std": self.baseline_stats["std"],
                        "adaptive_threshold": self.adaptive_threshold,
                        "residual": residual,
                        "deviation_info": deviation_info
                    }
                )
                
                self.current_deviation = drift_event
                self.deviation_events.append(drift_event)
                
                logger.warning(f"QoE deviation detected: {drift_type.value}, severity: {severity.value}")
                
                return True, drift_event
            
            else:
                # Update existing deviation
                self.current_deviation.magnitude = max(self.current_deviation.magnitude, deviation_info["magnitude"])
                self.current_deviation.persistence = deviation_info["persistence_ratio"]
                self.current_deviation.metadata["latest_residual"] = residual
                
                return False, None  # Continuation, not new detection
        
        else:
            # No persistent deviation
            if self.current_deviation is not None:
                # End of deviation period
                self.current_deviation.metadata["end_timestamp"] = current_time
                self.current_deviation = None
                logger.info("QoE deviation period ended")
            
            return False, None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get quantifier statistics"""
        stats = {
            "baseline_mean": self.baseline_stats["mean"],
            "baseline_std": self.baseline_stats["std"],
            "adaptive_threshold": self.adaptive_threshold,
            "total_deviations": len(self.deviation_events),
            "current_deviation_active": self.current_deviation is not None,
            "residual_history_size": len(self.residual_history)
        }
        
        if self.residual_history:
            recent_residuals = list(self.residual_history)[-10:]
            stats.update({
                "recent_mean_residual": np.mean(recent_residuals),
                "recent_std_residual": np.std(recent_residuals),
                "recent_max_residual": np.max(np.abs(recent_residuals))
            })
        
        # Deviation type distribution
        if self.deviation_events:
            type_counts = defaultdict(int)
            severity_counts = defaultdict(int)
            
            for event in self.deviation_events:
                type_counts[event.drift_type.value] += 1
                severity_counts[event.severity.value] += 1
            
            stats["deviation_types"] = dict(type_counts)
            stats["severity_distribution"] = dict(severity_counts)
        
        return stats

class AdvancedDriftDetectionEngine:
    """
    Main drift detection engine integrating HDDM-A, UADF, and QoE deviation quantifier
    Provides comprehensive drift detection with ensemble voting
    """
    
    def __init__(self, config: DriftDetectionConfig, feature_dim: int):
        self.config = config
        self.feature_dim = feature_dim
        
        # Initialize components
        self.hddm_detector = HoeffdingDriftDetectionMethod(config)
        self.uadf_filter = UncertaintyAwareDynamicFilter(config)
        self.qoe_model = QoEPredictionModel(config, feature_dim)
        self.deviation_quantifier = QoEDeviationQuantifier(config)
        
        # Ensemble state
        self.ensemble_history = deque(maxlen=100)
        self.drift_events = []
        
        # Performance tracking
        self.detection_stats = {
            "total_samples": 0,
            "hddm_detections": 0,
            "uadf_detections": 0,
            "deviation_detections": 0,
            "ensemble_detections": 0,
            "false_positives": 0,
            "true_positives": 0
        }
        
        logger.info("Advanced drift detection engine initialized")
    
    def fit_qoe_model(self, features: np.ndarray, qoe_targets: np.ndarray) -> Dict[str, Any]:
        """Fit the QoE prediction model"""
        return self.qoe_model.fit(features, qoe_targets)
    
    def _ensemble_voting(self, hddm_result: bool, uadf_result: bool, 
                        deviation_result: bool, confidences: Dict[str, float]) -> Tuple[bool, float]:
        """Perform ensemble voting for final drift decision"""
        if not self.config.enable_ensemble:
            # Simple OR logic if ensemble disabled
            return hddm_result or uadf_result or deviation_result, 1.0
        
        # Weighted voting based on confidence scores
        votes = []
        weights = []
        
        if hddm_result:
            votes.append(1)
            weights.append(confidences.get("hddm", 0.5))
        else:
            votes.append(0)
            weights.append(1 - confidences.get("hddm", 0.5))
        
        if uadf_result:
            votes.append(1)
            weights.append(confidences.get("uadf", 0.5))
        else:
            votes.append(0)
            weights.append(1 - confidences.get("uadf", 0.5))
        
        if deviation_result:
            votes.append(1)
            weights.append(confidences.get("deviation", 0.5))
        else:
            votes.append(0)
            weights.append(1 - confidences.get("deviation", 0.5))
        
        # Compute weighted average
        weighted_vote = np.average(votes, weights=weights)
        ensemble_decision = weighted_vote >= self.config.ensemble_voting_threshold
        
        return ensemble_decision, weighted_vote
    
    def detect_drift(self, features: np.ndarray, actual_qoe: float, 
                    ground_truth_drift: Optional[bool] = None) -> Tuple[bool, Optional[DriftEvent], Dict[str, Any]]:
        """
        Perform comprehensive drift detection
        
        Args:
            features: Current feature vector
            actual_qoe: Actual QoE value
            ground_truth_drift: Optional ground truth for evaluation
            
        Returns:
            (drift_detected, drift_event, detection_info)
        """
        self.detection_stats["total_samples"] += 1
        current_time = time.time()
        
        # Get QoE prediction and confidence
        if self.qoe_model.is_fitted:
            qoe_pred, confidence = self.qoe_model.predict(features.reshape(1, -1), return_confidence=True)
            qoe_prediction = qoe_pred[0] if len(qoe_pred) > 0 else actual_qoe
            qoe_confidence = confidence[0] if len(confidence) > 0 else 0.5
            prediction_error = abs(qoe_prediction - actual_qoe)
        else:
            qoe_prediction = actual_qoe  # Fallback
            qoe_confidence = 0.5
            prediction_error = 0.0
        
        # HDDM-A Detection
        hddm_drift, hddm_warning, hddm_info = self.hddm_detector.update(prediction_error)
        if hddm_drift:
            self.detection_stats["hddm_detections"] += 1
        
        # UADF Detection
        uadf_drift, adapted_threshold, uadf_info = self.uadf_filter.update(
            qoe_prediction, actual_qoe, qoe_confidence, features
        )
        if uadf_drift:
            self.detection_stats["uadf_detections"] += 1
        
        # QoE Deviation Detection
        deviation_detected, deviation_event = self.deviation_quantifier.update(
            qoe_prediction, actual_qoe, features
        )
        if deviation_detected:
            self.detection_stats["deviation_detections"] += 1
        
        # Ensemble Decision
        confidences = {
            "hddm": hddm_info.get("test_statistic", 0.0),
            "uadf": 1.0 - uadf_info.get("uncertainty", 0.5),
            "deviation": deviation_event.confidence if deviation_event else 0.0
        }
        
        ensemble_decision, ensemble_confidence = self._ensemble_voting(
            hddm_drift, uadf_drift, deviation_detected, confidences
        )
        
        if ensemble_decision:
            self.detection_stats["ensemble_detections"] += 1
        
        # Create comprehensive drift event if detected
        final_drift_event = None
        if ensemble_decision:
            # Determine dominant detection method
            detection_sources = []
            if hddm_drift:
                detection_sources.append("hddm")
            if uadf_drift:
                detection_sources.append("uadf")
            if deviation_detected:
                detection_sources.append("deviation")
            
            # Use deviation event as base if available, otherwise create new
            if deviation_event:
                final_drift_event = deviation_event
                final_drift_event.metadata["detection_sources"] = detection_sources
                final_drift_event.metadata["ensemble_confidence"] = ensemble_confidence
            else:
                # Create new drift event
                final_drift_event = DriftEvent(
                    timestamp=current_time,
                    drift_type=DriftType.GRADUAL,  # Default type
                    severity=DriftSeverity.MEDIUM,  # Default severity
                    confidence=ensemble_confidence,
                    affected_features=["qoe"],
                    trigger_source="ensemble",
                    magnitude=prediction_error,
                    persistence=1.0,
                    metadata={
                        "detection_sources": detection_sources,
                        "hddm_info": hddm_info,
                        "uadf_info": uadf_info,
                        "qoe_prediction": qoe_prediction,
                        "actual_qoe": actual_qoe,
                        "prediction_error": prediction_error
                    }
                )
            
            self.drift_events.append(final_drift_event)
        
        # Update performance tracking
        if ground_truth_drift is not None:
            if ensemble_decision and ground_truth_drift:
                self.detection_stats["true_positives"] += 1
            elif ensemble_decision and not ground_truth_drift:
                self.detection_stats["false_positives"] += 1
        
        # Store ensemble history
        ensemble_record = {
            "timestamp": current_time,
            "hddm_drift": hddm_drift,
            "uadf_drift": uadf_drift,
            "deviation_detected": deviation_detected,
            "ensemble_decision": ensemble_decision,
            "ensemble_confidence": ensemble_confidence,
            "prediction_error": prediction_error
        }
        self.ensemble_history.append(ensemble_record)
        
        # Compile detection info
        detection_info = {
            "qoe_prediction": qoe_prediction,
            "qoe_confidence": qoe_confidence,
            "prediction_error": prediction_error,
            "hddm": {"detected": hddm_drift, "warning": hddm_warning, "info": hddm_info},
            "uadf": {"detected": uadf_drift, "threshold": adapted_threshold, "info": uadf_info},
            "deviation": {"detected": deviation_detected, "event": deviation_event},
            "ensemble": {"decision": ensemble_decision, "confidence": ensemble_confidence},
            "detection_sources": detection_sources if ensemble_decision else []
        }
        
        return ensemble_decision, final_drift_event, detection_info
    
    def reset_detectors(self):
        """Reset all drift detectors"""
        self.hddm_detector.reset()
        # UADF and deviation quantifier maintain some history for adaptation
        logger.info("Drift detectors reset")
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components"""
        stats = {
            "detection_stats": self.detection_stats.copy(),
            "hddm_stats": self.hddm_detector.get_statistics(),
            "uadf_stats": self.uadf_filter.get_statistics(),
            "qoe_model_stats": self.qoe_model.get_statistics(),
            "deviation_stats": self.deviation_quantifier.get_statistics(),
            "total_drift_events": len(self.drift_events),
            "ensemble_history_size": len(self.ensemble_history)
        }
        
        # Compute performance metrics if available
        if self.detection_stats["true_positives"] + self.detection_stats["false_positives"] > 0:
            precision = self.detection_stats["true_positives"] / (
                self.detection_stats["true_positives"] + self.detection_stats["false_positives"]
            )
            stats["precision"] = precision
        
        # Recent ensemble decisions
        if self.ensemble_history:
            recent_decisions = [r["ensemble_decision"] for r in list(self.ensemble_history)[-20:]]
            stats["recent_detection_rate"] = sum(recent_decisions) / len(recent_decisions)
        
        return stats
    
    def save_model(self, filepath: str):
        """Save the QoE prediction model and scalers"""
        if self.qoe_model.is_fitted:
            # Save Keras model
            model_path = filepath.replace('.joblib', '_qoe_model.h5')
            self.qoe_model.model.save(model_path)
            
            # Save scaler and other components
            save_data = {
                "scaler": self.qoe_model.scaler,
                "config": self.config,
                "detection_stats": self.detection_stats,
                "is_fitted": self.qoe_model.is_fitted
            }
            joblib.dump(save_data, filepath)
            
            logger.info(f"Drift detection engine saved to {filepath}")
        else:
            logger.warning("QoE model not fitted, cannot save")
    
    def load_model(self, filepath: str):
        """Load the QoE prediction model and scalers"""
        if os.path.exists(filepath):
            # Load components
            save_data = joblib.load(filepath)
            self.qoe_model.scaler = save_data["scaler"]
            self.detection_stats = save_data.get("detection_stats", self.detection_stats)
            
            # Load Keras model
            model_path = filepath.replace('.joblib', '_qoe_model.h5')
            if os.path.exists(model_path):
                self.qoe_model.model = tf.keras.models.load_model(model_path)
                self.qoe_model.is_fitted = True
                logger.info(f"Drift detection engine loaded from {filepath}")
            else:
                logger.warning(f"QoE model file not found: {model_path}")
        else:
            logger.warning(f"Save file not found: {filepath}")

# Example usage and testing
if __name__ == "__main__":
    # Create configuration
    config = DriftDetectionConfig()
    
    # Initialize drift detection engine
    feature_dim = 24  # From multi-modal data architecture
    drift_engine = AdvancedDriftDetectionEngine(config, feature_dim)
    
    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Synthetic features (24-dimensional)
    features = np.random.randn(n_samples, feature_dim)
    
    # Synthetic QoE with concept drift
    qoe_values = []
    for i in range(n_samples):
        if i < 300:
            # Stable period
            qoe = 4.0 + 0.5 * np.sin(i * 0.1) + np.random.normal(0, 0.2)
        elif i < 600:
            # Gradual drift
            drift_factor = (i - 300) / 300.0
            qoe = 4.0 - drift_factor * 1.5 + 0.5 * np.sin(i * 0.1) + np.random.normal(0, 0.3)
        else:
            # Abrupt change
            qoe = 2.0 + 0.3 * np.sin(i * 0.15) + np.random.normal(0, 0.4)
        
        qoe_values.append(np.clip(qoe, 1.0, 5.0))
    
    qoe_values = np.array(qoe_values)
    
    # Train QoE model on first 200 samples
    print("Training QoE prediction model...")
    training_features = features[:200]
    training_qoe = qoe_values[:200]
    
    history = drift_engine.fit_qoe_model(training_features, training_qoe)
    print(f"Model training completed. Final loss: {history['loss'][-1]:.4f}")
    
    # Test drift detection on remaining samples
    print("\nTesting drift detection...")
    drift_detections = []
    detection_info_list = []
    
    for i in range(200, n_samples):
        # Ground truth: drift occurs at samples 300 and 600
        ground_truth = i >= 300
        
        drift_detected, drift_event, detection_info = drift_engine.detect_drift(
            features[i], qoe_values[i], ground_truth
        )
        
        drift_detections.append(drift_detected)
        detection_info_list.append(detection_info)
        
        if drift_detected and drift_event:
            print(f"Sample {i}: Drift detected - Type: {drift_event.drift_type.value}, "
                  f"Severity: {drift_event.severity.value}, Confidence: {drift_event.confidence:.3f}")
    
    # Get comprehensive statistics
    stats = drift_engine.get_comprehensive_statistics()
    print(f"\nDrift Detection Statistics:")
    print(f"Total samples processed: {stats['detection_stats']['total_samples']}")
    print(f"HDDM detections: {stats['detection_stats']['hddm_detections']}")
    print(f"UADF detections: {stats['detection_stats']['uadf_detections']}")
    print(f"Deviation detections: {stats['detection_stats']['deviation_detections']}")
    print(f"Ensemble detections: {stats['detection_stats']['ensemble_detections']}")
    print(f"Total drift events: {stats['total_drift_events']}")
    
    if "precision" in stats:
        print(f"Precision: {stats['precision']:.3f}")
    
    if "recent_detection_rate" in stats:
        print(f"Recent detection rate: {stats['recent_detection_rate']:.3f}")
    
    # Save the trained model
    save_path = "/tmp/drift_detection_engine.joblib"
    drift_engine.save_model(save_path)
    print(f"\nModel saved to: {save_path}")
    
    print("\nAdvanced drift detection engine test completed successfully!")

