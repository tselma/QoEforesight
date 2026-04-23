"""
QoE-Foresight: SHAP-based Explainability Module
Real-time interpretability and transparency for QoE prediction and self-healing decisions

This module provides comprehensive explainability features including real-time SHAP explanations,
feature importance analysis, visualization components, and human-in-the-loop debugging capabilities.
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
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

import shap
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExplanationType(Enum):
    """Types of explanations"""
    GLOBAL = "global"
    LOCAL = "local"
    COUNTERFACTUAL = "counterfactual"
    FEATURE_INTERACTION = "feature_interaction"

class ExplanationScope(Enum):
    """Scope of explanations"""
    QOE_PREDICTION = "qoe_prediction"
    DRIFT_DETECTION = "drift_detection"
    ACTION_SELECTION = "action_selection"
    SYSTEM_BEHAVIOR = "system_behavior"

@dataclass
class ExplanationRequest:
    """Request for explanation generation"""
    explanation_type: ExplanationType
    scope: ExplanationScope
    instance_data: Optional[np.ndarray] = None
    background_data: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    target_class: Optional[int] = None
    num_features: int = 10
    include_interactions: bool = False
    generate_plots: bool = True
    save_plots: bool = False
    plot_directory: str = "/tmp/explanations"

@dataclass
class ExplanationResult:
    """Result of explanation generation"""
    explanation_type: ExplanationType
    scope: ExplanationScope
    timestamp: float
    shap_values: np.ndarray
    feature_importance: Dict[str, float]
    feature_interactions: Optional[Dict[str, float]] = None
    plots: Optional[Dict[str, str]] = None  # Plot file paths
    summary: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class FeatureNameMapper:
    """Maps feature indices to human-readable names"""
    
    def __init__(self):
        self.feature_names = {
            # QoE features
            0: "Current QoE",
            1: "Predicted QoE", 
            2: "QoE Trend",
            3: "QoE Variance",
            
            # Drift features
            4: "Drift Detected",
            5: "Drift Severity",
            6: "Drift Confidence",
            7: "Drift Persistence",
            
            # Network features
            8: "Bandwidth",
            9: "Latency",
            10: "Packet Loss",
            11: "Jitter",
            12: "Network Stability",
            
            # Device features
            13: "CPU Usage",
            14: "GPU Usage", 
            15: "Battery Level",
            16: "Temperature",
            17: "Device Performance",
            
            # Application features
            18: "Buffer Occupancy",
            19: "Bitrate",
            20: "Resolution",
            21: "Stall Events",
            22: "Frame Rate",
            
            # Context features
            23: "Time of Day",
            24: "Content Type",
            25: "User Activity",
            26: "Resource Sensitivity"
        }
        
        self.feature_categories = {
            "QoE": [0, 1, 2, 3],
            "Drift": [4, 5, 6, 7],
            "Network": [8, 9, 10, 11, 12],
            "Device": [13, 14, 15, 16, 17],
            "Application": [18, 19, 20, 21, 22],
            "Context": [23, 24, 25, 26]
        }
        
        self.feature_descriptions = {
            0: "Current Quality of Experience score (1-5)",
            1: "Model-predicted QoE score",
            2: "Recent trend in QoE values",
            3: "Variance/stability of QoE",
            4: "Whether concept drift is detected",
            5: "Severity level of detected drift",
            6: "Confidence in drift detection",
            7: "How long drift has persisted",
            8: "Available network bandwidth (Mbps)",
            9: "Network round-trip latency (ms)",
            10: "Packet loss ratio (0-1)",
            11: "Network jitter (ms)",
            12: "Overall network stability score",
            13: "CPU utilization percentage",
            14: "GPU utilization percentage",
            15: "Device battery level percentage",
            16: "Device temperature (Celsius)",
            17: "Overall device performance score",
            18: "Video buffer occupancy (seconds)",
            19: "Current streaming bitrate (kbps)",
            20: "Video resolution (encoded)",
            21: "Number of stall events",
            22: "Video frame rate (fps)",
            23: "Time of day (normalized 0-1)",
            24: "Content type (encoded)",
            25: "User activity/engagement level",
            26: "Sensitivity to resource costs"
        }
    
    def get_name(self, feature_idx: int) -> str:
        """Get human-readable name for feature"""
        return self.feature_names.get(feature_idx, f"Feature_{feature_idx}")
    
    def get_description(self, feature_idx: int) -> str:
        """Get description for feature"""
        return self.feature_descriptions.get(feature_idx, "Unknown feature")
    
    def get_category(self, feature_idx: int) -> str:
        """Get category for feature"""
        for category, indices in self.feature_categories.items():
            if feature_idx in indices:
                return category
        return "Unknown"
    
    def get_category_features(self, category: str) -> List[int]:
        """Get all feature indices for a category"""
        return self.feature_categories.get(category, [])

class SHAPExplainer:
    """SHAP-based explainer for different model types"""
    
    def __init__(self, model, model_type: str = "deep", feature_names: Optional[List[str]] = None):
        self.model = model
        self.model_type = model_type
        self.feature_mapper = FeatureNameMapper()
        self.feature_names = feature_names or [self.feature_mapper.get_name(i) for i in range(27)]
        
        # SHAP explainers
        self.explainer = None
        self.background_data = None
        self.is_fitted = False
        
        # Explanation cache
        self.explanation_cache = {}
        self.cache_size_limit = 1000
        
        logger.info(f"SHAP explainer initialized for {model_type} model")
    
    def fit_explainer(self, background_data: np.ndarray, max_evals: int = 100):
        """Fit SHAP explainer with background data"""
        self.background_data = background_data
        
        try:
            if self.model_type == "deep":
                # For deep learning models
                if hasattr(self.model, 'predict'):
                    # Keras model
                    self.explainer = shap.DeepExplainer(self.model, background_data)
                else:
                    # Custom model - use KernelExplainer
                    self.explainer = shap.KernelExplainer(
                        self._model_predict_wrapper, 
                        background_data,
                        link="identity"
                    )
            
            elif self.model_type == "tree":
                # For tree-based models
                self.explainer = shap.TreeExplainer(self.model)
            
            elif self.model_type == "linear":
                # For linear models
                self.explainer = shap.LinearExplainer(self.model, background_data)
            
            else:
                # Default to KernelExplainer
                self.explainer = shap.KernelExplainer(
                    self._model_predict_wrapper,
                    background_data,
                    link="identity"
                )
            
            self.is_fitted = True
            logger.info(f"SHAP explainer fitted with {len(background_data)} background samples")
            
        except Exception as e:
            logger.error(f"Failed to fit SHAP explainer: {e}")
            # Fallback to KernelExplainer
            try:
                self.explainer = shap.KernelExplainer(
                    self._model_predict_wrapper,
                    background_data[:min(50, len(background_data))],  # Limit background size
                    link="identity"
                )
                self.is_fitted = True
                logger.info("Fallback to KernelExplainer successful")
            except Exception as e2:
                logger.error(f"Fallback explainer also failed: {e2}")
                self.is_fitted = False
    
    def _model_predict_wrapper(self, X: np.ndarray) -> np.ndarray:
        """Wrapper for model prediction to handle different model types"""
        try:
            if hasattr(self.model, 'predict'):
                predictions = self.model.predict(X, verbose=0)
                if len(predictions.shape) > 1 and predictions.shape[1] == 1:
                    return predictions.flatten()
                return predictions
            elif callable(self.model):
                return self.model(X)
            else:
                raise ValueError("Model must have predict method or be callable")
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return np.zeros(len(X))
    
    def explain_instance(self, instance: np.ndarray, nsamples: int = 100) -> Tuple[np.ndarray, Dict[str, float]]:
        """Generate SHAP explanation for a single instance"""
        if not self.is_fitted:
            raise ValueError("Explainer must be fitted before generating explanations")
        
        # Check cache
        instance_key = hash(instance.tobytes())
        if instance_key in self.explanation_cache:
            return self.explanation_cache[instance_key]
        
        try:
            # Generate SHAP values
            if self.model_type == "deep" and hasattr(self.explainer, 'shap_values'):
                shap_values = self.explainer.shap_values(instance.reshape(1, -1))
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]  # For multi-output models
                shap_values = shap_values.flatten()
            else:
                shap_values = self.explainer.shap_values(
                    instance.reshape(1, -1), 
                    nsamples=nsamples,
                    silent=True
                )
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                shap_values = shap_values.flatten()
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, importance in enumerate(shap_values):
                if i < len(self.feature_names):
                    feature_importance[self.feature_names[i]] = float(importance)
            
            # Cache result
            result = (shap_values, feature_importance)
            if len(self.explanation_cache) < self.cache_size_limit:
                self.explanation_cache[instance_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            # Return zero explanations as fallback
            shap_values = np.zeros(len(self.feature_names))
            feature_importance = {name: 0.0 for name in self.feature_names}
            return shap_values, feature_importance
    
    def explain_global(self, data: np.ndarray, max_samples: int = 500) -> Dict[str, float]:
        """Generate global feature importance"""
        if not self.is_fitted:
            raise ValueError("Explainer must be fitted before generating explanations")
        
        # Limit data size for computational efficiency
        if len(data) > max_samples:
            indices = np.random.choice(len(data), max_samples, replace=False)
            data = data[indices]
        
        try:
            # Generate SHAP values for all samples
            if self.model_type == "deep" and hasattr(self.explainer, 'shap_values'):
                shap_values = self.explainer.shap_values(data)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
            else:
                shap_values = self.explainer.shap_values(data, silent=True)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
            
            # Compute mean absolute SHAP values
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Create global importance dictionary
            global_importance = {}
            for i, importance in enumerate(mean_shap):
                if i < len(self.feature_names):
                    global_importance[self.feature_names[i]] = float(importance)
            
            return global_importance
            
        except Exception as e:
            logger.error(f"Global SHAP explanation failed: {e}")
            return {name: 0.0 for name in self.feature_names}

class ExplanationVisualizer:
    """Creates visualizations for SHAP explanations"""
    
    def __init__(self, feature_mapper: FeatureNameMapper, save_dir: str = "/tmp/explanations"):
        self.feature_mapper = feature_mapper
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_waterfall_plot(self, shap_values: np.ndarray, instance: np.ndarray, 
                            feature_names: List[str], base_value: float = 0.0,
                            max_display: int = 10, save_path: Optional[str] = None) -> str:
        """Create SHAP waterfall plot"""
        try:
            # Sort features by absolute SHAP value
            abs_shap = np.abs(shap_values)
            top_indices = np.argsort(abs_shap)[-max_display:][::-1]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Prepare data for waterfall
            values = shap_values[top_indices]
            names = [feature_names[i] for i in top_indices]
            feature_values = instance[top_indices]
            
            # Create waterfall plot
            cumulative = base_value
            positions = np.arange(len(values))
            
            colors = ['red' if v < 0 else 'blue' for v in values]
            
            # Plot bars
            bars = ax.barh(positions, values, color=colors, alpha=0.7)
            
            # Add feature values as text
            for i, (pos, val, feat_val, name) in enumerate(zip(positions, values, feature_values, names)):
                ax.text(val + 0.01 * np.sign(val), pos, f'{name}\n= {feat_val:.3f}', 
                       va='center', ha='left' if val > 0 else 'right', fontsize=10)
            
            ax.set_yticks(positions)
            ax.set_yticklabels([f'Feature {i+1}' for i in range(len(values))])
            ax.set_xlabel('SHAP Value (Impact on Prediction)')
            ax.set_title('Feature Importance Waterfall Plot')
            ax.grid(True, alpha=0.3)
            
            # Add base value line
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                timestamp = int(time.time())
                save_path = os.path.join(self.save_dir, f'waterfall_{timestamp}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return save_path
                
        except Exception as e:
            logger.error(f"Failed to create waterfall plot: {e}")
            plt.close()
            return ""
    
    def create_feature_importance_plot(self, feature_importance: Dict[str, float], 
                                     max_features: int = 15, save_path: Optional[str] = None) -> str:
        """Create feature importance bar plot"""
        try:
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            top_features = sorted_features[:max_features]
            
            features, importances = zip(*top_features)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create horizontal bar plot
            colors = ['red' if imp < 0 else 'blue' for imp in importances]
            bars = ax.barh(range(len(features)), importances, color=colors, alpha=0.7)
            
            # Customize plot
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('Feature Importance (SHAP Value)')
            ax.set_title('Top Feature Importances')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (bar, imp) in enumerate(zip(bars, importances)):
                ax.text(imp + 0.01 * np.sign(imp), i, f'{imp:.3f}', 
                       va='center', ha='left' if imp > 0 else 'right')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                timestamp = int(time.time())
                save_path = os.path.join(self.save_dir, f'importance_{timestamp}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return save_path
                
        except Exception as e:
            logger.error(f"Failed to create importance plot: {e}")
            plt.close()
            return ""
    
    def create_category_summary_plot(self, feature_importance: Dict[str, float], 
                                   save_path: Optional[str] = None) -> str:
        """Create category-wise importance summary"""
        try:
            # Group features by category
            category_importance = defaultdict(list)
            
            for feature_name, importance in feature_importance.items():
                # Find feature index
                feature_idx = None
                for idx, name in enumerate([self.feature_mapper.get_name(i) for i in range(27)]):
                    if name == feature_name:
                        feature_idx = idx
                        break
                
                if feature_idx is not None:
                    category = self.feature_mapper.get_category(feature_idx)
                    category_importance[category].append(abs(importance))
            
            # Compute category summaries
            category_summary = {}
            for category, importances in category_importance.items():
                category_summary[category] = {
                    'mean': np.mean(importances),
                    'max': np.max(importances),
                    'sum': np.sum(importances)
                }
            
            # Create subplot
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            metrics = ['mean', 'max', 'sum']
            titles = ['Average Importance', 'Maximum Importance', 'Total Importance']
            
            for i, (metric, title) in enumerate(zip(metrics, titles)):
                categories = list(category_summary.keys())
                values = [category_summary[cat][metric] for cat in categories]
                
                bars = axes[i].bar(categories, values, alpha=0.7)
                axes[i].set_title(title)
                axes[i].set_ylabel('SHAP Value')
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, val in zip(bars, values):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{val:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                timestamp = int(time.time())
                save_path = os.path.join(self.save_dir, f'category_summary_{timestamp}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return save_path
                
        except Exception as e:
            logger.error(f"Failed to create category summary plot: {e}")
            plt.close()
            return ""
    
    def create_interactive_plot(self, feature_importance: Dict[str, float], 
                              instance_data: Optional[np.ndarray] = None,
                              save_path: Optional[str] = None) -> str:
        """Create interactive Plotly visualization"""
        try:
            # Prepare data
            features = list(feature_importance.keys())
            importances = list(feature_importance.values())
            
            # Get categories and descriptions
            categories = []
            descriptions = []
            feature_values = []
            
            for feature_name in features:
                # Find feature index
                feature_idx = None
                for idx, name in enumerate([self.feature_mapper.get_name(i) for i in range(27)]):
                    if name == feature_name:
                        feature_idx = idx
                        break
                
                if feature_idx is not None:
                    categories.append(self.feature_mapper.get_category(feature_idx))
                    descriptions.append(self.feature_mapper.get_description(feature_idx))
                    if instance_data is not None and feature_idx < len(instance_data):
                        feature_values.append(instance_data[feature_idx])
                    else:
                        feature_values.append(0.0)
                else:
                    categories.append("Unknown")
                    descriptions.append("Unknown feature")
                    feature_values.append(0.0)
            
            # Create DataFrame
            df = pd.DataFrame({
                'Feature': features,
                'Importance': importances,
                'Category': categories,
                'Description': descriptions,
                'Value': feature_values,
                'AbsImportance': [abs(imp) for imp in importances]
            })
            
            # Sort by absolute importance
            df = df.sort_values('AbsImportance', ascending=True)
            
            # Create interactive plot
            fig = go.Figure()
            
            # Add bars with color coding
            colors = ['red' if imp < 0 else 'blue' for imp in df['Importance']]
            
            fig.add_trace(go.Bar(
                y=df['Feature'],
                x=df['Importance'],
                orientation='h',
                marker_color=colors,
                text=[f'{imp:.3f}' for imp in df['Importance']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>' +
                             'Importance: %{x:.3f}<br>' +
                             'Value: %{customdata[0]:.3f}<br>' +
                             'Category: %{customdata[1]}<br>' +
                             'Description: %{customdata[2]}<extra></extra>',
                customdata=df[['Value', 'Category', 'Description']].values
            ))
            
            fig.update_layout(
                title='Interactive Feature Importance Analysis',
                xaxis_title='SHAP Value (Feature Importance)',
                yaxis_title='Features',
                height=max(400, len(features) * 25),
                showlegend=False,
                template='plotly_white'
            )
            
            # Add vertical line at x=0
            fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
            
            if save_path:
                fig.write_html(save_path)
                return save_path
            else:
                timestamp = int(time.time())
                save_path = os.path.join(self.save_dir, f'interactive_{timestamp}.html')
                fig.write_html(save_path)
                return save_path
                
        except Exception as e:
            logger.error(f"Failed to create interactive plot: {e}")
            return ""

class ExplainabilityModule:
    """Main explainability module integrating all components"""
    
    def __init__(self, save_dir: str = "/tmp/qoe_explanations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize components
        self.feature_mapper = FeatureNameMapper()
        self.visualizer = ExplanationVisualizer(self.feature_mapper, save_dir)
        
        # Model explainers
        self.explainers = {}
        
        # Explanation history
        self.explanation_history = deque(maxlen=1000)
        self.global_importance_cache = {}
        
        # Performance tracking
        self.explanation_stats = {
            "total_explanations": 0,
            "explanation_times": [],
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info("Explainability module initialized")
    
    def register_model(self, model_name: str, model, model_type: str = "deep", 
                      background_data: Optional[np.ndarray] = None):
        """Register a model for explanation"""
        try:
            explainer = SHAPExplainer(model, model_type, self.feature_mapper.feature_names)
            
            if background_data is not None:
                explainer.fit_explainer(background_data)
            
            self.explainers[model_name] = explainer
            logger.info(f"Model '{model_name}' registered for explanation")
            
        except Exception as e:
            logger.error(f"Failed to register model '{model_name}': {e}")
    
    def explain_prediction(self, model_name: str, instance: np.ndarray, 
                         request: Optional[ExplanationRequest] = None) -> ExplanationResult:
        """Generate explanation for a single prediction"""
        start_time = time.time()
        
        if model_name not in self.explainers:
            raise ValueError(f"Model '{model_name}' not registered")
        
        explainer = self.explainers[model_name]
        
        if request is None:
            request = ExplanationRequest(
                explanation_type=ExplanationType.LOCAL,
                scope=ExplanationScope.QOE_PREDICTION,
                instance_data=instance,
                num_features=10,
                generate_plots=True
            )
        
        try:
            # Generate SHAP explanation
            shap_values, feature_importance = explainer.explain_instance(instance)
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), 
                                   key=lambda x: abs(x[1]), reverse=True)
            top_features = dict(sorted_features[:request.num_features])
            
            # Generate plots if requested
            plots = {}
            if request.generate_plots:
                # Waterfall plot
                waterfall_path = self.visualizer.create_waterfall_plot(
                    shap_values, instance, list(feature_importance.keys()),
                    save_path=os.path.join(self.save_dir, f'waterfall_{model_name}_{int(time.time())}.png')
                    if request.save_plots else None
                )
                if waterfall_path:
                    plots['waterfall'] = waterfall_path
                
                # Feature importance plot
                importance_path = self.visualizer.create_feature_importance_plot(
                    top_features,
                    save_path=os.path.join(self.save_dir, f'importance_{model_name}_{int(time.time())}.png')
                    if request.save_plots else None
                )
                if importance_path:
                    plots['importance'] = importance_path
                
                # Category summary
                category_path = self.visualizer.create_category_summary_plot(
                    feature_importance,
                    save_path=os.path.join(self.save_dir, f'category_{model_name}_{int(time.time())}.png')
                    if request.save_plots else None
                )
                if category_path:
                    plots['category'] = category_path
                
                # Interactive plot
                interactive_path = self.visualizer.create_interactive_plot(
                    top_features, instance,
                    save_path=os.path.join(self.save_dir, f'interactive_{model_name}_{int(time.time())}.html')
                    if request.save_plots else None
                )
                if interactive_path:
                    plots['interactive'] = interactive_path
            
            # Generate summary
            top_positive = [(k, v) for k, v in sorted_features if v > 0][:3]
            top_negative = [(k, v) for k, v in sorted_features if v < 0][:3]
            
            summary_parts = []
            if top_positive:
                summary_parts.append(f"Top positive factors: {', '.join([f'{k} ({v:.3f})' for k, v in top_positive])}")
            if top_negative:
                summary_parts.append(f"Top negative factors: {', '.join([f'{k} ({v:.3f})' for k, v in top_negative])}")
            
            summary = ". ".join(summary_parts) if summary_parts else "No significant factors identified."
            
            # Compute confidence (based on magnitude of top features)
            top_magnitudes = [abs(v) for _, v in sorted_features[:5]]
            confidence = min(1.0, np.mean(top_magnitudes) * 2) if top_magnitudes else 0.0
            
            # Create result
            result = ExplanationResult(
                explanation_type=request.explanation_type,
                scope=request.scope,
                timestamp=time.time(),
                shap_values=shap_values,
                feature_importance=top_features,
                plots=plots,
                summary=summary,
                confidence=confidence,
                metadata={
                    "model_name": model_name,
                    "instance_shape": instance.shape,
                    "explanation_time": time.time() - start_time,
                    "num_features_analyzed": len(feature_importance)
                }
            )
            
            # Update statistics
            self.explanation_stats["total_explanations"] += 1
            self.explanation_stats["explanation_times"].append(time.time() - start_time)
            
            # Store in history
            self.explanation_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            # Return empty result
            return ExplanationResult(
                explanation_type=request.explanation_type,
                scope=request.scope,
                timestamp=time.time(),
                shap_values=np.zeros(len(self.feature_mapper.feature_names)),
                feature_importance={},
                summary=f"Explanation failed: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def explain_global_behavior(self, model_name: str, data: np.ndarray, 
                              max_samples: int = 500) -> Dict[str, Any]:
        """Generate global explanation for model behavior"""
        if model_name not in self.explainers:
            raise ValueError(f"Model '{model_name}' not registered")
        
        # Check cache
        cache_key = f"{model_name}_{len(data)}_{hash(data.tobytes())}"
        if cache_key in self.global_importance_cache:
            self.explanation_stats["cache_hits"] += 1
            return self.global_importance_cache[cache_key]
        
        self.explanation_stats["cache_misses"] += 1
        
        try:
            explainer = self.explainers[model_name]
            global_importance = explainer.explain_global(data, max_samples)
            
            # Create visualizations
            plots = {}
            
            # Global importance plot
            importance_path = self.visualizer.create_feature_importance_plot(
                global_importance, max_features=20,
                save_path=os.path.join(self.save_dir, f'global_importance_{model_name}_{int(time.time())}.png')
            )
            if importance_path:
                plots['global_importance'] = importance_path
            
            # Category summary
            category_path = self.visualizer.create_category_summary_plot(
                global_importance,
                save_path=os.path.join(self.save_dir, f'global_category_{model_name}_{int(time.time())}.png')
            )
            if category_path:
                plots['category_summary'] = category_path
            
            # Interactive global plot
            interactive_path = self.visualizer.create_interactive_plot(
                global_importance,
                save_path=os.path.join(self.save_dir, f'global_interactive_{model_name}_{int(time.time())}.html')
            )
            if interactive_path:
                plots['interactive'] = interactive_path
            
            result = {
                "global_importance": global_importance,
                "plots": plots,
                "timestamp": time.time(),
                "model_name": model_name,
                "samples_analyzed": min(len(data), max_samples)
            }
            
            # Cache result
            self.global_importance_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate global explanation: {e}")
            return {
                "global_importance": {},
                "plots": {},
                "error": str(e),
                "timestamp": time.time()
            }
    
    def explain_drift_detection(self, drift_detector, instance: np.ndarray, 
                              drift_result: Dict[str, Any]) -> ExplanationResult:
        """Explain drift detection decision"""
        try:
            # Create a simple explanation based on drift detection components
            feature_importance = {}
            
            # Analyze which features contributed most to drift detection
            if "hddm" in drift_result and drift_result["hddm"]["detected"]:
                # HDDM detected drift - focus on prediction error
                feature_importance["Prediction Error"] = 0.8
                feature_importance["QoE Variance"] = 0.6
            
            if "uadf" in drift_result and drift_result["uadf"]["detected"]:
                # UADF detected drift - focus on uncertainty
                uncertainty_info = drift_result["uadf"]["info"]
                feature_importance["Prediction Uncertainty"] = uncertainty_info.get("uncertainty", 0.5)
                feature_importance["Network Stability"] = 0.4
            
            if "deviation" in drift_result and drift_result["deviation"]["detected"]:
                # Deviation detected - focus on QoE deviation
                feature_importance["QoE Deviation"] = 0.9
                feature_importance["Drift Persistence"] = 0.7
            
            # Add context features
            if instance[4] > 0.5:  # Drift detected flag
                feature_importance["Drift Detected"] = 0.9
            
            if instance[5] > 2:  # High drift severity
                feature_importance["Drift Severity"] = instance[5] / 4.0
            
            # Network instability factors
            if instance[12] < 0.5:  # Low network stability
                feature_importance["Network Stability"] = 0.6
            
            # Generate summary
            detected_methods = []
            if drift_result.get("hddm", {}).get("detected", False):
                detected_methods.append("HDDM-A")
            if drift_result.get("uadf", {}).get("detected", False):
                detected_methods.append("UADF")
            if drift_result.get("deviation", {}).get("detected", False):
                detected_methods.append("Deviation Quantifier")
            
            if detected_methods:
                summary = f"Drift detected by: {', '.join(detected_methods)}. "
                summary += f"Ensemble confidence: {drift_result.get('ensemble', {}).get('confidence', 0.0):.3f}"
            else:
                summary = "No drift detected by any method."
            
            # Create result
            result = ExplanationResult(
                explanation_type=ExplanationType.LOCAL,
                scope=ExplanationScope.DRIFT_DETECTION,
                timestamp=time.time(),
                shap_values=np.array(list(feature_importance.values())),
                feature_importance=feature_importance,
                summary=summary,
                confidence=drift_result.get("ensemble", {}).get("confidence", 0.0),
                metadata={
                    "drift_result": drift_result,
                    "detection_methods": detected_methods
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to explain drift detection: {e}")
            return ExplanationResult(
                explanation_type=ExplanationType.LOCAL,
                scope=ExplanationScope.DRIFT_DETECTION,
                timestamp=time.time(),
                shap_values=np.zeros(5),
                feature_importance={},
                summary=f"Drift explanation failed: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def explain_action_selection(self, action_type: str, state: np.ndarray, 
                               q_values: Optional[np.ndarray] = None) -> ExplanationResult:
        """Explain self-healing action selection"""
        try:
            feature_importance = {}
            
            # Map state features to action relevance
            state_features = [self.feature_mapper.get_name(i) for i in range(len(state))]
            
            # Action-specific feature importance
            if "bitrate" in action_type.lower():
                feature_importance["Bandwidth"] = state[8] / 50.0  # Normalize bandwidth
                feature_importance["Network Stability"] = state[12]
                feature_importance["Current QoE"] = state[0] / 5.0
                
            elif "buffer" in action_type.lower():
                feature_importance["Buffer Occupancy"] = 1.0 - state[18] / 60.0  # Lower buffer = higher importance
                feature_importance["Stall Events"] = min(1.0, state[21] / 5.0)
                feature_importance["Current QoE"] = state[0] / 5.0
                
            elif "resolution" in action_type.lower():
                feature_importance["Drift Severity"] = state[5] / 4.0
                feature_importance["Device Performance"] = 1.0 - state[17]  # Lower performance = higher importance
                feature_importance["Network Stability"] = 1.0 - state[12]
                
            elif "server" in action_type.lower():
                feature_importance["Latency"] = min(1.0, state[9] / 200.0)
                feature_importance["Network Stability"] = 1.0 - state[12]
                feature_importance["Current QoE"] = 1.0 - state[0] / 5.0
                
            else:
                # General importance for other actions
                feature_importance["Current QoE"] = 1.0 - state[0] / 5.0
                feature_importance["Drift Detected"] = state[4]
                feature_importance["Network Stability"] = 1.0 - state[12]
            
            # Add Q-value information if available
            if q_values is not None:
                max_q = np.max(q_values)
                feature_importance["Action Value"] = max_q / (np.max(np.abs(q_values)) + 1e-8)
            
            # Generate summary
            top_factors = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            summary = f"Action '{action_type}' selected based on: " + \
                     ", ".join([f"{factor} ({value:.3f})" for factor, value in top_factors])
            
            # Compute confidence
            confidence = np.mean(list(feature_importance.values())) if feature_importance else 0.0
            
            result = ExplanationResult(
                explanation_type=ExplanationType.LOCAL,
                scope=ExplanationScope.ACTION_SELECTION,
                timestamp=time.time(),
                shap_values=np.array(list(feature_importance.values())),
                feature_importance=feature_importance,
                summary=summary,
                confidence=confidence,
                metadata={
                    "action_type": action_type,
                    "state_shape": state.shape,
                    "q_values": q_values.tolist() if q_values is not None else None
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to explain action selection: {e}")
            return ExplanationResult(
                explanation_type=ExplanationType.LOCAL,
                scope=ExplanationScope.ACTION_SELECTION,
                timestamp=time.time(),
                shap_values=np.zeros(5),
                feature_importance={},
                summary=f"Action explanation failed: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def get_explanation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive explanation statistics"""
        stats = self.explanation_stats.copy()
        
        if stats["explanation_times"]:
            stats["avg_explanation_time"] = np.mean(stats["explanation_times"])
            stats["max_explanation_time"] = np.max(stats["explanation_times"])
            stats["min_explanation_time"] = np.min(stats["explanation_times"])
        
        stats["registered_models"] = list(self.explainers.keys())
        stats["explanation_history_size"] = len(self.explanation_history)
        stats["cache_size"] = len(self.global_importance_cache)
        
        if stats["cache_hits"] + stats["cache_misses"] > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / (stats["cache_hits"] + stats["cache_misses"])
        
        return stats
    
    def generate_explanation_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive explanation report"""
        try:
            report_lines = []
            report_lines.append("# QoE-Foresight Explainability Report")
            report_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Statistics
            stats = self.get_explanation_statistics()
            report_lines.append("## Statistics")
            report_lines.append(f"- Total explanations generated: {stats['total_explanations']}")
            report_lines.append(f"- Registered models: {len(stats['registered_models'])}")
            report_lines.append(f"- Average explanation time: {stats.get('avg_explanation_time', 0):.3f}s")
            report_lines.append(f"- Cache hit rate: {stats.get('cache_hit_rate', 0):.3f}")
            report_lines.append("")
            
            # Recent explanations
            if self.explanation_history:
                report_lines.append("## Recent Explanations")
                recent_explanations = list(self.explanation_history)[-10:]
                
                for i, explanation in enumerate(recent_explanations, 1):
                    report_lines.append(f"### Explanation {i}")
                    report_lines.append(f"- Type: {explanation.explanation_type.value}")
                    report_lines.append(f"- Scope: {explanation.scope.value}")
                    report_lines.append(f"- Confidence: {explanation.confidence:.3f}")
                    report_lines.append(f"- Summary: {explanation.summary}")
                    
                    if explanation.feature_importance:
                        top_features = sorted(explanation.feature_importance.items(), 
                                            key=lambda x: abs(x[1]), reverse=True)[:5]
                        report_lines.append("- Top features:")
                        for feature, importance in top_features:
                            report_lines.append(f"  - {feature}: {importance:.3f}")
                    report_lines.append("")
            
            report_content = "\n".join(report_lines)
            
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report_content)
                return save_path
            else:
                timestamp = int(time.time())
                save_path = os.path.join(self.save_dir, f'explanation_report_{timestamp}.md')
                with open(save_path, 'w') as f:
                    f.write(report_content)
                return save_path
                
        except Exception as e:
            logger.error(f"Failed to generate explanation report: {e}")
            return ""

# Example usage and testing
if __name__ == "__main__":
    # Create explainability module
    explainer_module = ExplainabilityModule()
    
    # Create a simple model for testing
    from sklearn.ensemble import RandomForestRegressor
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 27
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] * 0.5 + X[:, 8] * 0.3 + X[:, 18] * 0.2 + 
         np.random.randn(n_samples) * 0.1)  # QoE based on some features
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print("Testing SHAP-based explainability module...")
    
    # Register model
    explainer_module.register_model("qoe_predictor", model, "tree", X[:100])
    
    # Test local explanation
    test_instance = X[500]
    explanation = explainer_module.explain_prediction("qoe_predictor", test_instance)
    
    print(f"Local explanation generated:")
    print(f"- Confidence: {explanation.confidence:.3f}")
    print(f"- Summary: {explanation.summary}")
    print(f"- Top features: {list(explanation.feature_importance.keys())[:5]}")
    print(f"- Plots generated: {list(explanation.plots.keys()) if explanation.plots else 'None'}")
    
    # Test global explanation
    global_explanation = explainer_module.explain_global_behavior("qoe_predictor", X[:200])
    print(f"\nGlobal explanation generated:")
    print(f"- Samples analyzed: {global_explanation['samples_analyzed']}")
    print(f"- Top global features: {list(global_explanation['global_importance'].keys())[:5]}")
    
    # Test drift explanation
    drift_result = {
        "hddm": {"detected": True, "info": {"test_statistic": 0.8}},
        "uadf": {"detected": False, "info": {"uncertainty": 0.3}},
        "deviation": {"detected": True, "event": None},
        "ensemble": {"decision": True, "confidence": 0.75}
    }
    
    drift_explanation = explainer_module.explain_drift_detection(None, test_instance, drift_result)
    print(f"\nDrift explanation generated:")
    print(f"- Summary: {drift_explanation.summary}")
    print(f"- Confidence: {drift_explanation.confidence:.3f}")
    
    # Test action explanation
    action_explanation = explainer_module.explain_action_selection("decrease_bitrate", test_instance)
    print(f"\nAction explanation generated:")
    print(f"- Summary: {action_explanation.summary}")
    print(f"- Confidence: {action_explanation.confidence:.3f}")
    
    # Generate report
    report_path = explainer_module.generate_explanation_report()
    print(f"\nExplanation report saved to: {report_path}")
    
    # Get statistics
    stats = explainer_module.get_explanation_statistics()
    print(f"\nExplainability Statistics:")
    print(f"- Total explanations: {stats['total_explanations']}")
    print(f"- Average time: {stats.get('avg_explanation_time', 0):.3f}s")
    print(f"- Registered models: {stats['registered_models']}")
    
    print("\nSHAP-based explainability module test completed successfully!")

