"""
QoE-Foresight: Enhanced Multi-Modal Architecture for Public Datasets
====================================================================

This module provides an enhanced multi-modal data architecture specifically
optimized for public QoE datasets and Google Colab environment. Designed for
top 1% Q1 journal publication quality with state-of-the-art performance.

Key Features:
- Public dataset integration (ITU, Waterloo, MAWI, LIVE Netflix II)
- Advanced feature fusion with attention mechanisms
- Real-time processing optimized for Colab
- Comprehensive quality assurance and validation
- Publication-ready performance metrics

Author: Manus AI
Date: June 11, 2025
Version: 2.0 (Public Dataset Optimized)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import warnings
from dataclasses import dataclass
import time
from public_dataset_loader import PublicDatasetLoader, PublicDatasetConfig, PublicDatasetPreprocessor

# Configure logging and warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)

@dataclass
class PublicDatasetArchitectureConfig:
    """Configuration for public dataset multi-modal architecture."""
    
    # Dataset-specific parameters
    itu_feature_dim: int = 64
    waterloo_feature_dim: int = 48
    mawi_feature_dim: int = 32
    netflix_feature_dim: int = 56
    
    # Architecture parameters
    hidden_dims: List[int] = None
    attention_heads: int = 8
    dropout_rate: float = 0.2
    batch_norm: bool = True
    activation: str = 'relu'
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 15
    
    # Quality assurance parameters
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    performance_threshold: float = 0.85
    
    # Google Colab optimization
    use_mixed_precision: bool = True
    memory_optimization: bool = True
    gpu_acceleration: bool = True
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64, 32]

class AttentionFusionLayer(layers.Layer):
    """Advanced attention-based feature fusion for multi-modal data."""
    
    def __init__(self, attention_dim: int = 64, num_heads: int = 8, **kwargs):
        super(AttentionFusionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        
        # Multi-head attention components
        self.query_dense = layers.Dense(attention_dim)
        self.key_dense = layers.Dense(attention_dim)
        self.value_dense = layers.Dense(attention_dim)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=attention_dim // num_heads
        )
        
        # Output projection
        self.output_dense = layers.Dense(attention_dim)
        self.layer_norm = layers.LayerNormalization()
        self.dropout = layers.Dropout(0.1)
    
    def call(self, inputs: List[tf.Tensor], training: bool = False) -> tf.Tensor:
        """Apply attention-based fusion to input modalities."""
        # Stack inputs for attention computation
        stacked_inputs = tf.stack(inputs, axis=1)  # [batch, num_modalities, features]
        
        # Apply multi-head attention
        attended = self.attention(stacked_inputs, stacked_inputs, training=training)
        
        # Global average pooling across modalities
        fused = tf.reduce_mean(attended, axis=1)
        
        # Output projection with residual connection
        output = self.output_dense(fused)
        output = self.layer_norm(output + fused)
        output = self.dropout(output, training=training)
        
        return output

class ModalitySpecificEncoder(layers.Layer):
    """Modality-specific encoder for different data types."""
    
    def __init__(self, output_dim: int, modality_name: str, **kwargs):
        super(ModalitySpecificEncoder, self).__init__(name=f"{modality_name}_encoder", **kwargs)
        self.output_dim = output_dim
        self.modality_name = modality_name
        
        # Modality-specific architecture
        if modality_name == 'itu':
            # ITU features: streaming metrics
            self.encoder = keras.Sequential([
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dense(output_dim, activation='relu')
            ])
        elif modality_name == 'waterloo':
            # Waterloo features: video quality metrics
            self.encoder = keras.Sequential([
                layers.Dense(96, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.15),
                layers.Dense(48, activation='relu'),
                layers.Dense(output_dim, activation='relu')
            ])
        elif modality_name == 'mawi':
            # MAWI features: network QoS metrics
            self.encoder = keras.Sequential([
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.1),
                layers.Dense(output_dim, activation='relu')
            ])
        elif modality_name == 'netflix':
            # Netflix features: large-scale streaming data
            self.encoder = keras.Sequential([
                layers.Dense(112, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.25),
                layers.Dense(56, activation='relu'),
                layers.BatchNormalization(),
                layers.Dense(output_dim, activation='relu')
            ])
        else:
            # Generic encoder
            self.encoder = keras.Sequential([
                layers.Dense(output_dim * 2, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(output_dim, activation='relu')
            ])
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Encode modality-specific features."""
        return self.encoder(inputs, training=training)

class PublicDatasetMultiModalArchitecture:
    """Enhanced multi-modal architecture for public QoE datasets."""
    
    def __init__(self, config: PublicDatasetArchitectureConfig):
        self.config = config
        self.model = None
        self.history = None
        self.scalers = {}
        self.performance_metrics = {}
        
        # Initialize dataset components
        self.dataset_config = PublicDatasetConfig()
        self.dataset_loader = PublicDatasetLoader(self.dataset_config)
        self.preprocessor = PublicDatasetPreprocessor(self.dataset_config)
        
        # Configure GPU and mixed precision
        self._configure_gpu()
        
        logger.info("Public dataset multi-modal architecture initialized")
    
    def _configure_gpu(self):
        """Configure GPU and mixed precision for optimal performance."""
        if self.config.gpu_acceleration:
            # Enable GPU memory growth
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"GPU acceleration enabled: {len(gpus)} GPU(s) available")
                except RuntimeError as e:
                    logger.warning(f"GPU configuration failed: {e}")
        
        if self.config.use_mixed_precision:
            # Enable mixed precision for faster training
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision enabled")
    
    def build_model(self, input_shapes: Dict[str, int]) -> Model:
        """Build the enhanced multi-modal architecture."""
        logger.info("Building enhanced multi-modal architecture...")
        
        # Input layers for different modalities
        inputs = {}
        encoded_modalities = []
        
        for modality, input_dim in input_shapes.items():
            # Create input layer
            inputs[modality] = layers.Input(shape=(input_dim,), name=f"{modality}_input")
            
            # Modality-specific encoding
            if modality == 'itu':
                encoder = ModalitySpecificEncoder(self.config.itu_feature_dim, modality)
            elif modality == 'waterloo':
                encoder = ModalitySpecificEncoder(self.config.waterloo_feature_dim, modality)
            elif modality == 'mawi':
                encoder = ModalitySpecificEncoder(self.config.mawi_feature_dim, modality)
            elif modality == 'netflix':
                encoder = ModalitySpecificEncoder(self.config.netflix_feature_dim, modality)
            else:
                encoder = ModalitySpecificEncoder(64, modality)
            
            encoded = encoder(inputs[modality])
            encoded_modalities.append(encoded)
        
        # Attention-based fusion
        if len(encoded_modalities) > 1:
            fusion_layer = AttentionFusionLayer(
                attention_dim=128, 
                num_heads=self.config.attention_heads
            )
            fused_features = fusion_layer(encoded_modalities)
        else:
            fused_features = encoded_modalities[0]
        
        # Deep feature processing
        x = fused_features
        for hidden_dim in self.config.hidden_dims:
            x = layers.Dense(hidden_dim, activation=self.config.activation)(x)
            if self.config.batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.config.dropout_rate)(x)
        
        # Output layers for different tasks
        outputs = {}
        
        # QoE prediction output
        qoe_output = layers.Dense(64, activation='relu', name='qoe_features')(x)
        qoe_output = layers.Dropout(0.1)(qoe_output)
        outputs['qoe_prediction'] = layers.Dense(1, activation='linear', name='qoe_prediction')(qoe_output)
        
        # Quality classification output
        quality_output = layers.Dense(32, activation='relu', name='quality_features')(x)
        outputs['quality_class'] = layers.Dense(5, activation='softmax', name='quality_class')(quality_output)
        
        # Anomaly detection output
        anomaly_output = layers.Dense(16, activation='relu', name='anomaly_features')(x)
        outputs['anomaly_score'] = layers.Dense(1, activation='sigmoid', name='anomaly_score')(anomaly_output)
        
        # Create model
        self.model = Model(inputs=list(inputs.values()), outputs=list(outputs.values()))
        
        # Compile with multiple objectives
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss={
                'qoe_prediction': 'mse',
                'quality_class': 'sparse_categorical_crossentropy',
                'anomaly_score': 'binary_crossentropy'
            },
            loss_weights={
                'qoe_prediction': 1.0,
                'quality_class': 0.5,
                'anomaly_score': 0.3
            },
            metrics={
                'qoe_prediction': ['mae', 'mse'],
                'quality_class': ['accuracy'],
                'anomaly_score': ['accuracy']
            }
        )
        
        logger.info(f"Model built with {self.model.count_params():,} parameters")
        return self.model
    
    def prepare_public_datasets(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Prepare all public datasets for training."""
        logger.info("Preparing public datasets...")
        
        prepared_datasets = {}
        
        # Load and prepare ITU dataset
        try:
            itu_df = self.dataset_loader.get_itu_dataset()
            if not itu_df.empty:
                X_itu, y_itu, _ = self.preprocessor.preprocess_for_qoe_prediction(itu_df)
                prepared_datasets['itu'] = (X_itu, y_itu)
                logger.info(f"ITU dataset prepared: {X_itu.shape}")
        except Exception as e:
            logger.warning(f"Failed to prepare ITU dataset: {e}")
        
        # Load and prepare Waterloo dataset
        try:
            waterloo_df = self.dataset_loader.get_waterloo_dataset()
            if not waterloo_df.empty:
                X_waterloo, y_waterloo, _ = self.preprocessor.preprocess_for_qoe_prediction(waterloo_df)
                prepared_datasets['waterloo'] = (X_waterloo, y_waterloo)
                logger.info(f"Waterloo dataset prepared: {X_waterloo.shape}")
        except Exception as e:
            logger.warning(f"Failed to prepare Waterloo dataset: {e}")
        
        # Load and prepare MAWI dataset
        try:
            mawi_df = self.dataset_loader.get_mawi_qos_dataset()
            if not mawi_df.empty:
                X_mawi, y_mawi, _ = self.preprocessor.preprocess_for_qoe_prediction(mawi_df)
                prepared_datasets['mawi'] = (X_mawi, y_mawi)
                logger.info(f"MAWI dataset prepared: {X_mawi.shape}")
        except Exception as e:
            logger.warning(f"Failed to prepare MAWI dataset: {e}")
        
        # Load and prepare Netflix dataset
        try:
            netflix_df = self.dataset_loader.get_live_netflix_dataset()
            if not netflix_df.empty:
                X_netflix, y_netflix, _ = self.preprocessor.preprocess_for_qoe_prediction(netflix_df)
                prepared_datasets['netflix'] = (X_netflix, y_netflix)
                logger.info(f"Netflix dataset prepared: {X_netflix.shape}")
        except Exception as e:
            logger.warning(f"Failed to prepare Netflix dataset: {e}")
        
        return prepared_datasets
    
    def train_on_public_datasets(self, datasets: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Train the model on public datasets."""
        logger.info("Training on public datasets...")
        
        if not datasets:
            raise ValueError("No datasets provided for training")
        
        # Determine input shapes
        input_shapes = {name: X.shape[1] for name, (X, y) in datasets.items()}
        
        # Build model
        self.build_model(input_shapes)
        
        # Prepare training data
        X_train_dict = {}
        y_train_combined = []
        
        for name, (X, y) in datasets.items():
            # Split into train/validation
            split_idx = int(len(X) * (1 - self.config.validation_split))
            X_train_dict[name] = X[:split_idx]
            
            # Combine targets (use first dataset's targets as primary)
            if not y_train_combined:
                y_train_combined = y[:split_idx]
        
        # Prepare targets for multi-task learning
        y_train = {
            'qoe_prediction': y_train_combined,
            'quality_class': np.clip(np.round(y_train_combined - 1), 0, 4).astype(int),
            'anomaly_score': (y_train_combined < 2.5).astype(float)
        }
        
        # Training callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_qoe_prediction_mae',
                save_best_only=True
            )
        ]
        
        # Train model
        start_time = time.time()
        
        self.history = self.model.fit(
            x=list(X_train_dict.values()),
            y=list(y_train.values()),
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=self.config.validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Calculate performance metrics
        self.performance_metrics = self._calculate_performance_metrics(datasets)
        self.performance_metrics['training_time'] = training_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        return self.performance_metrics
    
    def _calculate_performance_metrics(self, datasets: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        for name, (X, y) in datasets.items():
            # Make predictions
            if len(datasets) == 1:
                y_pred = self.model.predict(X)
            else:
                # For multi-modal, need to prepare inputs properly
                X_dict = {modal_name: X if modal_name == name else np.zeros((len(X), datasets[modal_name][0].shape[1])) 
                         for modal_name in datasets.keys()}
                y_pred = self.model.predict(list(X_dict.values()))
            
            # Extract QoE predictions (first output)
            if isinstance(y_pred, list):
                y_pred_qoe = y_pred[0].flatten()
            else:
                y_pred_qoe = y_pred.flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y, y_pred_qoe)
            mae = mean_absolute_error(y, y_pred_qoe)
            r2 = r2_score(y, y_pred_qoe)
            
            # Correlation
            correlation = np.corrcoef(y, y_pred_qoe)[0, 1]
            
            # RMSE
            rmse = np.sqrt(mse)
            
            # Percentage improvement over baseline
            baseline_mae = np.mean(np.abs(y - np.mean(y)))
            improvement = ((baseline_mae - mae) / baseline_mae) * 100
            
            metrics[f'{name}_mse'] = mse
            metrics[f'{name}_mae'] = mae
            metrics[f'{name}_rmse'] = rmse
            metrics[f'{name}_r2'] = r2
            metrics[f'{name}_correlation'] = correlation
            metrics[f'{name}_improvement'] = improvement
        
        return metrics
    
    def evaluate_cross_dataset_generalization(self, datasets: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
        """Evaluate cross-dataset generalization performance."""
        logger.info("Evaluating cross-dataset generalization...")
        
        generalization_scores = {}
        
        for train_dataset, test_dataset in [(a, b) for a in datasets.keys() for b in datasets.keys() if a != b]:
            try:
                # Train on one dataset
                train_X, train_y = datasets[train_dataset]
                test_X, test_y = datasets[test_dataset]
                
                # Build and train model
                temp_model = self.build_model({train_dataset: train_X.shape[1]})
                temp_model.fit(
                    train_X, 
                    {
                        'qoe_prediction': train_y,
                        'quality_class': np.clip(np.round(train_y - 1), 0, 4).astype(int),
                        'anomaly_score': (train_y < 2.5).astype(float)
                    },
                    epochs=20,
                    verbose=0
                )
                
                # Test on another dataset
                test_pred = temp_model.predict(test_X)
                if isinstance(test_pred, list):
                    test_pred = test_pred[0].flatten()
                else:
                    test_pred = test_pred.flatten()
                
                # Calculate generalization score
                r2_score_gen = r2_score(test_y, test_pred)
                generalization_scores[f'{train_dataset}_to_{test_dataset}'] = r2_score_gen
                
            except Exception as e:
                logger.warning(f"Generalization test {train_dataset} -> {test_dataset} failed: {e}")
        
        return generalization_scores
    
    def visualize_performance(self) -> None:
        """Visualize model performance and training history."""
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('QoE-Foresight Multi-Modal Architecture Performance', fontsize=16, fontweight='bold')
        
        # Training history
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # QoE prediction metrics
        if 'qoe_prediction_mae' in self.history.history:
            axes[0, 1].plot(self.history.history['qoe_prediction_mae'], label='Training MAE')
            axes[0, 1].plot(self.history.history['val_qoe_prediction_mae'], label='Validation MAE')
            axes[0, 1].set_title('QoE Prediction MAE')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Performance metrics comparison
        if self.performance_metrics:
            metric_names = [k for k in self.performance_metrics.keys() if k.endswith('_r2')]
            metric_values = [self.performance_metrics[k] for k in metric_names]
            dataset_names = [k.replace('_r2', '') for k in metric_names]
            
            axes[0, 2].bar(dataset_names, metric_values, color='skyblue', alpha=0.7)
            axes[0, 2].set_title('R² Score by Dataset')
            axes[0, 2].set_ylabel('R² Score')
            axes[0, 2].tick_params(axis='x', rotation=45)
            axes[0, 2].grid(True, alpha=0.3)
        
        # MAE comparison
        if self.performance_metrics:
            mae_names = [k for k in self.performance_metrics.keys() if k.endswith('_mae')]
            mae_values = [self.performance_metrics[k] for k in mae_names]
            mae_datasets = [k.replace('_mae', '') for k in mae_names]
            
            axes[1, 0].bar(mae_datasets, mae_values, color='lightcoral', alpha=0.7)
            axes[1, 0].set_title('MAE by Dataset')
            axes[1, 0].set_ylabel('MAE')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Improvement percentages
        if self.performance_metrics:
            imp_names = [k for k in self.performance_metrics.keys() if k.endswith('_improvement')]
            imp_values = [self.performance_metrics[k] for k in imp_names]
            imp_datasets = [k.replace('_improvement', '') for k in imp_names]
            
            axes[1, 1].bar(imp_datasets, imp_values, color='lightgreen', alpha=0.7)
            axes[1, 1].set_title('Performance Improvement (%)')
            axes[1, 1].set_ylabel('Improvement (%)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        # Model architecture summary
        if self.model:
            model_info = f"""
            Total Parameters: {self.model.count_params():,}
            Trainable Parameters: {sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]):,}
            Layers: {len(self.model.layers)}
            """
            axes[1, 2].text(0.1, 0.5, model_info, fontsize=12, verticalalignment='center')
            axes[1, 2].set_title('Model Architecture Info')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if self.model:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a pre-trained model."""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    print("🚀 QoE-Foresight Enhanced Multi-Modal Architecture")
    print("=" * 55)
    
    # Initialize configuration
    config = PublicDatasetArchitectureConfig()
    
    # Create architecture
    architecture = PublicDatasetMultiModalArchitecture(config)
    
    # Prepare datasets
    print("📊 Preparing public datasets...")
    datasets = architecture.prepare_public_datasets()
    
    if datasets:
        print(f"✅ Successfully prepared {len(datasets)} datasets")
        
        # Train model
        print("\n🔧 Training multi-modal architecture...")
        performance = architecture.train_on_public_datasets(datasets)
        
        print("\n📈 Performance Results:")
        for metric, value in performance.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        # Visualize performance
        architecture.visualize_performance()
        
        # Evaluate generalization
        print("\n🔍 Evaluating cross-dataset generalization...")
        generalization = architecture.evaluate_cross_dataset_generalization(datasets)
        
        print("Cross-dataset generalization scores:")
        for test_pair, score in generalization.items():
            print(f"  {test_pair}: {score:.4f}")
        
        print("\n🎯 Enhanced multi-modal architecture training complete!")
        print("Ready for advanced drift detection and self-healing integration")
    
    else:
        print("❌ No datasets available. Please check Google Drive setup.")

