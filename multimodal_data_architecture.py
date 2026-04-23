"""
QoE-Foresight: Performance-Aware Drift Detection and Self-Healing Framework
Enhanced Multi-Modal Data Architecture

This module implements the comprehensive data acquisition layer for the QoE-Foresight framework,
integrating network QoS metrics, device-level signals, application logs, and contextual metadata.
"""

import os
import json
import time
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import deque, defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DataStreamConfig:
    """Configuration for individual data streams"""
    name: str
    sampling_rate: float  # Hz
    buffer_size: int = 1000
    enabled: bool = True
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MultiModalConfig:
    """Configuration for multi-modal data acquisition"""
    # Network QoS Configuration
    network_config: DataStreamConfig = field(default_factory=lambda: DataStreamConfig(
        name="network_qos",
        sampling_rate=1.0,  # 1 Hz
        buffer_size=1000,
        preprocessing={
            "outlier_detection": True,
            "smoothing_window": 5,
            "normalization": "minmax"
        },
        validation_rules={
            "bandwidth_range": (0, 1000),  # Mbps
            "latency_range": (0, 1000),    # ms
            "packet_loss_range": (0, 1.0), # ratio
            "jitter_range": (0, 100)       # ms
        }
    ))
    
    # Device-level Configuration
    device_config: DataStreamConfig = field(default_factory=lambda: DataStreamConfig(
        name="device_telemetry",
        sampling_rate=0.5,  # 0.5 Hz
        buffer_size=500,
        preprocessing={
            "temperature_smoothing": True,
            "battery_interpolation": True,
            "resource_normalization": "robust"
        },
        validation_rules={
            "cpu_range": (0, 100),        # percentage
            "gpu_range": (0, 100),        # percentage
            "battery_range": (0, 100),    # percentage
            "temperature_range": (0, 100) # Celsius
        }
    ))
    
    # Application Configuration
    app_config: DataStreamConfig = field(default_factory=lambda: DataStreamConfig(
        name="application_logs",
        sampling_rate=2.0,  # 2 Hz
        buffer_size=2000,
        preprocessing={
            "event_aggregation": True,
            "buffer_smoothing": True,
            "bitrate_normalization": "standard"
        },
        validation_rules={
            "buffer_range": (0, 60),      # seconds
            "bitrate_range": (0, 50000),  # kbps
            "resolution_values": ["240p", "360p", "480p", "720p", "1080p", "4K"]
        }
    ))
    
    # Synchronization Configuration
    sync_tolerance: float = 0.1  # seconds
    fusion_window: float = 1.0   # seconds
    max_delay: float = 5.0       # seconds

class DataValidator:
    """Validates incoming data streams according to configured rules"""
    
    def __init__(self, config: MultiModalConfig):
        self.config = config
        self.validation_stats = defaultdict(lambda: {"valid": 0, "invalid": 0, "total": 0})
    
    def validate_network_data(self, data: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Validate network QoS data"""
        errors = []
        rules = self.config.network_config.validation_rules
        
        # Bandwidth validation
        if "bandwidth" in data:
            bw_min, bw_max = rules["bandwidth_range"]
            if not (bw_min <= data["bandwidth"] <= bw_max):
                errors.append(f"Bandwidth {data['bandwidth']} outside range [{bw_min}, {bw_max}]")
        
        # Latency validation
        if "latency" in data:
            lat_min, lat_max = rules["latency_range"]
            if not (lat_min <= data["latency"] <= lat_max):
                errors.append(f"Latency {data['latency']} outside range [{lat_min}, {lat_max}]")
        
        # Packet loss validation
        if "packet_loss" in data:
            pl_min, pl_max = rules["packet_loss_range"]
            if not (pl_min <= data["packet_loss"] <= pl_max):
                errors.append(f"Packet loss {data['packet_loss']} outside range [{pl_min}, {pl_max}]")
        
        # Jitter validation
        if "jitter" in data:
            jit_min, jit_max = rules["jitter_range"]
            if not (jit_min <= data["jitter"] <= jit_max):
                errors.append(f"Jitter {data['jitter']} outside range [{jit_min}, {jit_max}]")
        
        is_valid = len(errors) == 0
        self.validation_stats["network"]["total"] += 1
        self.validation_stats["network"]["valid" if is_valid else "invalid"] += 1
        
        return is_valid, errors
    
    def validate_device_data(self, data: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Validate device telemetry data"""
        errors = []
        rules = self.config.device_config.validation_rules
        
        # CPU validation
        if "cpu_usage" in data:
            cpu_min, cpu_max = rules["cpu_range"]
            if not (cpu_min <= data["cpu_usage"] <= cpu_max):
                errors.append(f"CPU usage {data['cpu_usage']} outside range [{cpu_min}, {cpu_max}]")
        
        # GPU validation
        if "gpu_usage" in data:
            gpu_min, gpu_max = rules["gpu_range"]
            if not (gpu_min <= data["gpu_usage"] <= gpu_max):
                errors.append(f"GPU usage {data['gpu_usage']} outside range [{gpu_min}, {gpu_max}]")
        
        # Battery validation
        if "battery_level" in data:
            bat_min, bat_max = rules["battery_range"]
            if not (bat_min <= data["battery_level"] <= bat_max):
                errors.append(f"Battery level {data['battery_level']} outside range [{bat_min}, {bat_max}]")
        
        # Temperature validation
        if "temperature" in data:
            temp_min, temp_max = rules["temperature_range"]
            if not (temp_min <= data["temperature"] <= temp_max):
                errors.append(f"Temperature {data['temperature']} outside range [{temp_min}, {temp_max}]")
        
        is_valid = len(errors) == 0
        self.validation_stats["device"]["total"] += 1
        self.validation_stats["device"]["valid" if is_valid else "invalid"] += 1
        
        return is_valid, errors
    
    def validate_application_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate application log data"""
        errors = []
        rules = self.config.app_config.validation_rules
        
        # Buffer validation
        if "buffer_occupancy" in data:
            buf_min, buf_max = rules["buffer_range"]
            if not (buf_min <= data["buffer_occupancy"] <= buf_max):
                errors.append(f"Buffer occupancy {data['buffer_occupancy']} outside range [{buf_min}, {buf_max}]")
        
        # Bitrate validation
        if "bitrate" in data:
            br_min, br_max = rules["bitrate_range"]
            if not (br_min <= data["bitrate"] <= br_max):
                errors.append(f"Bitrate {data['bitrate']} outside range [{br_min}, {br_max}]")
        
        # Resolution validation
        if "resolution" in data:
            valid_resolutions = rules["resolution_values"]
            if data["resolution"] not in valid_resolutions:
                errors.append(f"Resolution {data['resolution']} not in valid values {valid_resolutions}")
        
        is_valid = len(errors) == 0
        self.validation_stats["application"]["total"] += 1
        self.validation_stats["application"]["valid" if is_valid else "invalid"] += 1
        
        return is_valid, errors
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get validation statistics report"""
        report = {}
        for stream_name, stats in self.validation_stats.items():
            if stats["total"] > 0:
                report[stream_name] = {
                    "total_samples": stats["total"],
                    "valid_samples": stats["valid"],
                    "invalid_samples": stats["invalid"],
                    "validity_rate": stats["valid"] / stats["total"]
                }
        return report

class DataPreprocessor:
    """Preprocesses and normalizes multi-modal data streams"""
    
    def __init__(self, config: MultiModalConfig):
        self.config = config
        self.scalers = {}
        self.filters = {}
        self.history_buffers = defaultdict(lambda: deque(maxlen=100))
    
    def _get_or_create_scaler(self, stream_name: str, method: str):
        """Get or create scaler for a data stream"""
        key = f"{stream_name}_{method}"
        if key not in self.scalers:
            if method == "minmax":
                self.scalers[key] = MinMaxScaler()
            elif method == "standard":
                self.scalers[key] = StandardScaler()
            elif method == "robust":
                self.scalers[key] = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
        return self.scalers[key]
    
    def _apply_outlier_detection(self, data: np.ndarray, method: str = "iqr", threshold: float = 1.5) -> np.ndarray:
        """Apply outlier detection and correction"""
        if method == "iqr":
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Replace outliers with median
            median_val = np.median(data)
            data_clean = np.where((data < lower_bound) | (data > upper_bound), median_val, data)
            return data_clean
        elif method == "zscore":
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            median_val = np.median(data)
            data_clean = np.where(z_scores > threshold, median_val, data)
            return data_clean
        else:
            return data
    
    def _apply_smoothing(self, data: np.ndarray, window_size: int = 5, method: str = "moving_average") -> np.ndarray:
        """Apply smoothing to data"""
        if len(data) < window_size:
            return data
        
        if method == "moving_average":
            return np.convolve(data, np.ones(window_size)/window_size, mode='same')
        elif method == "savgol":
            if len(data) >= window_size and window_size % 2 == 1:
                return signal.savgol_filter(data, window_size, 3)
            else:
                return data
        elif method == "exponential":
            alpha = 2.0 / (window_size + 1)
            smoothed = np.zeros_like(data)
            smoothed[0] = data[0]
            for i in range(1, len(data)):
                smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
            return smoothed
        else:
            return data
    
    def preprocess_network_data(self, data: Dict[str, float], timestamp: float) -> Dict[str, float]:
        """Preprocess network QoS data"""
        config = self.config.network_config.preprocessing
        processed_data = data.copy()
        
        # Store in history buffer
        for key, value in data.items():
            self.history_buffers[f"network_{key}"].append(value)
        
        # Apply outlier detection
        if config.get("outlier_detection", False):
            for key in ["bandwidth", "latency", "packet_loss", "jitter"]:
                if key in processed_data and len(self.history_buffers[f"network_{key}"]) > 10:
                    history = np.array(list(self.history_buffers[f"network_{key}"]))
                    cleaned = self._apply_outlier_detection(history)
                    processed_data[key] = cleaned[-1]  # Use the latest cleaned value
        
        # Apply smoothing
        window_size = config.get("smoothing_window", 5)
        if window_size > 1:
            for key in ["bandwidth", "latency", "jitter"]:
                if key in processed_data and len(self.history_buffers[f"network_{key}"]) >= window_size:
                    history = np.array(list(self.history_buffers[f"network_{key}"]))
                    smoothed = self._apply_smoothing(history, window_size)
                    processed_data[key] = smoothed[-1]
        
        # Apply normalization
        norm_method = config.get("normalization", "minmax")
        if norm_method and norm_method != "none":
            scaler = self._get_or_create_scaler("network", norm_method)
            
            # Fit scaler if we have enough history
            if len(self.history_buffers["network_bandwidth"]) > 50:
                history_data = []
                for key in ["bandwidth", "latency", "packet_loss", "jitter"]:
                    if key in processed_data:
                        history_data.append(list(self.history_buffers[f"network_{key}"]))
                
                if history_data:
                    history_array = np.array(history_data).T
                    if not hasattr(scaler, 'scale_'):
                        scaler.fit(history_array)
                    
                    # Transform current data
                    current_array = np.array([[processed_data.get(key, 0) for key in ["bandwidth", "latency", "packet_loss", "jitter"]]])
                    normalized = scaler.transform(current_array)[0]
                    
                    for i, key in enumerate(["bandwidth", "latency", "packet_loss", "jitter"]):
                        if key in processed_data:
                            processed_data[f"{key}_normalized"] = normalized[i]
        
        processed_data["timestamp"] = timestamp
        return processed_data
    
    def preprocess_device_data(self, data: Dict[str, float], timestamp: float) -> Dict[str, float]:
        """Preprocess device telemetry data"""
        config = self.config.device_config.preprocessing
        processed_data = data.copy()
        
        # Store in history buffer
        for key, value in data.items():
            self.history_buffers[f"device_{key}"].append(value)
        
        # Apply temperature smoothing
        if config.get("temperature_smoothing", False) and "temperature" in processed_data:
            if len(self.history_buffers["device_temperature"]) >= 5:
                history = np.array(list(self.history_buffers["device_temperature"]))
                smoothed = self._apply_smoothing(history, 5, "exponential")
                processed_data["temperature"] = smoothed[-1]
        
        # Apply battery interpolation for missing values
        if config.get("battery_interpolation", False) and "battery_level" in processed_data:
            if processed_data["battery_level"] is None or np.isnan(processed_data["battery_level"]):
                if len(self.history_buffers["device_battery_level"]) > 1:
                    # Linear interpolation
                    recent_values = [v for v in list(self.history_buffers["device_battery_level"])[-5:] if v is not None and not np.isnan(v)]
                    if recent_values:
                        processed_data["battery_level"] = np.mean(recent_values)
        
        # Apply resource normalization
        norm_method = config.get("resource_normalization", "robust")
        if norm_method and norm_method != "none":
            scaler = self._get_or_create_scaler("device", norm_method)
            
            # Fit scaler if we have enough history
            if len(self.history_buffers["device_cpu_usage"]) > 30:
                history_data = []
                for key in ["cpu_usage", "gpu_usage", "battery_level", "temperature"]:
                    if key in processed_data:
                        history_values = [v for v in list(self.history_buffers[f"device_{key}"]) if v is not None and not np.isnan(v)]
                        if history_values:
                            history_data.append(history_values[-30:])  # Use last 30 values
                
                if history_data and all(len(h) > 10 for h in history_data):
                    # Pad to same length
                    max_len = max(len(h) for h in history_data)
                    history_array = np.array([h + [h[-1]] * (max_len - len(h)) for h in history_data]).T
                    
                    if not hasattr(scaler, 'scale_'):
                        scaler.fit(history_array)
                    
                    # Transform current data
                    current_values = []
                    current_keys = []
                    for key in ["cpu_usage", "gpu_usage", "battery_level", "temperature"]:
                        if key in processed_data and processed_data[key] is not None and not np.isnan(processed_data[key]):
                            current_values.append(processed_data[key])
                            current_keys.append(key)
                    
                    if current_values:
                        current_array = np.array([current_values])
                        normalized = scaler.transform(current_array)[0]
                        
                        for i, key in enumerate(current_keys):
                            processed_data[f"{key}_normalized"] = normalized[i]
        
        processed_data["timestamp"] = timestamp
        return processed_data
    
    def preprocess_application_data(self, data: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Preprocess application log data"""
        config = self.config.app_config.preprocessing
        processed_data = data.copy()
        
        # Store numeric values in history buffer
        for key, value in data.items():
            if isinstance(value, (int, float)):
                self.history_buffers[f"app_{key}"].append(value)
        
        # Apply event aggregation
        if config.get("event_aggregation", False):
            # Aggregate stall events over time window
            if "stall_events" in processed_data:
                window_size = 10
                if len(self.history_buffers["app_stall_events"]) >= window_size:
                    recent_stalls = list(self.history_buffers["app_stall_events"])[-window_size:]
                    processed_data["stall_rate"] = sum(recent_stalls) / window_size
        
        # Apply buffer smoothing
        if config.get("buffer_smoothing", False) and "buffer_occupancy" in processed_data:
            if len(self.history_buffers["app_buffer_occupancy"]) >= 5:
                history = np.array(list(self.history_buffers["app_buffer_occupancy"]))
                smoothed = self._apply_smoothing(history, 5, "moving_average")
                processed_data["buffer_occupancy"] = smoothed[-1]
        
        # Apply bitrate normalization
        norm_method = config.get("bitrate_normalization", "standard")
        if norm_method and norm_method != "none" and "bitrate" in processed_data:
            scaler = self._get_or_create_scaler("application_bitrate", norm_method)
            
            if len(self.history_buffers["app_bitrate"]) > 20:
                history = np.array(list(self.history_buffers["app_bitrate"])).reshape(-1, 1)
                
                if not hasattr(scaler, 'scale_'):
                    scaler.fit(history)
                
                current_bitrate = np.array([[processed_data["bitrate"]]])
                normalized_bitrate = scaler.transform(current_bitrate)[0][0]
                processed_data["bitrate_normalized"] = normalized_bitrate
        
        processed_data["timestamp"] = timestamp
        return processed_data
    
    def save_scalers(self, filepath: str):
        """Save all fitted scalers"""
        joblib.dump(self.scalers, filepath)
        logger.info(f"Saved {len(self.scalers)} scalers to {filepath}")
    
    def load_scalers(self, filepath: str):
        """Load fitted scalers"""
        if os.path.exists(filepath):
            self.scalers = joblib.load(filepath)
            logger.info(f"Loaded {len(self.scalers)} scalers from {filepath}")
        else:
            logger.warning(f"Scaler file not found: {filepath}")

class TemporalSynchronizer:
    """Synchronizes multi-modal data streams with different sampling rates"""
    
    def __init__(self, config: MultiModalConfig):
        self.config = config
        self.stream_buffers = {
            "network": deque(maxlen=1000),
            "device": deque(maxlen=500),
            "application": deque(maxlen=2000)
        }
        self.last_sync_time = time.time()
        self.sync_stats = {
            "total_syncs": 0,
            "successful_syncs": 0,
            "dropped_samples": 0
        }
    
    def add_network_sample(self, data: Dict[str, float], timestamp: float):
        """Add network QoS sample"""
        self.stream_buffers["network"].append({"data": data, "timestamp": timestamp})
    
    def add_device_sample(self, data: Dict[str, float], timestamp: float):
        """Add device telemetry sample"""
        self.stream_buffers["device"].append({"data": data, "timestamp": timestamp})
    
    def add_application_sample(self, data: Dict[str, Any], timestamp: float):
        """Add application log sample"""
        self.stream_buffers["application"].append({"data": data, "timestamp": timestamp})
    
    def _find_closest_sample(self, buffer: deque, target_time: float, tolerance: float) -> Optional[Dict]:
        """Find the closest sample to target time within tolerance"""
        if not buffer:
            return None
        
        closest_sample = None
        min_diff = float('inf')
        
        for sample in buffer:
            time_diff = abs(sample["timestamp"] - target_time)
            if time_diff <= tolerance and time_diff < min_diff:
                min_diff = time_diff
                closest_sample = sample
        
        return closest_sample
    
    def synchronize(self, target_time: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Synchronize data streams at target time"""
        if target_time is None:
            target_time = time.time()
        
        self.sync_stats["total_syncs"] += 1
        tolerance = self.config.sync_tolerance
        
        # Find closest samples for each stream
        network_sample = self._find_closest_sample(self.stream_buffers["network"], target_time, tolerance)
        device_sample = self._find_closest_sample(self.stream_buffers["device"], target_time, tolerance)
        app_sample = self._find_closest_sample(self.stream_buffers["application"], target_time, tolerance)
        
        # Check if we have at least network and application data (device is optional)
        if network_sample is None or app_sample is None:
            self.sync_stats["dropped_samples"] += 1
            return None
        
        # Create synchronized sample
        synchronized_data = {
            "timestamp": target_time,
            "network": network_sample["data"],
            "device": device_sample["data"] if device_sample else {},
            "application": app_sample["data"],
            "sync_quality": {
                "network_delay": abs(network_sample["timestamp"] - target_time),
                "device_delay": abs(device_sample["timestamp"] - target_time) if device_sample else None,
                "application_delay": abs(app_sample["timestamp"] - target_time)
            }
        }
        
        self.sync_stats["successful_syncs"] += 1
        self.last_sync_time = target_time
        
        return synchronized_data
    
    def get_sync_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics"""
        if self.sync_stats["total_syncs"] > 0:
            success_rate = self.sync_stats["successful_syncs"] / self.sync_stats["total_syncs"]
        else:
            success_rate = 0.0
        
        return {
            "total_synchronizations": self.sync_stats["total_syncs"],
            "successful_synchronizations": self.sync_stats["successful_syncs"],
            "dropped_samples": self.sync_stats["dropped_samples"],
            "success_rate": success_rate,
            "buffer_sizes": {
                "network": len(self.stream_buffers["network"]),
                "device": len(self.stream_buffers["device"]),
                "application": len(self.stream_buffers["application"])
            }
        }

class FeatureFusionEngine:
    """Fuses synchronized multi-modal data into unified feature tensors"""
    
    def __init__(self, config: MultiModalConfig):
        self.config = config
        self.feature_mapping = self._create_feature_mapping()
        self.fusion_history = deque(maxlen=1000)
        self.fusion_stats = {
            "total_fusions": 0,
            "feature_importance": defaultdict(float),
            "modality_contributions": defaultdict(float)
        }
    
    def _create_feature_mapping(self) -> Dict[str, Dict[str, int]]:
        """Create mapping from feature names to tensor indices"""
        mapping = {
            "network": {
                "bandwidth": 0, "bandwidth_normalized": 1,
                "latency": 2, "latency_normalized": 3,
                "packet_loss": 4, "packet_loss_normalized": 5,
                "jitter": 6, "jitter_normalized": 7
            },
            "device": {
                "cpu_usage": 8, "cpu_usage_normalized": 9,
                "gpu_usage": 10, "gpu_usage_normalized": 11,
                "battery_level": 12, "battery_level_normalized": 13,
                "temperature": 14, "temperature_normalized": 15
            },
            "application": {
                "buffer_occupancy": 16, "bitrate": 17, "bitrate_normalized": 18,
                "stall_events": 19, "stall_rate": 20, "frame_rate": 21,
                "resolution_encoded": 22, "rebuffer_duration": 23
            }
        }
        return mapping
    
    def _encode_categorical_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Encode categorical features to numerical values"""
        encoded_data = data.copy()
        
        # Encode resolution
        if "resolution" in data:
            resolution_mapping = {"240p": 0.2, "360p": 0.4, "480p": 0.6, "720p": 0.8, "1080p": 1.0, "4K": 1.2}
            encoded_data["resolution_encoded"] = resolution_mapping.get(data["resolution"], 0.6)
        
        # Encode content type if present
        if "content_type" in data:
            content_mapping = {"live": 0.0, "vod": 1.0, "interactive": 0.5}
            encoded_data["content_type_encoded"] = content_mapping.get(data["content_type"], 0.5)
        
        return encoded_data
    
    def fuse_features(self, synchronized_data: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fuse multi-modal data into unified feature tensor"""
        self.fusion_stats["total_fusions"] += 1
        
        # Initialize feature tensor
        feature_tensor = np.zeros(24)  # Total features across all modalities
        feature_metadata = {
            "timestamp": synchronized_data["timestamp"],
            "modalities_present": [],
            "feature_sources": {},
            "sync_quality": synchronized_data.get("sync_quality", {})
        }
        
        # Process network features
        if "network" in synchronized_data and synchronized_data["network"]:
            network_data = synchronized_data["network"]
            feature_metadata["modalities_present"].append("network")
            
            for feature_name, tensor_idx in self.feature_mapping["network"].items():
                if feature_name in network_data:
                    feature_tensor[tensor_idx] = network_data[feature_name]
                    feature_metadata["feature_sources"][tensor_idx] = f"network.{feature_name}"
                    self.fusion_stats["feature_importance"][feature_name] += 1
            
            self.fusion_stats["modality_contributions"]["network"] += 1
        
        # Process device features
        if "device" in synchronized_data and synchronized_data["device"]:
            device_data = synchronized_data["device"]
            feature_metadata["modalities_present"].append("device")
            
            for feature_name, tensor_idx in self.feature_mapping["device"].items():
                if feature_name in device_data and device_data[feature_name] is not None:
                    feature_tensor[tensor_idx] = device_data[feature_name]
                    feature_metadata["feature_sources"][tensor_idx] = f"device.{feature_name}"
                    self.fusion_stats["feature_importance"][feature_name] += 1
            
            self.fusion_stats["modality_contributions"]["device"] += 1
        
        # Process application features
        if "application" in synchronized_data and synchronized_data["application"]:
            app_data = self._encode_categorical_features(synchronized_data["application"])
            feature_metadata["modalities_present"].append("application")
            
            for feature_name, tensor_idx in self.feature_mapping["application"].items():
                if feature_name in app_data:
                    feature_tensor[tensor_idx] = app_data[feature_name]
                    feature_metadata["feature_sources"][tensor_idx] = f"application.{feature_name}"
                    self.fusion_stats["feature_importance"][feature_name] += 1
            
            self.fusion_stats["modality_contributions"]["application"] += 1
        
        # Store in fusion history
        fusion_record = {
            "timestamp": synchronized_data["timestamp"],
            "feature_tensor": feature_tensor.copy(),
            "metadata": feature_metadata
        }
        self.fusion_history.append(fusion_record)
        
        return feature_tensor, feature_metadata
    
    def get_feature_sequences(self, sequence_length: int, step_size: int = 1) -> Tuple[np.ndarray, List[Dict]]:
        """Extract feature sequences from fusion history"""
        if len(self.fusion_history) < sequence_length:
            return np.array([]), []
        
        sequences = []
        metadata_sequences = []
        
        for i in range(0, len(self.fusion_history) - sequence_length + 1, step_size):
            sequence_data = []
            sequence_metadata = []
            
            for j in range(i, i + sequence_length):
                sequence_data.append(self.fusion_history[j]["feature_tensor"])
                sequence_metadata.append(self.fusion_history[j]["metadata"])
            
            sequences.append(np.array(sequence_data))
            metadata_sequences.append(sequence_metadata)
        
        return np.array(sequences), metadata_sequences
    
    def get_fusion_stats(self) -> Dict[str, Any]:
        """Get feature fusion statistics"""
        stats = {
            "total_fusions": self.fusion_stats["total_fusions"],
            "fusion_history_size": len(self.fusion_history),
            "modality_contributions": dict(self.fusion_stats["modality_contributions"]),
            "feature_importance": dict(self.fusion_stats["feature_importance"])
        }
        
        # Calculate relative importance
        if self.fusion_stats["total_fusions"] > 0:
            for modality in stats["modality_contributions"]:
                stats["modality_contributions"][modality] /= self.fusion_stats["total_fusions"]
            
            for feature in stats["feature_importance"]:
                stats["feature_importance"][feature] /= self.fusion_stats["total_fusions"]
        
        return stats

class MultiModalDataAcquisition:
    """Main class for multi-modal data acquisition and processing"""
    
    def __init__(self, config: MultiModalConfig, save_dir: str = "/tmp/qoe_foresight"):
        self.config = config
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize components
        self.validator = DataValidator(config)
        self.preprocessor = DataPreprocessor(config)
        self.synchronizer = TemporalSynchronizer(config)
        self.fusion_engine = FeatureFusionEngine(config)
        
        # Runtime state
        self.is_running = False
        self.acquisition_thread = None
        self.data_buffer = deque(maxlen=10000)
        
        # Statistics
        self.acquisition_stats = {
            "start_time": None,
            "total_samples": 0,
            "network_samples": 0,
            "device_samples": 0,
            "application_samples": 0,
            "fusion_samples": 0,
            "errors": 0
        }
        
        logger.info("Multi-modal data acquisition system initialized")
    
    def start_acquisition(self):
        """Start data acquisition process"""
        if self.is_running:
            logger.warning("Data acquisition already running")
            return
        
        self.is_running = True
        self.acquisition_stats["start_time"] = time.time()
        
        # Start acquisition thread
        self.acquisition_thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        self.acquisition_thread.start()
        
        logger.info("Data acquisition started")
    
    def stop_acquisition(self):
        """Stop data acquisition process"""
        if not self.is_running:
            logger.warning("Data acquisition not running")
            return
        
        self.is_running = False
        
        if self.acquisition_thread:
            self.acquisition_thread.join(timeout=5.0)
        
        logger.info("Data acquisition stopped")
    
    def _acquisition_loop(self):
        """Main acquisition loop (runs in separate thread)"""
        last_network_time = 0
        last_device_time = 0
        last_app_time = 0
        last_fusion_time = 0
        
        network_interval = 1.0 / self.config.network_config.sampling_rate
        device_interval = 1.0 / self.config.device_config.sampling_rate
        app_interval = 1.0 / self.config.app_config.sampling_rate
        fusion_interval = self.config.fusion_window
        
        while self.is_running:
            current_time = time.time()
            
            try:
                # Simulate network data collection
                if current_time - last_network_time >= network_interval:
                    network_data = self._simulate_network_data()
                    self.add_network_sample(network_data, current_time)
                    last_network_time = current_time
                
                # Simulate device data collection
                if current_time - last_device_time >= device_interval:
                    device_data = self._simulate_device_data()
                    self.add_device_sample(device_data, current_time)
                    last_device_time = current_time
                
                # Simulate application data collection
                if current_time - last_app_time >= app_interval:
                    app_data = self._simulate_application_data()
                    self.add_application_sample(app_data, current_time)
                    last_app_time = current_time
                
                # Perform fusion
                if current_time - last_fusion_time >= fusion_interval:
                    self._perform_fusion(current_time)
                    last_fusion_time = current_time
                
                time.sleep(0.1)  # Small sleep to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in acquisition loop: {e}")
                self.acquisition_stats["errors"] += 1
    
    def _simulate_network_data(self) -> Dict[str, float]:
        """Simulate network QoS data (replace with real data collection)"""
        return {
            "bandwidth": np.random.normal(50, 10),  # Mbps
            "latency": np.random.exponential(20),   # ms
            "packet_loss": np.random.beta(1, 100), # ratio
            "jitter": np.random.gamma(2, 2)        # ms
        }
    
    def _simulate_device_data(self) -> Dict[str, float]:
        """Simulate device telemetry data (replace with real data collection)"""
        return {
            "cpu_usage": np.random.normal(60, 15),    # percentage
            "gpu_usage": np.random.normal(40, 20),    # percentage
            "battery_level": max(0, 100 - time.time() % 3600 / 36),  # percentage (decreasing)
            "temperature": np.random.normal(45, 5)    # Celsius
        }
    
    def _simulate_application_data(self) -> Dict[str, Any]:
        """Simulate application log data (replace with real data collection)"""
        resolutions = ["240p", "360p", "480p", "720p", "1080p"]
        return {
            "buffer_occupancy": np.random.normal(15, 5),  # seconds
            "bitrate": np.random.normal(2000, 500),       # kbps
            "stall_events": np.random.poisson(0.1),       # count
            "frame_rate": np.random.choice([24, 30, 60]), # fps
            "resolution": np.random.choice(resolutions),
            "rebuffer_duration": np.random.exponential(0.5)  # seconds
        }
    
    def add_network_sample(self, data: Dict[str, float], timestamp: float = None):
        """Add network QoS sample"""
        if timestamp is None:
            timestamp = time.time()
        
        # Validate data
        is_valid, errors = self.validator.validate_network_data(data)
        if not is_valid:
            logger.warning(f"Invalid network data: {errors}")
            return
        
        # Preprocess data
        processed_data = self.preprocessor.preprocess_network_data(data, timestamp)
        
        # Add to synchronizer
        self.synchronizer.add_network_sample(processed_data, timestamp)
        
        self.acquisition_stats["network_samples"] += 1
        self.acquisition_stats["total_samples"] += 1
    
    def add_device_sample(self, data: Dict[str, float], timestamp: float = None):
        """Add device telemetry sample"""
        if timestamp is None:
            timestamp = time.time()
        
        # Validate data
        is_valid, errors = self.validator.validate_device_data(data)
        if not is_valid:
            logger.warning(f"Invalid device data: {errors}")
            return
        
        # Preprocess data
        processed_data = self.preprocessor.preprocess_device_data(data, timestamp)
        
        # Add to synchronizer
        self.synchronizer.add_device_sample(processed_data, timestamp)
        
        self.acquisition_stats["device_samples"] += 1
        self.acquisition_stats["total_samples"] += 1
    
    def add_application_sample(self, data: Dict[str, Any], timestamp: float = None):
        """Add application log sample"""
        if timestamp is None:
            timestamp = time.time()
        
        # Validate data
        is_valid, errors = self.validator.validate_application_data(data)
        if not is_valid:
            logger.warning(f"Invalid application data: {errors}")
            return
        
        # Preprocess data
        processed_data = self.preprocessor.preprocess_application_data(data, timestamp)
        
        # Add to synchronizer
        self.synchronizer.add_application_sample(processed_data, timestamp)
        
        self.acquisition_stats["application_samples"] += 1
        self.acquisition_stats["total_samples"] += 1
    
    def _perform_fusion(self, timestamp: float):
        """Perform data fusion at specified timestamp"""
        # Synchronize data streams
        synchronized_data = self.synchronizer.synchronize(timestamp)
        
        if synchronized_data is None:
            return
        
        # Fuse features
        feature_tensor, metadata = self.fusion_engine.fuse_features(synchronized_data)
        
        # Store fused data
        fused_sample = {
            "timestamp": timestamp,
            "feature_tensor": feature_tensor,
            "metadata": metadata
        }
        self.data_buffer.append(fused_sample)
        
        self.acquisition_stats["fusion_samples"] += 1
    
    def get_latest_features(self, sequence_length: int = 1) -> Tuple[Optional[np.ndarray], Optional[List[Dict]]]:
        """Get latest feature sequences"""
        if len(self.data_buffer) < sequence_length:
            return None, None
        
        # Extract latest sequence
        latest_samples = list(self.data_buffer)[-sequence_length:]
        
        feature_sequence = np.array([sample["feature_tensor"] for sample in latest_samples])
        metadata_sequence = [sample["metadata"] for sample in latest_samples]
        
        return feature_sequence, metadata_sequence
    
    def get_feature_sequences(self, sequence_length: int, num_sequences: int = None) -> Tuple[np.ndarray, List[List[Dict]]]:
        """Get multiple feature sequences"""
        return self.fusion_engine.get_feature_sequences(sequence_length)
    
    def save_data(self, filepath: str = None):
        """Save collected data to file"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.save_dir, f"multimodal_data_{timestamp}.json")
        
        # Convert data buffer to serializable format
        data_to_save = []
        for sample in self.data_buffer:
            serializable_sample = {
                "timestamp": sample["timestamp"],
                "feature_tensor": sample["feature_tensor"].tolist(),
                "metadata": sample["metadata"]
            }
            data_to_save.append(serializable_sample)
        
        # Save data
        with open(filepath, 'w') as f:
            json.dump({
                "config": {
                    "network_sampling_rate": self.config.network_config.sampling_rate,
                    "device_sampling_rate": self.config.device_config.sampling_rate,
                    "app_sampling_rate": self.config.app_config.sampling_rate,
                    "fusion_window": self.config.fusion_window
                },
                "statistics": self.get_acquisition_stats(),
                "data": data_to_save
            }, f, indent=2)
        
        logger.info(f"Saved {len(data_to_save)} samples to {filepath}")
        
        # Save scalers
        scaler_filepath = filepath.replace('.json', '_scalers.joblib')
        self.preprocessor.save_scalers(scaler_filepath)
        
        return filepath
    
    def load_data(self, filepath: str):
        """Load data from file"""
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
        
        # Restore data buffer
        self.data_buffer.clear()
        for sample_data in loaded_data["data"]:
            sample = {
                "timestamp": sample_data["timestamp"],
                "feature_tensor": np.array(sample_data["feature_tensor"]),
                "metadata": sample_data["metadata"]
            }
            self.data_buffer.append(sample)
        
        logger.info(f"Loaded {len(self.data_buffer)} samples from {filepath}")
        
        # Load scalers
        scaler_filepath = filepath.replace('.json', '_scalers.joblib')
        self.preprocessor.load_scalers(scaler_filepath)
        
        return loaded_data
    
    def get_acquisition_stats(self) -> Dict[str, Any]:
        """Get comprehensive acquisition statistics"""
        current_time = time.time()
        runtime = current_time - self.acquisition_stats["start_time"] if self.acquisition_stats["start_time"] else 0
        
        stats = {
            "runtime_seconds": runtime,
            "total_samples": self.acquisition_stats["total_samples"],
            "network_samples": self.acquisition_stats["network_samples"],
            "device_samples": self.acquisition_stats["device_samples"],
            "application_samples": self.acquisition_stats["application_samples"],
            "fusion_samples": self.acquisition_stats["fusion_samples"],
            "errors": self.acquisition_stats["errors"],
            "data_buffer_size": len(self.data_buffer),
            "sampling_rates": {
                "network": self.acquisition_stats["network_samples"] / runtime if runtime > 0 else 0,
                "device": self.acquisition_stats["device_samples"] / runtime if runtime > 0 else 0,
                "application": self.acquisition_stats["application_samples"] / runtime if runtime > 0 else 0,
                "fusion": self.acquisition_stats["fusion_samples"] / runtime if runtime > 0 else 0
            },
            "validation_stats": self.validator.get_validation_report(),
            "synchronization_stats": self.synchronizer.get_sync_stats(),
            "fusion_stats": self.fusion_engine.get_fusion_stats()
        }
        
        return stats

# Example usage and testing
if __name__ == "__main__":
    # Create configuration
    config = MultiModalConfig()
    
    # Initialize data acquisition system
    acquisition_system = MultiModalDataAcquisition(config, save_dir="/tmp/qoe_foresight_test")
    
    # Start acquisition
    acquisition_system.start_acquisition()
    
    # Let it run for a few seconds
    time.sleep(10)
    
    # Get some feature sequences
    sequences, metadata = acquisition_system.get_feature_sequences(sequence_length=5)
    print(f"Generated {len(sequences)} feature sequences of length 5")
    print(f"Feature tensor shape: {sequences[0].shape if len(sequences) > 0 else 'No sequences'}")
    
    # Get statistics
    stats = acquisition_system.get_acquisition_stats()
    print(f"Acquisition statistics: {json.dumps(stats, indent=2)}")
    
    # Save data
    filepath = acquisition_system.save_data()
    print(f"Data saved to: {filepath}")
    
    # Stop acquisition
    acquisition_system.stop_acquisition()
    
    print("Multi-modal data acquisition test completed successfully!")

