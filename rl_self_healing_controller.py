"""
QoE-Foresight: Reinforcement Learning Self-Healing Controller
Advanced MDP formulation with DQN agent for optimal recovery strategies

This module implements the sophisticated self-healing controller using reinforcement learning
to dynamically optimize responses to concept drift in QoE prediction systems.
"""

import os
import json
import time
import logging
import random
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from collections import deque, defaultdict
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Types of self-healing actions"""
    NO_ACTION = "no_action"
    DECREASE_BITRATE = "decrease_bitrate"
    INCREASE_BITRATE = "increase_bitrate"
    ADJUST_BUFFER = "adjust_buffer"
    CHANGE_RESOLUTION = "change_resolution"
    SWITCH_SERVER = "switch_server"
    REFRESH_CACHE = "refresh_cache"
    ENABLE_PREFETCH = "enable_prefetch"
    ADJUST_CODEC = "adjust_codec"
    OPTIMIZE_NETWORK = "optimize_network"

class DriftSeverity(Enum):
    """Drift severity levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class SystemState:
    """Represents the current system state for MDP"""
    # QoE-related state
    current_qoe: float
    predicted_qoe: float
    qoe_trend: float  # Recent QoE trend
    qoe_variance: float  # QoE stability
    
    # Drift-related state
    drift_detected: bool
    drift_type: str  # "none", "abrupt", "gradual", "incremental"
    drift_severity: DriftSeverity
    drift_confidence: float
    drift_persistence: float
    
    # Network state
    bandwidth: float
    latency: float
    packet_loss: float
    jitter: float
    network_stability: float
    
    # Device state
    cpu_usage: float
    gpu_usage: float
    battery_level: float
    temperature: float
    device_performance: float
    
    # Application state
    buffer_occupancy: float
    bitrate: float
    resolution: float  # Encoded as numeric value
    stall_events: int
    frame_rate: float
    
    # Context state
    time_of_day: float  # 0-1 normalized
    content_type: float  # Encoded content type
    user_activity: float  # User engagement level
    
    # Resource constraints
    available_actions: List[ActionType]
    resource_cost_sensitivity: float
    
    def to_vector(self) -> np.ndarray:
        """Convert state to feature vector"""
        vector = np.array([
            self.current_qoe,
            self.predicted_qoe,
            self.qoe_trend,
            self.qoe_variance,
            float(self.drift_detected),
            self.drift_severity.value,
            self.drift_confidence,
            self.drift_persistence,
            self.bandwidth,
            self.latency,
            self.packet_loss,
            self.jitter,
            self.network_stability,
            self.cpu_usage,
            self.gpu_usage,
            self.battery_level,
            self.temperature,
            self.device_performance,
            self.buffer_occupancy,
            self.bitrate,
            self.resolution,
            self.stall_events,
            self.frame_rate,
            self.time_of_day,
            self.content_type,
            self.user_activity,
            self.resource_cost_sensitivity
        ])
        return vector

@dataclass
class ActionResult:
    """Result of executing a self-healing action"""
    action: ActionType
    success: bool
    qoe_improvement: float
    resource_cost: float
    execution_time: float
    side_effects: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RLConfig:
    """Configuration for reinforcement learning controller"""
    # State space configuration
    state_dim: int = 27
    action_dim: int = len(ActionType)
    
    # DQN configuration
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    learning_rate: float = 0.0005
    discount_factor: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Experience replay configuration
    replay_buffer_size: int = 50000
    batch_size: int = 64
    target_update_frequency: int = 1000
    min_replay_size: int = 1000
    
    # Training configuration
    training_frequency: int = 4
    max_episodes: int = 10000
    max_steps_per_episode: int = 1000
    
    # Reward configuration
    qoe_improvement_weight: float = 1.0
    resource_cost_weight: float = 0.3
    stability_weight: float = 0.2
    efficiency_weight: float = 0.1
    
    # Advanced features
    use_double_dqn: bool = True
    use_dueling_dqn: bool = True
    use_prioritized_replay: bool = True
    use_noisy_networks: bool = False
    
    # Multi-objective optimization
    enable_multi_objective: bool = True
    pareto_front_size: int = 10

class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer for DQN"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = 0.001
        
        # Storage
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # Segment tree for efficient sampling
        self._build_segment_tree()
    
    def _build_segment_tree(self):
        """Build segment tree for efficient priority sampling"""
        # Simple implementation - can be optimized with proper segment tree
        pass
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool, priority: Optional[float] = None):
        """Add experience to buffer"""
        experience = (state, action, reward, next_state, done)
        
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.position] = experience
        
        # Set priority (max priority for new experiences)
        if priority is None:
            priority = np.max(self.priorities[:self.size]) if self.size > 0 else 1.0
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """Sample batch with prioritized sampling"""
        if self.size < batch_size:
            return [], np.array([]), np.array([])
        
        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        # Compute importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small epsilon to avoid zero priority
    
    def __len__(self):
        return self.size

class DuelingDQN(tf.keras.Model):
    """Dueling DQN architecture for value and advantage estimation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int]):
        super(DuelingDQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature layers
        self.feature_layers = []
        for i, units in enumerate(hidden_layers[:-1]):
            self.feature_layers.append(Dense(units, activation='relu', name=f'feature_{i}'))
            self.feature_layers.append(BatchNormalization())
            self.feature_layers.append(Dropout(0.2))
        
        # Value stream
        self.value_dense = Dense(hidden_layers[-1], activation='relu', name='value_dense')
        self.value_output = Dense(1, name='value_output')
        
        # Advantage stream
        self.advantage_dense = Dense(hidden_layers[-1], activation='relu', name='advantage_dense')
        self.advantage_output = Dense(action_dim, name='advantage_output')
    
    def call(self, inputs, training=None):
        """Forward pass"""
        x = inputs
        
        # Shared features
        for layer in self.feature_layers:
            x = layer(x, training=training)
        
        # Value stream
        value = self.value_dense(x, training=training)
        value = self.value_output(value)
        
        # Advantage stream
        advantage = self.advantage_dense(x, training=training)
        advantage = self.advantage_output(advantage)
        
        # Combine value and advantage
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
        q_values = value + advantage - advantage_mean
        
        return q_values

class AdvancedDQNAgent:
    """Advanced DQN agent with multiple enhancements"""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        
        # Networks
        if config.use_dueling_dqn:
            self.q_network = DuelingDQN(config.state_dim, config.action_dim, config.hidden_layers)
            self.target_network = DuelingDQN(config.state_dim, config.action_dim, config.hidden_layers)
        else:
            self.q_network = self._build_standard_dqn()
            self.target_network = self._build_standard_dqn()
        
        # Optimizers
        self.optimizer = Adam(learning_rate=config.learning_rate)
        
        # Experience replay
        if config.use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(config.replay_buffer_size)
        else:
            self.replay_buffer = deque(maxlen=config.replay_buffer_size)
        
        # Training state
        self.epsilon = config.epsilon_start
        self.step_count = 0
        self.episode_count = 0
        self.training_step = 0
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.loss_history = []
        self.q_value_history = []
        
        # Multi-objective tracking
        self.pareto_front = []
        
        # Initialize target network
        self._update_target_network()
        
        logger.info("Advanced DQN agent initialized")
    
    def _build_standard_dqn(self) -> tf.keras.Model:
        """Build standard DQN architecture"""
        model = Sequential()
        model.add(Input(shape=(self.state_dim,)))
        
        for i, units in enumerate(self.config.hidden_layers):
            model.add(Dense(units, activation='relu', kernel_regularizer=l2(0.001)))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
        
        model.add(Dense(self.action_dim, activation='linear'))
        return model
    
    def _update_target_network(self):
        """Update target network weights"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def select_action(self, state: np.ndarray, available_actions: Optional[List[int]] = None,
                     training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            # Random action
            if available_actions:
                return np.random.choice(available_actions)
            else:
                return np.random.randint(self.action_dim)
        else:
            # Greedy action
            state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
            q_values = self.q_network(state_tensor, training=False)
            q_values_np = q_values.numpy()[0]
            
            # Mask unavailable actions
            if available_actions:
                masked_q_values = np.full(self.action_dim, -np.inf)
                masked_q_values[available_actions] = q_values_np[available_actions]
                return np.argmax(masked_q_values)
            else:
                return np.argmax(q_values_np)
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        if self.config.use_prioritized_replay:
            # Compute TD error for priority
            state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
            next_state_tensor = tf.expand_dims(tf.convert_to_tensor(next_state, dtype=tf.float32), 0)
            
            current_q = self.q_network(state_tensor, training=False)[0, action]
            
            if done:
                target_q = reward
            else:
                if self.config.use_double_dqn:
                    # Double DQN: use main network to select action, target network to evaluate
                    next_action = tf.argmax(self.q_network(next_state_tensor, training=False), axis=1)[0]
                    target_q = reward + self.config.discount_factor * \
                              self.target_network(next_state_tensor, training=False)[0, next_action]
                else:
                    target_q = reward + self.config.discount_factor * \
                              tf.reduce_max(self.target_network(next_state_tensor, training=False))
            
            td_error = abs(float(current_q - target_q))
            self.replay_buffer.add(state, action, reward, next_state, done, td_error)
        else:
            self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """Perform one training step"""
        if self.config.use_prioritized_replay:
            if len(self.replay_buffer) < self.config.min_replay_size:
                return
            
            experiences, indices, weights = self.replay_buffer.sample(self.config.batch_size)
            if not experiences:
                return
        else:
            if len(self.replay_buffer) < self.config.min_replay_size:
                return
            
            experiences = random.sample(self.replay_buffer, self.config.batch_size)
            weights = np.ones(self.config.batch_size)
            indices = None
        
        # Unpack experiences
        states = np.array([e[0] for e in experiences])
        actions = np.array([e[1] for e in experiences])
        rewards = np.array([e[2] for e in experiences])
        next_states = np.array([e[3] for e in experiences])
        dones = np.array([e[4] for e in experiences])
        
        # Convert to tensors
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones_tensor = tf.convert_to_tensor(dones, dtype=tf.float32)
        weights_tensor = tf.convert_to_tensor(weights, dtype=tf.float32)
        
        # Compute target Q-values
        if self.config.use_double_dqn:
            # Double DQN
            next_actions = tf.argmax(self.q_network(next_states_tensor, training=False), axis=1)
            next_q_values = tf.reduce_sum(
                self.target_network(next_states_tensor, training=False) * 
                tf.one_hot(next_actions, self.action_dim), axis=1
            )
        else:
            next_q_values = tf.reduce_max(self.target_network(next_states_tensor, training=False), axis=1)
        
        target_q_values = rewards_tensor + (1 - dones_tensor) * self.config.discount_factor * next_q_values
        
        # Training step
        with tf.GradientTape() as tape:
            current_q_values = self.q_network(states_tensor, training=True)
            action_q_values = tf.reduce_sum(
                current_q_values * tf.one_hot(actions, self.action_dim), axis=1
            )
            
            # Compute loss
            td_errors = target_q_values - action_q_values
            loss = tf.reduce_mean(weights_tensor * tf.square(td_errors))
        
        # Apply gradients
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        # Update priorities if using prioritized replay
        if self.config.use_prioritized_replay and indices is not None:
            new_priorities = np.abs(td_errors.numpy()) + 1e-6
            self.replay_buffer.update_priorities(indices, new_priorities)
        
        # Track statistics
        self.loss_history.append(float(loss))
        self.q_value_history.append(float(tf.reduce_mean(current_q_values)))
        
        self.training_step += 1
        
        # Update target network
        if self.training_step % self.config.target_update_frequency == 0:
            self._update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.config.epsilon_end:
            self.epsilon *= self.config.epsilon_decay
    
    def update_pareto_front(self, qoe_improvement: float, resource_cost: float):
        """Update Pareto front for multi-objective optimization"""
        if not self.config.enable_multi_objective:
            return
        
        new_point = (qoe_improvement, -resource_cost)  # Negative cost for maximization
        
        # Check if point is dominated
        dominated = False
        for point in self.pareto_front:
            if point[0] >= new_point[0] and point[1] >= new_point[1]:
                if point[0] > new_point[0] or point[1] > new_point[1]:
                    dominated = True
                    break
        
        if not dominated:
            # Remove dominated points
            self.pareto_front = [
                point for point in self.pareto_front
                if not (new_point[0] >= point[0] and new_point[1] >= point[1] and
                       (new_point[0] > point[0] or new_point[1] > point[1]))
            ]
            
            # Add new point
            self.pareto_front.append(new_point)
            
            # Limit size
            if len(self.pareto_front) > self.config.pareto_front_size:
                # Keep diverse points (simple heuristic)
                self.pareto_front.sort()
                step = len(self.pareto_front) // self.config.pareto_front_size
                self.pareto_front = self.pareto_front[::step][:self.config.pareto_front_size]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        stats = {
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "training_step": self.training_step,
            "epsilon": self.epsilon,
            "replay_buffer_size": len(self.replay_buffer),
            "pareto_front_size": len(self.pareto_front)
        }
        
        if self.episode_rewards:
            stats.update({
                "avg_episode_reward": np.mean(self.episode_rewards[-100:]),
                "max_episode_reward": np.max(self.episode_rewards),
                "min_episode_reward": np.min(self.episode_rewards)
            })
        
        if self.loss_history:
            stats.update({
                "avg_loss": np.mean(self.loss_history[-100:]),
                "recent_loss": self.loss_history[-1] if self.loss_history else 0
            })
        
        if self.q_value_history:
            stats.update({
                "avg_q_value": np.mean(self.q_value_history[-100:]),
                "recent_q_value": self.q_value_history[-1] if self.q_value_history else 0
            })
        
        return stats
    
    def save_model(self, filepath: str):
        """Save agent model and state"""
        try:
            # Build models if not already built
            dummy_input = tf.zeros((1, self.state_dim))
            self.q_network(dummy_input)
            self.target_network(dummy_input)
            
            # Save main network
            self.q_network.save_weights(filepath + "_q_network.weights.h5")
            self.target_network.save_weights(filepath + "_target_network.weights.h5")
            logger.info(f"Model weights saved to {filepath}")
        except Exception as e:
            logger.warning(f"Could not save model weights: {e}")
        
        # Save agent state
        agent_state = {
            "config": self.config,
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "training_step": self.training_step,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "pareto_front": self.pareto_front
        }
        
        with open(filepath + "_agent_state.json", 'w') as f:
            json.dump(agent_state, f, indent=2, default=str)
        
        logger.info(f"DQN agent saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load agent model and state"""
        # Load networks
        if os.path.exists(filepath + "_q_network.weights.h5"):
            self.q_network.load_weights(filepath + "_q_network.weights.h5")
            self.target_network.load_weights(filepath + "_target_network.weights.h5")
        
        # Load agent state
        if os.path.exists(filepath + "_agent_state.json"):
            with open(filepath + "_agent_state.json", 'r') as f:
                agent_state = json.load(f)
            
            self.epsilon = agent_state.get("epsilon", self.config.epsilon_start)
            self.step_count = agent_state.get("step_count", 0)
            self.episode_count = agent_state.get("episode_count", 0)
            self.training_step = agent_state.get("training_step", 0)
            self.episode_rewards = agent_state.get("episode_rewards", [])
            self.episode_lengths = agent_state.get("episode_lengths", [])
            self.pareto_front = agent_state.get("pareto_front", [])
        
        logger.info(f"DQN agent loaded from {filepath}")

class ActionExecutor:
    """Executes self-healing actions and measures their effects"""
    
    def __init__(self):
        self.execution_history = []
        self.action_effectiveness = defaultdict(list)
        
    def execute_action(self, action: ActionType, current_state: SystemState) -> ActionResult:
        """Execute a self-healing action and return the result"""
        start_time = time.time()
        
        # Simulate action execution (replace with real implementation)
        success, qoe_improvement, resource_cost, side_effects = self._simulate_action_execution(
            action, current_state
        )
        
        execution_time = time.time() - start_time
        
        result = ActionResult(
            action=action,
            success=success,
            qoe_improvement=qoe_improvement,
            resource_cost=resource_cost,
            execution_time=execution_time,
            side_effects=side_effects,
            metadata={
                "timestamp": time.time(),
                "state_snapshot": current_state.to_vector().tolist()
            }
        )
        
        # Track execution history
        self.execution_history.append(result)
        self.action_effectiveness[action].append(qoe_improvement)
        
        return result
    
    def _simulate_action_execution(self, action: ActionType, state: SystemState) -> Tuple[bool, float, float, Dict]:
        """Simulate action execution (replace with real implementation)"""
        # This is a simplified simulation - replace with actual action execution
        
        if action == ActionType.NO_ACTION:
            return True, 0.0, 0.0, {}
        
        elif action == ActionType.DECREASE_BITRATE:
            if state.bitrate > 500:  # Can decrease
                qoe_improvement = 0.1 if state.network_stability < 0.5 else -0.05
                resource_cost = 0.1
                side_effects = {"bitrate_change": -500}
                return True, qoe_improvement, resource_cost, side_effects
            else:
                return False, 0.0, 0.0, {"error": "bitrate_too_low"}
        
        elif action == ActionType.INCREASE_BITRATE:
            if state.bandwidth > state.bitrate * 1.5:  # Sufficient bandwidth
                qoe_improvement = 0.2 if state.network_stability > 0.7 else -0.1
                resource_cost = 0.2
                side_effects = {"bitrate_change": 500}
                return True, qoe_improvement, resource_cost, side_effects
            else:
                return False, 0.0, 0.0, {"error": "insufficient_bandwidth"}
        
        elif action == ActionType.ADJUST_BUFFER:
            qoe_improvement = 0.15 if state.buffer_occupancy < 10 else 0.05
            resource_cost = 0.15
            side_effects = {"buffer_change": 5.0}
            return True, qoe_improvement, resource_cost, side_effects
        
        elif action == ActionType.CHANGE_RESOLUTION:
            if state.drift_severity.value >= 2:  # Medium or higher drift
                qoe_improvement = 0.25
                resource_cost = 0.3
                side_effects = {"resolution_change": -1}
                return True, qoe_improvement, resource_cost, side_effects
            else:
                qoe_improvement = 0.1
                resource_cost = 0.2
                side_effects = {"resolution_change": 0}
                return True, qoe_improvement, resource_cost, side_effects
        
        elif action == ActionType.SWITCH_SERVER:
            qoe_improvement = 0.3 if state.latency > 100 else 0.1
            resource_cost = 0.5
            side_effects = {"server_switch": True}
            return True, qoe_improvement, resource_cost, side_effects
        
        elif action == ActionType.REFRESH_CACHE:
            qoe_improvement = 0.2
            resource_cost = 0.25
            side_effects = {"cache_refreshed": True}
            return True, qoe_improvement, resource_cost, side_effects
        
        elif action == ActionType.ENABLE_PREFETCH:
            if state.battery_level > 30:  # Sufficient battery
                qoe_improvement = 0.15
                resource_cost = 0.4
                side_effects = {"prefetch_enabled": True}
                return True, qoe_improvement, resource_cost, side_effects
            else:
                return False, 0.0, 0.0, {"error": "low_battery"}
        
        elif action == ActionType.ADJUST_CODEC:
            qoe_improvement = 0.1
            resource_cost = 0.2
            side_effects = {"codec_changed": True}
            return True, qoe_improvement, resource_cost, side_effects
        
        elif action == ActionType.OPTIMIZE_NETWORK:
            qoe_improvement = 0.2 if state.network_stability < 0.6 else 0.05
            resource_cost = 0.3
            side_effects = {"network_optimized": True}
            return True, qoe_improvement, resource_cost, side_effects
        
        else:
            return False, 0.0, 0.0, {"error": "unknown_action"}
    
    def get_action_effectiveness(self, action: ActionType) -> Dict[str, float]:
        """Get effectiveness statistics for an action"""
        if action not in self.action_effectiveness:
            return {"mean": 0.0, "std": 0.0, "count": 0}
        
        improvements = self.action_effectiveness[action]
        return {
            "mean": np.mean(improvements),
            "std": np.std(improvements),
            "count": len(improvements),
            "success_rate": sum(1 for imp in improvements if imp > 0) / len(improvements)
        }

class RewardFunction:
    """Sophisticated reward function for multi-objective optimization"""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.reward_history = []
        
    def compute_reward(self, state: SystemState, action: ActionType, 
                      action_result: ActionResult, next_state: SystemState) -> float:
        """Compute reward for state-action-next_state transition"""
        
        # QoE improvement component
        qoe_reward = action_result.qoe_improvement * self.config.qoe_improvement_weight
        
        # Resource cost penalty
        cost_penalty = -action_result.resource_cost * self.config.resource_cost_weight
        
        # Stability reward (prefer actions that improve system stability)
        stability_improvement = next_state.qoe_variance - state.qoe_variance
        stability_reward = -stability_improvement * self.config.stability_weight  # Negative because lower variance is better
        
        # Efficiency reward (prefer faster actions)
        efficiency_reward = (1.0 / (action_result.execution_time + 0.1)) * self.config.efficiency_weight
        
        # Drift-specific rewards
        drift_reward = 0.0
        if state.drift_detected:
            if action_result.qoe_improvement > 0:
                # Bonus for successful drift mitigation
                drift_reward = 0.5 * state.drift_severity.value / 4.0
            else:
                # Penalty for ineffective action during drift
                drift_reward = -0.2 * state.drift_severity.value / 4.0
        
        # Action appropriateness reward
        appropriateness_reward = self._compute_action_appropriateness(state, action, action_result)
        
        # Combine all components
        total_reward = (
            qoe_reward + 
            cost_penalty + 
            stability_reward + 
            efficiency_reward + 
            drift_reward + 
            appropriateness_reward
        )
        
        # Store reward components for analysis
        reward_components = {
            "total": total_reward,
            "qoe": qoe_reward,
            "cost": cost_penalty,
            "stability": stability_reward,
            "efficiency": efficiency_reward,
            "drift": drift_reward,
            "appropriateness": appropriateness_reward
        }
        
        self.reward_history.append(reward_components)
        
        return total_reward
    
    def _compute_action_appropriateness(self, state: SystemState, action: ActionType, 
                                      result: ActionResult) -> float:
        """Compute reward based on action appropriateness for current state"""
        
        # No action appropriateness
        if action == ActionType.NO_ACTION:
            if not state.drift_detected and state.current_qoe > 4.0:
                return 0.1  # Good to do nothing when everything is fine
            elif state.drift_detected:
                return -0.2  # Bad to do nothing during drift
            else:
                return 0.0
        
        # Bitrate actions appropriateness
        elif action == ActionType.DECREASE_BITRATE:
            if state.network_stability < 0.5 or state.bandwidth < state.bitrate:
                return 0.1  # Appropriate when network is unstable
            else:
                return -0.05  # Less appropriate when network is stable
        
        elif action == ActionType.INCREASE_BITRATE:
            if state.network_stability > 0.7 and state.bandwidth > state.bitrate * 2:
                return 0.1  # Appropriate when network can handle it
            else:
                return -0.05
        
        # Buffer actions appropriateness
        elif action == ActionType.ADJUST_BUFFER:
            if state.stall_events > 0 or state.buffer_occupancy < 5:
                return 0.15  # Very appropriate when buffering issues
            else:
                return 0.05
        
        # Resolution actions appropriateness
        elif action == ActionType.CHANGE_RESOLUTION:
            if state.drift_severity.value >= 2:
                return 0.1  # Appropriate during significant drift
            else:
                return 0.0
        
        # Server switching appropriateness
        elif action == ActionType.SWITCH_SERVER:
            if state.latency > 100 or state.network_stability < 0.4:
                return 0.2  # Very appropriate for network issues
            else:
                return -0.1  # Expensive action when not needed
        
        # Default appropriateness
        else:
            return 0.0
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get reward statistics"""
        if not self.reward_history:
            return {}
        
        recent_rewards = self.reward_history[-100:]
        
        stats = {}
        for component in ["total", "qoe", "cost", "stability", "efficiency", "drift", "appropriateness"]:
            values = [r[component] for r in recent_rewards]
            stats[f"{component}_mean"] = np.mean(values)
            stats[f"{component}_std"] = np.std(values)
        
        return stats

class SelfHealingController:
    """Main self-healing controller integrating all components"""
    
    def __init__(self, config: RLConfig):
        self.config = config
        
        # Initialize components
        self.agent = AdvancedDQNAgent(config)
        self.action_executor = ActionExecutor()
        self.reward_function = RewardFunction(config)
        
        # State tracking
        self.current_state = None
        self.state_history = deque(maxlen=1000)
        
        # Performance tracking
        self.episode_stats = []
        self.total_episodes = 0
        self.total_steps = 0
        
        logger.info("Self-healing controller initialized")
    
    def create_state_from_data(self, qoe_data: Dict[str, float], 
                              drift_data: Dict[str, Any],
                              network_data: Dict[str, float],
                              device_data: Dict[str, float],
                              app_data: Dict[str, float]) -> SystemState:
        """Create system state from multi-modal data"""
        
        # Compute derived features
        qoe_trend = self._compute_qoe_trend()
        qoe_variance = self._compute_qoe_variance()
        network_stability = self._compute_network_stability(network_data)
        device_performance = self._compute_device_performance(device_data)
        
        # Determine available actions based on current constraints
        available_actions = self._determine_available_actions(network_data, device_data, app_data)
        
        state = SystemState(
            current_qoe=qoe_data.get("current_qoe", 3.0),
            predicted_qoe=qoe_data.get("predicted_qoe", 3.0),
            qoe_trend=qoe_trend,
            qoe_variance=qoe_variance,
            drift_detected=drift_data.get("detected", False),
            drift_type=drift_data.get("type", "none"),
            drift_severity=DriftSeverity(drift_data.get("severity", 0)),
            drift_confidence=drift_data.get("confidence", 0.0),
            drift_persistence=drift_data.get("persistence", 0.0),
            bandwidth=network_data.get("bandwidth", 10.0),
            latency=network_data.get("latency", 50.0),
            packet_loss=network_data.get("packet_loss", 0.01),
            jitter=network_data.get("jitter", 5.0),
            network_stability=network_stability,
            cpu_usage=device_data.get("cpu_usage", 50.0),
            gpu_usage=device_data.get("gpu_usage", 30.0),
            battery_level=device_data.get("battery_level", 80.0),
            temperature=device_data.get("temperature", 40.0),
            device_performance=device_performance,
            buffer_occupancy=app_data.get("buffer_occupancy", 15.0),
            bitrate=app_data.get("bitrate", 2000.0),
            resolution=app_data.get("resolution_encoded", 0.8),
            stall_events=int(app_data.get("stall_events", 0)),
            frame_rate=app_data.get("frame_rate", 30.0),
            time_of_day=time.time() % 86400 / 86400,  # Normalized time of day
            content_type=app_data.get("content_type_encoded", 0.5),
            user_activity=app_data.get("user_activity", 0.8),
            available_actions=available_actions,
            resource_cost_sensitivity=self._compute_resource_sensitivity(device_data)
        )
        
        return state
    
    def _compute_qoe_trend(self) -> float:
        """Compute recent QoE trend"""
        if len(self.state_history) < 5:
            return 0.0
        
        recent_qoe = [state.current_qoe for state in list(self.state_history)[-5:]]
        if len(recent_qoe) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(recent_qoe))
        slope, _, _, _, _ = stats.linregress(x, recent_qoe)
        return slope
    
    def _compute_qoe_variance(self) -> float:
        """Compute recent QoE variance"""
        if len(self.state_history) < 3:
            return 0.0
        
        recent_qoe = [state.current_qoe for state in list(self.state_history)[-10:]]
        return np.var(recent_qoe)
    
    def _compute_network_stability(self, network_data: Dict[str, float]) -> float:
        """Compute network stability metric"""
        # Simple stability metric based on current values
        bandwidth = network_data.get("bandwidth", 10.0)
        latency = network_data.get("latency", 50.0)
        packet_loss = network_data.get("packet_loss", 0.01)
        jitter = network_data.get("jitter", 5.0)
        
        # Normalize and combine (higher is better)
        bandwidth_score = min(bandwidth / 50.0, 1.0)  # Normalize to 50 Mbps
        latency_score = max(0, 1.0 - latency / 200.0)  # Normalize to 200ms
        loss_score = max(0, 1.0 - packet_loss * 100)  # Normalize to 1% loss
        jitter_score = max(0, 1.0 - jitter / 50.0)  # Normalize to 50ms jitter
        
        stability = (bandwidth_score + latency_score + loss_score + jitter_score) / 4.0
        return stability
    
    def _compute_device_performance(self, device_data: Dict[str, float]) -> float:
        """Compute device performance metric"""
        cpu = device_data.get("cpu_usage", 50.0)
        gpu = device_data.get("gpu_usage", 30.0)
        battery = device_data.get("battery_level", 80.0)
        temp = device_data.get("temperature", 40.0)
        
        # Normalize (higher is better, except for usage and temperature)
        cpu_score = max(0, 1.0 - cpu / 100.0)
        gpu_score = max(0, 1.0 - gpu / 100.0)
        battery_score = battery / 100.0
        temp_score = max(0, 1.0 - (temp - 20) / 60.0)  # Optimal around 20°C
        
        performance = (cpu_score + gpu_score + battery_score + temp_score) / 4.0
        return performance
    
    def _determine_available_actions(self, network_data: Dict[str, float],
                                   device_data: Dict[str, float],
                                   app_data: Dict[str, float]) -> List[ActionType]:
        """Determine which actions are available given current constraints"""
        available = [ActionType.NO_ACTION]  # Always available
        
        # Bitrate actions
        if app_data.get("bitrate", 2000) > 500:
            available.append(ActionType.DECREASE_BITRATE)
        
        if network_data.get("bandwidth", 10) > app_data.get("bitrate", 2000) * 1.5 / 1000:
            available.append(ActionType.INCREASE_BITRATE)
        
        # Buffer actions
        available.append(ActionType.ADJUST_BUFFER)
        
        # Resolution actions
        available.append(ActionType.CHANGE_RESOLUTION)
        
        # Server switching (if latency is high)
        if network_data.get("latency", 50) > 80:
            available.append(ActionType.SWITCH_SERVER)
        
        # Cache refresh
        available.append(ActionType.REFRESH_CACHE)
        
        # Prefetch (if battery is sufficient)
        if device_data.get("battery_level", 80) > 30:
            available.append(ActionType.ENABLE_PREFETCH)
        
        # Codec adjustment
        available.append(ActionType.ADJUST_CODEC)
        
        # Network optimization
        if network_data.get("packet_loss", 0.01) > 0.005:
            available.append(ActionType.OPTIMIZE_NETWORK)
        
        return available
    
    def _compute_resource_sensitivity(self, device_data: Dict[str, float]) -> float:
        """Compute resource cost sensitivity based on device state"""
        battery = device_data.get("battery_level", 80.0)
        cpu = device_data.get("cpu_usage", 50.0)
        temp = device_data.get("temperature", 40.0)
        
        # Higher sensitivity when resources are constrained
        battery_factor = 1.0 - battery / 100.0
        cpu_factor = cpu / 100.0
        temp_factor = max(0, (temp - 40) / 40.0)
        
        sensitivity = (battery_factor + cpu_factor + temp_factor) / 3.0
        return sensitivity
    
    def select_action(self, state: SystemState, training: bool = True) -> ActionType:
        """Select optimal action for current state"""
        # Convert available actions to indices
        available_indices = [action.value for action in state.available_actions]
        action_to_index = {action.value: i for i, action in enumerate(ActionType)}
        available_action_indices = [action_to_index[action] for action in available_indices]
        
        # Select action using agent
        action_index = self.agent.select_action(
            state.to_vector(), 
            available_action_indices, 
            training
        )
        
        # Convert back to action
        action = list(ActionType)[action_index]
        return action
    
    def execute_healing_step(self, qoe_data: Dict[str, float], 
                           drift_data: Dict[str, Any],
                           network_data: Dict[str, float],
                           device_data: Dict[str, float],
                           app_data: Dict[str, float],
                           training: bool = True) -> Tuple[ActionType, ActionResult, float]:
        """Execute one step of self-healing process"""
        
        # Create current state
        current_state = self.create_state_from_data(
            qoe_data, drift_data, network_data, device_data, app_data
        )
        
        # Select action
        action = self.select_action(current_state, training)
        
        # Execute action
        action_result = self.action_executor.execute_action(action, current_state)
        
        # Simulate next state (in practice, this would come from the environment)
        next_state = self._simulate_next_state(current_state, action, action_result)
        
        # Compute reward
        reward = self.reward_function.compute_reward(
            current_state, action, action_result, next_state
        )
        
        # Store experience for training
        if training and self.current_state is not None:
            done = False  # In continuous operation, episodes don't naturally end
            self.agent.store_experience(
                self.current_state.to_vector(),
                list(ActionType).index(action),
                reward,
                current_state.to_vector(),
                done
            )
            
            # Train agent
            if self.total_steps % self.config.training_frequency == 0:
                self.agent.train_step()
        
        # Update state tracking
        self.current_state = current_state
        self.state_history.append(current_state)
        self.total_steps += 1
        
        # Update Pareto front
        self.agent.update_pareto_front(action_result.qoe_improvement, action_result.resource_cost)
        
        return action, action_result, reward
    
    def _simulate_next_state(self, current_state: SystemState, action: ActionType, 
                           result: ActionResult) -> SystemState:
        """Simulate next state after action execution (simplified)"""
        # This is a simplified simulation - in practice, the next state would come from
        # the actual system after action execution
        
        next_state = SystemState(
            current_qoe=current_state.current_qoe + result.qoe_improvement,
            predicted_qoe=current_state.predicted_qoe,
            qoe_trend=current_state.qoe_trend,
            qoe_variance=current_state.qoe_variance * 0.9,  # Assume actions reduce variance
            drift_detected=current_state.drift_detected,
            drift_type=current_state.drift_type,
            drift_severity=current_state.drift_severity,
            drift_confidence=current_state.drift_confidence,
            drift_persistence=current_state.drift_persistence,
            bandwidth=current_state.bandwidth,
            latency=current_state.latency,
            packet_loss=current_state.packet_loss,
            jitter=current_state.jitter,
            network_stability=current_state.network_stability,
            cpu_usage=current_state.cpu_usage,
            gpu_usage=current_state.gpu_usage,
            battery_level=max(0, current_state.battery_level - result.resource_cost * 5),
            temperature=current_state.temperature,
            device_performance=current_state.device_performance,
            buffer_occupancy=current_state.buffer_occupancy + result.side_effects.get("buffer_change", 0),
            bitrate=current_state.bitrate + result.side_effects.get("bitrate_change", 0),
            resolution=current_state.resolution + result.side_effects.get("resolution_change", 0),
            stall_events=max(0, current_state.stall_events - 1) if result.qoe_improvement > 0 else current_state.stall_events,
            frame_rate=current_state.frame_rate,
            time_of_day=current_state.time_of_day,
            content_type=current_state.content_type,
            user_activity=current_state.user_activity,
            available_actions=current_state.available_actions,
            resource_cost_sensitivity=current_state.resource_cost_sensitivity
        )
        
        return next_state
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components"""
        stats = {
            "controller": {
                "total_episodes": self.total_episodes,
                "total_steps": self.total_steps,
                "state_history_size": len(self.state_history)
            },
            "agent": self.agent.get_statistics(),
            "reward": self.reward_function.get_reward_statistics()
        }
        
        # Action effectiveness statistics
        action_stats = {}
        for action in ActionType:
            action_stats[action.value] = self.action_executor.get_action_effectiveness(action)
        stats["action_effectiveness"] = action_stats
        
        return stats
    
    def save_controller(self, filepath: str):
        """Save complete controller state"""
        # Save agent
        self.agent.save_model(filepath + "_agent")
        
        # Save controller state
        controller_state = {
            "config": self.config,
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "episode_stats": self.episode_stats
        }
        
        with open(filepath + "_controller.json", 'w') as f:
            json.dump(controller_state, f, indent=2, default=str)
        
        # Save execution history
        execution_data = [
            {
                "action": result.action.value,
                "success": result.success,
                "qoe_improvement": result.qoe_improvement,
                "resource_cost": result.resource_cost,
                "execution_time": result.execution_time,
                "metadata": result.metadata
            }
            for result in self.action_executor.execution_history
        ]
        
        with open(filepath + "_execution_history.json", 'w') as f:
            json.dump(execution_data, f, indent=2)
        
        logger.info(f"Self-healing controller saved to {filepath}")
    
    def load_controller(self, filepath: str):
        """Load complete controller state"""
        # Load agent
        self.agent.load_model(filepath + "_agent")
        
        # Load controller state
        if os.path.exists(filepath + "_controller.json"):
            with open(filepath + "_controller.json", 'r') as f:
                controller_state = json.load(f)
            
            self.total_episodes = controller_state.get("total_episodes", 0)
            self.total_steps = controller_state.get("total_steps", 0)
            self.episode_stats = controller_state.get("episode_stats", [])
        
        logger.info(f"Self-healing controller loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    # Create configuration
    config = RLConfig()
    
    # Initialize self-healing controller
    controller = SelfHealingController(config)
    
    # Simulate self-healing episodes
    print("Testing self-healing controller...")
    
    for episode in range(10):
        episode_reward = 0
        episode_steps = 0
        
        for step in range(50):
            # Simulate multi-modal data
            qoe_data = {
                "current_qoe": 3.0 + np.random.normal(0, 0.5),
                "predicted_qoe": 3.2 + np.random.normal(0, 0.3)
            }
            
            drift_data = {
                "detected": np.random.random() < 0.1,
                "type": "gradual",
                "severity": np.random.randint(0, 4),
                "confidence": np.random.random(),
                "persistence": np.random.random()
            }
            
            network_data = {
                "bandwidth": 20 + np.random.normal(0, 5),
                "latency": 50 + np.random.normal(0, 20),
                "packet_loss": np.random.exponential(0.01),
                "jitter": np.random.gamma(2, 2)
            }
            
            device_data = {
                "cpu_usage": 50 + np.random.normal(0, 15),
                "gpu_usage": 30 + np.random.normal(0, 10),
                "battery_level": max(10, 80 - step * 0.5),
                "temperature": 40 + np.random.normal(0, 5)
            }
            
            app_data = {
                "buffer_occupancy": 15 + np.random.normal(0, 5),
                "bitrate": 2000 + np.random.normal(0, 500),
                "resolution_encoded": 0.8,
                "stall_events": np.random.poisson(0.1),
                "frame_rate": 30,
                "content_type_encoded": 0.5,
                "user_activity": 0.8
            }
            
            # Execute healing step
            action, result, reward = controller.execute_healing_step(
                qoe_data, drift_data, network_data, device_data, app_data, training=True
            )
            
            episode_reward += reward
            episode_steps += 1
            
            if step % 10 == 0:
                print(f"Episode {episode}, Step {step}: Action={action.value}, "
                      f"QoE_improvement={result.qoe_improvement:.3f}, Reward={reward:.3f}")
        
        controller.total_episodes += 1
        controller.agent.episode_rewards.append(episode_reward)
        controller.agent.episode_lengths.append(episode_steps)
        controller.agent.episode_count += 1
        
        print(f"Episode {episode} completed: Total reward={episode_reward:.3f}, Steps={episode_steps}")
    
    # Get comprehensive statistics
    stats = controller.get_comprehensive_statistics()
    print(f"\nSelf-Healing Controller Statistics:")
    print(f"Total episodes: {stats['controller']['total_episodes']}")
    print(f"Total steps: {stats['controller']['total_steps']}")
    print(f"Agent epsilon: {stats['agent']['epsilon']:.3f}")
    print(f"Average episode reward: {stats['agent'].get('avg_episode_reward', 0):.3f}")
    print(f"Replay buffer size: {stats['agent']['replay_buffer_size']}")
    print(f"Pareto front size: {stats['agent']['pareto_front_size']}")
    
    # Save the controller
    save_path = "/tmp/self_healing_controller"
    controller.save_controller(save_path)
    print(f"\nController saved to: {save_path}")
    
    print("\nReinforcement learning self-healing controller test completed successfully!")

