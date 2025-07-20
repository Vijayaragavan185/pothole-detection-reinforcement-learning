from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset configuration
DATASET_CONFIG = {
    "name": "pothole_videos_mendeley",
    "doi": "10.17632/5bwfg4v4cd.3",
    "total_videos": 619,
    "resolution": (1080, 1080),
    "frames_per_video": 48,
    "fps": 24,
    "duration_seconds": 2,
    "splits": {
        "train": 372,
        "val": 124,
        "test": 123
    }
}

# Video processing configuration
VIDEO_CONFIG = {
    "input_size": (224, 224),
    "sequence_length": 5,
    "overlap": 2,
    "augmentation": True
}

# RL Environment configuration
ENV_CONFIG = {
    "name": "VideoBasedPotholeEnv-v0",
    "action_space_size": 5,
    "action_thresholds": [0.3, 0.5, 0.7, 0.8, 0.9],
    "reward_correct": 10,
    "reward_false_positive": -5,
    "reward_missed": -20,
    "max_steps_per_episode": 100,
    "deterministic_loading": True,
    "target_sequence_count": 1000
}

# Model configuration
MODEL_CONFIG = {
    "backbone": "resnet18",
    "lstm_hidden_size": 256,
    "lstm_layers": 2,
    "dropout": 0.3,
    "learning_rate": 0.001
}

# Training configuration (non-DQN specific)
TRAINING_CONFIG = {
    "total_timesteps": 50000,
    "learning_starts": 1000,
    "exploration_fraction": 0.1
}

# ✅ FIXED: Complete DQN configurations
BASE_DQN = {
    "input_shape": (5, 224, 224, 3),
    "num_actions": 5,
    "learning_rate": 0.0005,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "batch_size": 16,
    "target_update": 100,
    "memory_size": 5000
}

DQN_CONFIGS = {
    "STANDARD_DQN": {
        **BASE_DQN,
        "use_double_dqn": False,
        "use_dueling": False,
        "use_prioritized_replay": False
    },
    "DUELING_DQN": {
        **BASE_DQN,
        "use_double_dqn": False,
        "use_dueling": True,
        "use_prioritized_replay": False,
        "memory_size": 8000
    },
    "ULTIMATE_DQN": {
        **BASE_DQN,
        "use_double_dqn": True,
        "use_dueling": True,
        "use_prioritized_replay": True,
        "learning_rate": 0.0003,
        "memory_size": 10000,
        "batch_size": 32,
        "target_update": 75
    }
}

# Paths
PATHS = {
    "raw_videos": DATA_DIR / "raw_videos",
    "processed_frames": DATA_DIR / "processed_frames", 
    "ground_truth": DATA_DIR / "ground_truth_masks",
    "splits": DATA_DIR / "splits",
    "models": RESULTS_DIR / "models",
    "logs": RESULTS_DIR / "logs",
    "metrics": RESULTS_DIR / "metrics"
}

# Create directories if they don't exist
for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# Performance and debug configs
PERFORMANCE_CONFIG = {
    "max_memory_mb": 2048,
    "max_sequences_per_load": 1000,
    "deterministic_loading": True,
    "memory_threshold": 0.8,
    "gc_frequency": 100,
    "file_size_limit_mb": 20,
    "array_size_limit": 5 * 1024 * 1024
}

DEBUG_CONFIG = {
    "verbose_loading": True,
    "log_memory_usage": True,
    "save_debug_info": True,
    "validate_data_consistency": True,
    "track_performance_metrics": True
}

# Export key configurations
__all__ = [
    'DATASET_CONFIG', 'VIDEO_CONFIG', 'ENV_CONFIG', 'MODEL_CONFIG',
    'TRAINING_CONFIG', 'DQN_CONFIGS', 'PATHS', 'PERFORMANCE_CONFIG',
    'DEBUG_CONFIG', 'BASE_DQN'
]
