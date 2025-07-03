import os
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
    "input_size": (224, 224),  # ResNet-18 input size
    "sequence_length": 5,      # Temporal frames for RL
    "overlap": 2,              # Frame overlap for sequences
    "augmentation": True
}

# RL Environment configuration
ENV_CONFIG = {
    "name": "VideoBasedPotholeEnv-v0",
    "action_space_size": 5,    # [0.3, 0.5, 0.7, 0.8, 0.9] thresholds
    "action_thresholds": [0.3, 0.5, 0.7, 0.8, 0.9],
    "reward_correct": 10,
    "reward_false_positive": -5,
    "reward_missed": -20,
    "max_steps_per_episode": 100
}

# Model configuration
MODEL_CONFIG = {
    "backbone": "resnet18",
    "lstm_hidden_size": 256,
    "lstm_layers": 2,
    "dropout": 0.3,
    "learning_rate": 0.001
}

# Training configuration
TRAINING_CONFIG = {
    "total_timesteps": 50000,
    "batch_size": 32,
    "buffer_size": 10000,
    "learning_starts": 1000,
    "target_update_interval": 1000,
    "exploration_fraction": 0.1,
    "exploration_final_eps": 0.02
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
