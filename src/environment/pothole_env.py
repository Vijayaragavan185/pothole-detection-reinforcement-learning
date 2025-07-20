import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
import random
import json
import gc
import psutil
from collections import deque
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import ENV_CONFIG, VIDEO_CONFIG, PATHS

class VideoBasedPotholeEnv(gym.Env):
    """
    🚀 REVOLUTIONARY RL ENVIRONMENT FOR POTHOLE DETECTION! 🚀
    
    Fixed version with consistent data loading and improved stability.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, split='train', render_mode=None, max_memory_mb=2048, target_sequences=1000):
        super().__init__()
        
        print("🎯 Initializing FIXED Video-Based RL Environment...")
        
        # Environment configuration
        self.split = split
        self.render_mode = render_mode
        self.sequence_length = VIDEO_CONFIG["sequence_length"]
        self.input_size = VIDEO_CONFIG["input_size"]
        self.max_memory_mb = max_memory_mb
        self.target_sequences = target_sequences  # FIXED: Consistent loading
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(5)
        self.action_thresholds = ENV_CONFIG["action_thresholds"]
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.sequence_length, self.input_size[1], self.input_size[0], 3),
            dtype=np.float32
        )
        
        # Reward system
        self.reward_correct = ENV_CONFIG["reward_correct"]
        self.reward_false_positive = ENV_CONFIG["reward_false_positive"]
        self.reward_missed = ENV_CONFIG["reward_missed"]
        
        # Environment state
        self.current_sequence = None
        self.current_ground_truth = None
        self.episode_sequences = []
        self.episode_ground_truths = []
        self.episode_metadata = []
        
        # Performance tracking
        self.episode_count = 0
        self.total_correct_detections = 0
        self.total_false_positives = 0
        self.total_missed_detections = 0
        self.successful_loads = 0
        self.failed_loads = 0
        
        # Load dataset
        self._load_dataset_fixed()
        
        print(f"✅ FIXED RL Environment Initialized!")
        print(f"   📊 Action Space: {self.action_space.n} confidence thresholds")
        print(f"   🖼️ Observation Space: {self.observation_space.shape}")
        print(f"   💾 Dataset Split: {self.split}")
        print(f"   🎯 Available Episodes: {len(self.episode_sequences):,}")
        print(f"   📈 Target Sequences: {self.target_sequences}")
    
    def _load_dataset_fixed(self):
        """FIXED: Load dataset with consistent sequence count"""
        print(f"📊 Loading {self.split} dataset with FIXED consistent loading...")
        
        self.episode_sequences = []
        self.episode_ground_truths = []
        self.episode_metadata = []
        self.successful_loads = 0
        self.failed_loads = 0
        
        # Try to load real data first
        optimized_dir = PATHS["processed_frames"] / f"{self.split}_optimized"
        ground_truth_dir = PATHS["ground_truth"] / self.split
        
        if optimized_dir.exists() and ground_truth_dir.exists():
            self._load_real_data(optimized_dir, ground_truth_dir)
        
        # If insufficient real data, fill with synthetic data
        current_count = len(self.episode_sequences)
        if current_count < self.target_sequences:
            needed = self.target_sequences - current_count
            print(f"   🔧 Creating {needed} synthetic sequences to reach target")
            self._create_synthetic_data(needed)
        
        print(f"🚀 FIXED Dataset loaded: {len(self.episode_sequences):,} sequences")
    
    def _load_real_data(self, data_dir, gt_dir):
        """Load real data with error handling"""
        video_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        
        for video_dir in video_dirs:
            if len(self.episode_sequences) >= self.target_sequences:
                break
                
            sequences_dir = video_dir / "sequences"
            if not sequences_dir.exists():
                continue
            
            seq_files = sorted(sequences_dir.glob("*.npy"))
            for seq_file in seq_files[:10]:  # Limit per video
                if len(self.episode_sequences) >= self.target_sequences:
                    break
                
                try:
                    # Load sequence
                    sequence = self._safe_load_sequence(seq_file)
                    if sequence is None:
                        self.failed_loads += 1
                        continue
                    
                    # Load ground truth
                    video_name = video_dir.name.split('_aug')[0]
                    seq_idx = seq_file.stem.split('_')[-1]
                    
                    gt_file = gt_dir / video_name / "sequences" / f"mask_sequence_{seq_idx}.npy"
                    ground_truth = self._safe_load_ground_truth(gt_file)
                    
                    # Add to dataset
                    self.episode_sequences.append(sequence)
                    self.episode_ground_truths.append(ground_truth)
                    self.episode_metadata.append({
                        "video_name": video_name,
                        "sequence_idx": seq_idx,
                        "data_type": "real",
                        "has_ground_truth": ground_truth is not None
                    })
                    
                    self.successful_loads += 1
                    
                except Exception as e:
                    self.failed_loads += 1
                    continue
    
    def _safe_load_sequence(self, seq_file):
        """Safely load sequence with validation"""
        try:
            if not seq_file.exists():
                return None
            
            # Check file size
            file_size_mb = seq_file.stat().st_size / (1024 * 1024)
            if file_size_mb > 15:  # Skip very large files
                return None
            
            # Load and validate
            sequence = np.load(seq_file)
            if sequence.shape[0] != self.sequence_length:
                return None
            if len(sequence.shape) != 4:
                return None
            
            # Ensure proper data type and range
            sequence = sequence.astype(np.float32)
            sequence = np.clip(sequence, 0.0, 1.0)
            
            return sequence
            
        except Exception:
            return None
    
    def _safe_load_ground_truth(self, gt_file):
        """Safely load ground truth"""
        try:
            if not gt_file.exists():
                return None
            
            ground_truth = np.load(gt_file)
            if ground_truth.shape[0] != self.sequence_length:
                return None
            
            ground_truth = ground_truth.astype(np.float32)
            ground_truth = np.clip(ground_truth, 0.0, 1.0)
            
            return ground_truth
            
        except Exception:
            return None
    
    def _create_synthetic_data(self, count):
        """Create synthetic data for consistent training"""
        for i in range(count):
            # Create deterministic sequence
            np.random.seed(i)  # Deterministic for consistency
            sequence = np.random.rand(self.sequence_length, 224, 224, 3).astype(np.float32)
            
            # Create ground truth with deterministic pothole presence
            ground_truth = np.zeros((self.sequence_length, 224, 224), dtype=np.float32)
            if i % 3 == 0:  # 33% have potholes
                ground_truth[:, 80:120, 80:120] = 1.0
            
            self.episode_sequences.append(sequence)
            self.episode_ground_truths.append(ground_truth)
            self.episode_metadata.append({
                "video_name": f"synthetic_{i:04d}",
                "sequence_idx": "000",
                "data_type": "synthetic",
                "has_ground_truth": True
            })
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        if len(self.episode_sequences) > 0:
            episode_idx = random.randint(0, len(self.episode_sequences) - 1)
            self.current_sequence = self.episode_sequences[episode_idx]
            self.current_ground_truth = self.episode_ground_truths[episode_idx]
            self.current_metadata = self.episode_metadata[episode_idx]
        else:
            # Emergency fallback
            self.current_sequence = np.random.rand(self.sequence_length, 224, 224, 3).astype(np.float32)
            self.current_ground_truth = np.zeros((self.sequence_length, 224, 224), dtype=np.float32)
            self.current_metadata = {"video_name": "emergency", "has_ground_truth": False}
        
        self.episode_count += 1
        
        observation = self.current_sequence.astype(np.float32)
        info = {
            "episode": self.episode_count,
            "metadata": self.current_metadata,
            "has_ground_truth": self.current_ground_truth is not None
        }
        
        return observation, info
    
    def step(self, action):
        """Execute action and return results"""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        
        confidence_threshold = self.action_thresholds[action]
        detection_confidence = self._simulate_detection_confidence()
        agent_detects_pothole = detection_confidence > confidence_threshold
        reward = self._calculate_reward(agent_detects_pothole)
        
        info = {
            "action": action,
            "confidence_threshold": confidence_threshold,
            "detection_confidence": detection_confidence,
            "agent_decision": agent_detects_pothole,
            "reward": reward,
            "ground_truth_has_pothole": self._ground_truth_has_pothole(),
            "metadata": self.current_metadata
        }
        
        return self.current_sequence.astype(np.float32), reward, True, False, info
    
    def _simulate_detection_confidence(self):
        """Simulate CNN detection confidence"""
        if self.current_ground_truth is not None:
            has_pothole = self._ground_truth_has_pothole()
            
            if has_pothole:
                # Higher confidence for actual potholes
                base_confidence = random.uniform(0.6, 0.9)
                noise = random.uniform(-0.1, 0.1)
                return np.clip(base_confidence + noise, 0.0, 1.0)
            else:
                # Lower confidence for non-potholes
                base_confidence = random.uniform(0.1, 0.4)
                noise = random.uniform(-0.1, 0.1)
                return np.clip(base_confidence + noise, 0.0, 1.0)
        else:
            return random.uniform(0.2, 0.8)
    
    def _ground_truth_has_pothole(self):
        """Check if ground truth indicates pothole presence"""
        if self.current_ground_truth is None:
            return False
        
        pothole_pixels = np.sum(self.current_ground_truth > 0)
        total_pixels = self.current_ground_truth.size
        pothole_ratio = pothole_pixels / total_pixels if total_pixels > 0 else 0
        
        return pothole_ratio > 0.01
    
    def _calculate_reward(self, agent_detects_pothole):
        """Calculate reward based on detection accuracy"""
        if self.current_ground_truth is None:
            return 0
        
        ground_truth_has_pothole = self._ground_truth_has_pothole()
        
        if ground_truth_has_pothole and agent_detects_pothole:
            self.total_correct_detections += 1
            return self.reward_correct
        elif not ground_truth_has_pothole and not agent_detects_pothole:
            self.total_correct_detections += 1
            return self.reward_correct
        elif not ground_truth_has_pothole and agent_detects_pothole:
            self.total_false_positives += 1
            return self.reward_false_positive
        else:
            self.total_missed_detections += 1
            return self.reward_missed
    
    def close(self):
        """Clean up resources"""
        try:
            import cv2
            cv2.destroyAllWindows()
        except:
            pass
        
        self.episode_sequences = []
        self.episode_ground_truths = []
        self.episode_metadata = []
        gc.collect()
    
    def get_performance_stats(self):
        """Get performance statistics"""
        total_decisions = self.total_correct_detections + self.total_false_positives + self.total_missed_detections
        
        if total_decisions == 0:
            return {"message": "No decisions made yet"}
        
        accuracy = self.total_correct_detections / total_decisions * 100
        
        return {
            "total_episodes": self.episode_count,
            "total_decisions": total_decisions,
            "correct_detections": self.total_correct_detections,
            "false_positives": self.total_false_positives,
            "missed_detections": self.total_missed_detections,
            "accuracy": round(accuracy, 2)
        }
