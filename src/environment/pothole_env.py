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
    
    This custom OpenAI Gym environment enables RL agents to learn optimal
    pothole detection strategies using temporal video sequences.
    
    Innovation: First RL environment to use video sequences for detection threshold optimization!
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, split='train', render_mode=None, max_memory_mb=2048, deterministic_loading=True):
        super().__init__()
        
        print("🎯 Initializing Revolutionary Video-Based RL Environment...")
        
        # Environment configuration
        self.split = split
        self.render_mode = render_mode
        self.sequence_length = VIDEO_CONFIG["sequence_length"]  # 5 frames
        self.input_size = VIDEO_CONFIG["input_size"]            # (224, 224)
        self.max_memory_mb = max_memory_mb  # Memory limit in MB
        self.deterministic_loading = deterministic_loading  # FIXED: Consistent loading
        
        # 🎯 ACTION SPACE: 5 Confidence Thresholds (Your Innovation!)
        self.action_space = spaces.Discrete(5)
        self.action_thresholds = ENV_CONFIG["action_thresholds"]  # [0.3, 0.5, 0.7, 0.8, 0.9]
        
        # 🖼️ OBSERVATION SPACE: Video Sequence (5 consecutive frames)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.sequence_length, self.input_size[1], self.input_size[0], 3),
            dtype=np.float32
        )
        
        # 🏆 REWARD SYSTEM: Based on Detection Accuracy
        self.reward_correct = ENV_CONFIG["reward_correct"]      # +10
        self.reward_false_positive = ENV_CONFIG["reward_false_positive"]  # -5
        self.reward_missed = ENV_CONFIG["reward_missed"]        # -20
        
        # 📊 Environment State
        self.current_sequence = None
        self.current_ground_truth = None
        self.sequence_index = 0
        self.episode_sequences = []
        self.episode_ground_truths = []
        self.episode_metadata = []
        
        # 📈 Performance Tracking
        self.episode_rewards = []
        self.episode_actions = []
        self.total_correct_detections = 0
        self.total_false_positives = 0
        self.total_missed_detections = 0
        self.episode_count = 0
        self.failed_loads = 0
        self.successful_loads = 0
        
        # FIXED: Deterministic sequence loading counter
        self.target_sequence_count = 1000  # Fixed target for consistency
        self.loaded_sequence_count = 0
        
        # 🎮 Load Dataset with memory management
        self._load_dataset()
        
        print(f"✅ RL Environment Initialized!")
        print(f"   📊 Action Space: {self.action_space.n} confidence thresholds")
        print(f"   🖼️ Observation Space: {self.observation_space.shape}")
        print(f"   💾 Dataset Split: {self.split}")
        print(f"   🎯 Available Episodes: {len(self.episode_sequences):,}")
        print(f"   📈 Load Success Rate: {self.successful_loads}/{self.successful_loads + self.failed_loads}")
        if self.failed_loads > 0:
            print(f"   ⚠️  Skipped {self.failed_loads} sequences due to memory/corruption issues")
    
    def _get_memory_usage_mb(self):
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0  # Fallback if psutil fails
    
    def _safe_load_sequence(self, seq_file):
        """Safely load numpy sequence with memory and error checking"""
        try:
            # Check file size first
            file_size_mb = seq_file.stat().st_size / (1024 * 1024)
            if file_size_mb > 20:  # FIXED: Reduced threshold from 50MB to 20MB
                return None
            
            # FIXED: More conservative memory management
            current_memory = self._get_memory_usage_mb()
            if current_memory > (self.max_memory_mb * 0.8):  # Use 80% threshold
                gc.collect()  # Force garbage collection
                current_memory = self._get_memory_usage_mb()
                if current_memory > self.max_memory_mb:
                    return None
            
            # Use memory mapping first to check array properties
            try:
                mapped_array = np.load(seq_file, mmap_mode='r')
                
                # Check array size and shape
                if mapped_array.size > 5 * 1024 * 1024:  # FIXED: Reduced from 10M to 5M elements
                    return None
                if len(mapped_array.shape) != 4:  # Should be 4D: (frames, height, width, channels)
                    return None
                if mapped_array.shape[0] != self.sequence_length:
                    return None
                
                # Load the actual array if checks pass
                sequence = np.array(mapped_array, dtype=np.float32)
                del mapped_array
                
                # Validate data ranges
                if np.any(np.isnan(sequence)) or np.any(np.isinf(sequence)):
                    return None
                
                # Ensure proper range [0, 1]
                if sequence.max() > 1.0 or sequence.min() < 0.0:
                    sequence = np.clip(sequence, 0.0, 1.0)
                
                return sequence
                
            except (ValueError, OSError, MemoryError) as e:
                return None
                
        except Exception as e:
            return None

    def _safe_load_ground_truth(self, gt_file):
        """Safely load ground truth mask with validation"""
        try:
            # Check file exists and size
            if not gt_file.exists():
                return None
            
            file_size_mb = gt_file.stat().st_size / (1024 * 1024)
            if file_size_mb > 10:  # FIXED: Reduced from 20MB to 10MB
                return None
            
            # Load with memory mapping
            mapped_gt = np.load(gt_file, mmap_mode='r')
            
            # Validate shape
            if len(mapped_gt.shape) != 3:  # Should be 3D: (frames, height, width)
                return None
            if mapped_gt.shape[0] != self.sequence_length:
                return None
            
            # Load actual array
            ground_truth = np.array(mapped_gt, dtype=np.float32)
            del mapped_gt
            
            # Ensure binary mask values
            ground_truth = np.clip(ground_truth, 0.0, 1.0)
            
            return ground_truth
            
        except Exception as e:
            return None
    
    def _load_dataset(self):
        """🔥 Load dataset with FIXED consistent loading for all configurations!"""
        print(f"📊 Loading {self.split} dataset with deterministic optimization...")
        
        # Load optimized and augmented sequences
        optimized_dir = PATHS["processed_frames"] / f"{self.split}_optimized"
        augmented_dir = PATHS["processed_frames"] / f"{self.split}_augmented"
        ground_truth_dir = PATHS["ground_truth"] / self.split
        
        self.episode_sequences = []
        self.episode_ground_truths = []
        self.episode_metadata = []
        self.failed_loads = 0
        self.successful_loads = 0
        self.loaded_sequence_count = 0
        
        # FIXED: Deterministic loading order
        all_sequence_files = []
        
        # Collect all sequence files in deterministic order
        if optimized_dir.exists():
            print("   📁 Collecting optimized sequences...")
            optimized_files = self._collect_sequence_files(optimized_dir, "optimized")
            all_sequence_files.extend(optimized_files)
        
        if augmented_dir.exists():
            print("   📁 Collecting augmented sequences...")
            augmented_files = self._collect_sequence_files(augmented_dir, "augmented")
            all_sequence_files.extend(augmented_files)
        
        # FIXED: Sort files for deterministic loading
        all_sequence_files.sort(key=lambda x: (x[0], x[1]))  # Sort by path, then filename
        
        # FIXED: Load exactly target_sequence_count sequences for consistency
        print(f"   🎯 Loading exactly {self.target_sequence_count} sequences for consistency...")
        
        for seq_info in all_sequence_files[:self.target_sequence_count]:
            if self._load_single_sequence(seq_info, ground_truth_dir):
                self.loaded_sequence_count += 1
                if self.loaded_sequence_count >= self.target_sequence_count:
                    break
        
        # Force garbage collection after loading
        gc.collect()
        
        print(f"🚀 Dataset loaded: {len(self.episode_sequences):,} sequences ready for RL training!")
        print(f"   📊 Target: {self.target_sequence_count}, Loaded: {self.loaded_sequence_count}")
        
        if len(self.episode_sequences) == 0:
            print("⚠️ Warning: No sequences found! Creating fallback data...")
            self._create_fallback_data()
    
    def _collect_sequence_files(self, sequence_dir, data_type):
        """Collect all sequence files in deterministic order"""
        sequence_files = []
        video_dirs = sorted([d for d in sequence_dir.iterdir() if d.is_dir()])
        
        for video_dir in video_dirs:
            sequences_dir = video_dir / "sequences"
            if not sequences_dir.exists():
                continue
            
            # Get all sequence files and sort them
            seq_files = sorted(sequences_dir.glob("*.npy"))
            for seq_file in seq_files:
                sequence_files.append((seq_file, data_type, video_dir.name))
        
        return sequence_files
    
    def _load_single_sequence(self, seq_info, ground_truth_dir):
        """Load a single sequence with ground truth"""
        seq_file, data_type, video_dir_name = seq_info
        
        try:
            # Safely load video sequence
            sequence = self._safe_load_sequence(seq_file)
            if sequence is None:
                self.failed_loads += 1
                return False
            
            # Get video name (remove augmentation suffix if present)
            video_name = video_dir_name.split('_aug')[0]
            
            # Find and load corresponding ground truth
            gt_video_dir = ground_truth_dir / video_name
            ground_truth = None
            metadata = {}
            
            if gt_video_dir.exists():
                gt_sequences_dir = gt_video_dir / "sequences"
                
                # Find matching ground truth file
                seq_idx = seq_file.stem.split('_')[-1]
                if 'aug' in seq_idx:
                    seq_idx = seq_idx.split('aug')[0]
                
                gt_file = gt_sequences_dir / f"mask_sequence_{seq_idx}.npy"
                ground_truth = self._safe_load_ground_truth(gt_file)
                
                # Load metadata
                metadata_file = gt_video_dir / "sequence_info.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            all_metadata = json.load(f)
                            seq_idx_int = int(seq_idx) if seq_idx.isdigit() else 0
                            if seq_idx_int < len(all_metadata):
                                metadata = all_metadata[seq_idx_int]
                    except (json.JSONDecodeError, ValueError, IndexError):
                        metadata = {}
            
            # Add to dataset
            self.episode_sequences.append(sequence)
            self.episode_ground_truths.append(ground_truth)
            self.episode_metadata.append({
                "video_name": video_name,
                "sequence_idx": seq_idx,
                "data_type": data_type,
                "has_ground_truth": ground_truth is not None,
                **metadata
            })
            
            self.successful_loads += 1
            return True
                    
        except Exception as e:
            self.failed_loads += 1
            return False
    
    def _create_fallback_data(self):
        """Create simple, reliable fallback data"""
        print("🔧 Creating simplified fallback test data...")
        
        # Define target sequence count if not already defined
        if not hasattr(self, 'target_sequence_count'):
            self.target_sequence_count = 1000  # Default target
        
        # Create only a reasonable number of sequences
        num_sequences = min(100, self.target_sequence_count)
        
        for i in range(num_sequences):
            # Create simple, valid video sequence
            sequence = np.random.rand(self.sequence_length, 224, 224, 3).astype(np.float32)
            sequence = np.clip(sequence, 0.0, 1.0)  # Ensure valid range
            
            # Create simple ground truth
            ground_truth = np.zeros((self.sequence_length, 224, 224), dtype=np.float32)
            if i % 2 == 0:  # Every other sequence has a pothole
                ground_truth[:, 80:120, 80:120] = 1.0  # Simple square pothole
            
            # Add to dataset with proper validation
            if sequence.shape == (self.sequence_length, 224, 224, 3):
                self.episode_sequences.append(sequence)
                self.episode_ground_truths.append(ground_truth)
                self.episode_metadata.append({
                    "video_name": f"fallback_{i:03d}",
                    "sequence_idx": f"{i:03d}",
                    "data_type": "fallback",
                    "has_ground_truth": True
                })
        
        print(f"✅ Created {len(self.episode_sequences)} fallback sequences")

    
    def reset(self, seed=None, options=None):
        """🔄 Reset environment for new episode"""
        super().reset(seed=seed)
        
        # Select random sequence for this episode
        if len(self.episode_sequences) > 0:
            episode_idx = random.randint(0, len(self.episode_sequences) - 1)
            self.current_sequence = self.episode_sequences[episode_idx]
            self.current_ground_truth = self.episode_ground_truths[episode_idx]
            self.current_metadata = self.episode_metadata[episode_idx]
        else:
            # Create emergency fallback
            self.current_sequence = np.random.rand(self.sequence_length, 224, 224, 3).astype(np.float32)
            self.current_ground_truth = np.zeros((self.sequence_length, 224, 224), dtype=np.float32)
            self.current_metadata = {"video_name": "emergency", "has_ground_truth": False}
        
        # Reset episode tracking
        self.sequence_index = 0
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_count += 1
        
        # Return initial observation and info
        observation = self.current_sequence.astype(np.float32)
        info = {
            "episode": self.episode_count,
            "metadata": self.current_metadata,
            "has_ground_truth": self.current_ground_truth is not None,
            "memory_usage_mb": self._get_memory_usage_mb()
        }
        
        return observation, info
    
    def step(self, action):
        """🎯 Execute action and return next state, reward, done, info"""
        
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in range [0, {self.action_space.n-1}]")
        
        # Get confidence threshold for this action
        confidence_threshold = self.action_thresholds[action]
        
        # 🤖 SIMULATE CNN DETECTION (In real implementation, this would be your CNN)
        detection_confidence = self._simulate_detection_confidence()
        
        # Apply threshold to get binary detection result
        agent_detects_pothole = detection_confidence > confidence_threshold
        
        # 🏆 CALCULATE REWARD BASED ON GROUND TRUTH
        reward = self._calculate_reward(agent_detects_pothole, action)
        
        # Update tracking
        self.episode_rewards.append(reward)
        self.episode_actions.append(action)
        
        # Episode is done after one decision (can be extended for multi-step)
        done = True
        truncated = False
        
        # Generate comprehensive info
        info = {
            "action": action,
            "confidence_threshold": confidence_threshold,
            "detection_confidence": detection_confidence,
            "agent_decision": agent_detects_pothole,
            "reward": reward,
            "episode_total_reward": sum(self.episode_rewards),
            "ground_truth_has_pothole": self._ground_truth_has_pothole(),
            "metadata": self.current_metadata,
            "memory_usage_mb": self._get_memory_usage_mb()
        }
        
        # Next observation (same sequence for single-step)
        next_observation = self.current_sequence.astype(np.float32)
        
        return next_observation, reward, done, truncated, info
    
    def _simulate_detection_confidence(self):
        """🎲 Simulate CNN detection confidence (placeholder for real CNN)"""
        if self.current_ground_truth is not None:
            # Check if current sequence actually contains potholes
            has_pothole = self._ground_truth_has_pothole()
            
            if has_pothole:
                # Simulate higher confidence for actual potholes (with realistic noise)
                base_confidence = random.uniform(0.65, 0.88)
                noise = random.uniform(-0.12, 0.12)
                return np.clip(base_confidence + noise, 0.0, 1.0)
            else:
                # Simulate lower confidence for non-potholes (with realistic noise)
                base_confidence = random.uniform(0.15, 0.42)
                noise = random.uniform(-0.12, 0.12)
                return np.clip(base_confidence + noise, 0.0, 1.0)
        else:
            # No ground truth - random confidence with bias toward uncertainty
            return random.uniform(0.25, 0.75)
    
    def _ground_truth_has_pothole(self):
        """🎯 Check if ground truth indicates pothole presence"""
        if self.current_ground_truth is None:
            return False
        
        # Count non-zero pixels (pothole pixels) across all frames
        pothole_pixels = np.sum(self.current_ground_truth > 0)
        
        # Threshold: if >1% of pixels are potholes, consider it a pothole sequence
        total_pixels = self.current_ground_truth.size
        pothole_ratio = pothole_pixels / total_pixels if total_pixels > 0 else 0
        
        return pothole_ratio > 0.01  # Adjustable threshold
    
    def _calculate_reward(self, agent_detects_pothole, action):
        """🏆 Calculate reward based on detection accuracy"""
        if self.current_ground_truth is None:
            return 0  # No ground truth available
        
        ground_truth_has_pothole = self._ground_truth_has_pothole()
        
        # Reward calculation logic
        if ground_truth_has_pothole and agent_detects_pothole:
            # ✅ TRUE POSITIVE: Correctly detected pothole
            self.total_correct_detections += 1
            return self.reward_correct
        
        elif not ground_truth_has_pothole and not agent_detects_pothole:
            # ✅ TRUE NEGATIVE: Correctly identified no pothole
            self.total_correct_detections += 1
            return self.reward_correct
        
        elif not ground_truth_has_pothole and agent_detects_pothole:
            # ❌ FALSE POSITIVE: Incorrectly detected pothole
            self.total_false_positives += 1
            return self.reward_false_positive
        
        else:  # ground_truth_has_pothole and not agent_detects_pothole
            # ❌ FALSE NEGATIVE: Missed actual pothole (DANGEROUS!)
            self.total_missed_detections += 1
            return self.reward_missed
    
    def render(self):
        """🖼️ Render current state (optional visualization)"""
        if self.render_mode is None:
            return None
        
        if self.current_sequence is not None:
            # Display first frame of current sequence
            frame = self.current_sequence[0]  # Get first frame
            frame_uint8 = (frame * 255).astype(np.uint8)
            
            if self.render_mode == "human":
                try:
                    import cv2
                    cv2.imshow("RL Environment - Current Frame", frame_uint8)
                    cv2.waitKey(1)
                except Exception:
                    pass  # Ignore GUI errors in headless environments
                return None
            elif self.render_mode == "rgb_array":
                return frame_uint8
    
    def close(self):
        """🔒 Clean up resources"""
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass  # Ignore errors if OpenCV GUI not available
        
        # Clean up memory
        self.episode_sequences = []
        self.episode_ground_truths = []
        self.episode_metadata = []
        gc.collect()
    
    def get_performance_stats(self):
        """📊 Get comprehensive performance statistics"""
        total_decisions = self.total_correct_detections + self.total_false_positives + self.total_missed_detections
        
        if total_decisions == 0:
            return {"message": "No decisions made yet"}
        
        accuracy = self.total_correct_detections / total_decisions * 100 if total_decisions > 0 else 0
        precision = self.total_correct_detections / (self.total_correct_detections + self.total_false_positives) * 100 if (self.total_correct_detections + self.total_false_positives) > 0 else 0
        recall = self.total_correct_detections / (self.total_correct_detections + self.total_missed_detections) * 100 if (self.total_correct_detections + self.total_missed_detections) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "total_episodes": self.episode_count,
            "total_decisions": total_decisions,
            "correct_detections": self.total_correct_detections,
            "false_positives": self.total_false_positives,
            "missed_detections": self.total_missed_detections,
            "accuracy": round(accuracy, 2),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1_score": round(f1_score, 2),
            "average_episode_reward": np.mean([sum(self.episode_rewards)]) if self.episode_rewards else 0,
            "memory_usage_mb": round(self._get_memory_usage_mb(), 2),
            "data_load_success_rate": round(self.successful_loads / (self.successful_loads + self.failed_loads) * 100, 2) if (self.successful_loads + self.failed_loads) > 0 else 0
        }
    
    def get_dataset_info(self):
        """📈 Get dataset information"""
        return {
            "total_sequences": len(self.episode_sequences),
            "split": self.split,
            "successful_loads": self.successful_loads,
            "failed_loads": self.failed_loads,
            "has_ground_truth": sum(1 for gt in self.episode_ground_truths if gt is not None),
            "memory_limit_mb": self.max_memory_mb,
            "current_memory_mb": round(self._get_memory_usage_mb(), 2),
            "target_sequences": self.target_sequence_count,
            "loaded_sequences": self.loaded_sequence_count
        }

# 🧪 TEST ENVIRONMENT FUNCTIONALITY
if __name__ == "__main__":
    print("🚀 TESTING FIXED RL ENVIRONMENT WITH CONSISTENT LOADING!")
    print("="*70)
    
    # Create environment with fixed memory limit
    try:
        env = VideoBasedPotholeEnv(split='train', max_memory_mb=2048, deterministic_loading=True)
        
        # Display dataset information
        dataset_info = env.get_dataset_info()
        print("\n📊 Dataset Information:")
        for key, value in dataset_info.items():
            print(f"   {key}: {value}")
        
        # Test basic functionality
        print("\n🧪 Running Environment Tests...")
        
        # Test reset
        observation, info = env.reset()
        print(f"✅ Reset successful. Observation shape: {observation.shape}")
        print(f"📊 Episode info keys: {list(info.keys())}")
        print(f"💾 Memory usage: {info.get('memory_usage_mb', 0):.1f} MB")
        
        # Test multiple actions
        print(f"\n🎯 Testing all {env.action_space.n} actions...")
        rewards = []
        
        for action in range(env.action_space.n):
            obs, info_reset = env.reset()
            obs, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            
            threshold = env.action_thresholds[action]
            print(f"Action {action} (threshold={threshold}): Reward={reward:+3}, Done={done}")
            print(f"   Confidence: {info['detection_confidence']:.3f}, Decision: {info['agent_decision']}")
            print(f"   Ground truth: {info['ground_truth_has_pothole']}, Memory: {info['memory_usage_mb']:.1f}MB")
        
        # Test performance stats
        print(f"\n📈 Performance Statistics:")
        stats = env.get_performance_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        env.close()
        print(f"\n🎉 FIXED ENVIRONMENT TEST COMPLETED SUCCESSFULLY!")
        print(f"🚀 Consistent loading: {dataset_info['loaded_sequences']}/{dataset_info['target_sequences']} sequences!")
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        print("🔧 Consider checking data paths or reducing memory requirements")
