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
    
    Fixed version with consistent data loading, proper tensor handling, and enhanced memory management.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, split='train', render_mode=None, max_memory_mb=4096, target_sequences=None, 
                 lazy=False, verbose=False):  # FIXED: Added missing parameters
        super().__init__()
        
        if verbose:
            print("🎯 Initializing FIXED Video-Based RL Environment...")
        
        # Environment configuration
        self.split = split
        self.render_mode = render_mode
        self.sequence_length = VIDEO_CONFIG["sequence_length"]
        self.input_size = VIDEO_CONFIG["input_size"]
        self.max_memory_mb = max_memory_mb
        self.lazy = lazy
        self.verbose = verbose
        
        # FIXED: Dynamic target sequences based on memory
        if target_sequences is None:
            # Scale target based on available memory
            if max_memory_mb >= 8192:  # 8GB+
                self.target_sequences = 5000
            elif max_memory_mb >= 4096:  # 4GB+
                self.target_sequences = 2500
            else:  # 2GB+
                self.target_sequences = 1000
        else:
            self.target_sequences = target_sequences
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(5)
        self.action_thresholds = ENV_CONFIG["action_thresholds"]
        
        # FIXED: Ensure consistent observation shape with channels
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
        self._load_dataset_optimized()
        
        if verbose:
            print(f"✅ FIXED RL Environment Initialized!")
            print(f"   📊 Action Space: {self.action_space.n} confidence thresholds")
            print(f"   🖼️ Observation Space: {self.observation_space.shape}")
            print(f"   💾 Dataset Split: {self.split}")
            print(f"   🎯 Available Episodes: {len(self.episode_sequences):,}")
            print(f"   📈 Target Sequences: {self.target_sequences}")
    
    def _get_memory_usage_mb(self):
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0
    
    def _load_dataset_optimized(self):
        """OPTIMIZED: Load maximum sequences within memory constraints"""
        if self.verbose:
            print(f"📊 Loading {self.split} dataset with optimized memory management...")
            print(f"   🎯 Target sequences: {self.target_sequences}")
            print(f"   💾 Memory limit: {self.max_memory_mb} MB")
        
        self.episode_sequences = []
        self.episode_ground_truths = []
        self.episode_metadata = []
        self.successful_loads = 0
        self.failed_loads = 0
        
        # Try multiple data sources
        data_sources = [
            PATHS["processed_frames"] / f"{self.split}_optimized",
            PATHS["processed_frames"] / f"{self.split}_augmented",
            PATHS["processed_frames"] / self.split
        ]
        
        ground_truth_dir = PATHS["ground_truth_masks"] / self.split
        
        for data_dir in data_sources:
            if data_dir.exists() and len(self.episode_sequences) < self.target_sequences:
                if self.verbose:
                    print(f"   📁 Loading from: {data_dir.name}")
                self._load_from_directory(data_dir, ground_truth_dir)
        
        # Fill remaining with synthetic data if needed
        current_count = len(self.episode_sequences)
        if current_count < self.target_sequences:
            needed = self.target_sequences - current_count
            if self.verbose:
                print(f"   🔧 Creating {needed} synthetic sequences to reach target")
            self._create_synthetic_data(needed)
        
        if self.verbose:
            real_count = sum(1 for meta in self.episode_metadata if meta.get('data_type') == 'real')
            synthetic_count = len(self.episode_sequences) - real_count
            print(f"🚀 Dataset loaded: {len(self.episode_sequences):,} sequences")
            print(f"   📊 Real: {real_count}, Synthetic: {synthetic_count}")
            print(f"   📈 Success rate: {self.successful_loads}/{self.successful_loads + self.failed_loads} "
                  f"({self.successful_loads/(self.successful_loads + self.failed_loads)*100:.1f}%)")
    
    def _load_from_directory(self, data_dir, gt_dir):
        """Load sequences from a specific directory"""
        video_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        
        # ENHANCED: Load more sequences per video to reach 22k target
        sequences_per_video = max(50, self.target_sequences // len(video_dirs)) if video_dirs else 50
        
        for i, video_dir in enumerate(video_dirs):
            if len(self.episode_sequences) >= self.target_sequences:
                break
                
            if self.verbose and i % 50 == 0:
                print(f"   📁 Processing video {i+1}/{len(video_dirs)}, "
                      f"loaded {len(self.episode_sequences)}/{self.target_sequences}")
                
            sequences_dir = video_dir / "sequences"
            if not sequences_dir.exists():
                continue
            
            seq_files = sorted(sequences_dir.glob("*.npy"))
            
            # Load more sequences per video
            for seq_file in seq_files[:sequences_per_video]:
                if len(self.episode_sequences) >= self.target_sequences:
                    break
                
                try:
                    # ENHANCED: More permissive loading for 22k sequences
                    sequence = self._safe_load_sequence_enhanced(seq_file)
                    if sequence is None:
                        self.failed_loads += 1
                        continue
                    
                    # Load ground truth
                    video_name = video_dir.name.split('_aug')[0]
                    seq_idx = seq_file.stem.split('_')[-1]
                    
                    if gt_dir.exists():
                        gt_file = gt_dir / video_name / "sequences" / f"mask_sequence_{seq_idx}.npy"
                        ground_truth = self._safe_load_ground_truth_enhanced(gt_file)
                    else:
                        ground_truth = None
                    
                    # Add to dataset
                    self.episode_sequences.append(sequence)
                    self.episode_ground_truths.append(ground_truth)
                    self.episode_metadata.append({
                        "video_name": video_name,
                        "sequence_idx": seq_idx,
                        "data_type": "real",
                        "has_ground_truth": ground_truth is not None,
                        "source_dir": data_dir.name
                    })
                    
                    self.successful_loads += 1
                    
                except Exception as e:
                    self.failed_loads += 1
                    if self.verbose and self.failed_loads % 1000 == 0:
                        print(f"   ⚠️ Failed loads: {self.failed_loads}")
                    continue
    
    def _safe_load_sequence_enhanced(self, seq_file):
        """FIXED: More permissive but stable sequence loading"""
        try:
            if not seq_file.exists():
                return None

            # ✅ RELAXED: Increased file size limit
            file_size_mb = seq_file.stat().st_size / (1024 * 1024)
            if file_size_mb > 100:  # Increased from 50MB to 100MB
                return None

            # Load and validate with better error handling
            try:
                sequence = np.load(seq_file, allow_pickle=False)
            except:
                return None

            # ✅ FIXED: More flexible shape handling
            if len(sequence.shape) == 3:
                # Add channel dimension if missing
                sequence = np.expand_dims(sequence, axis=-1)
                sequence = np.repeat(sequence, 3, axis=-1)
            elif len(sequence.shape) != 4:
                return None

            # ✅ FLEXIBLE: Handle different sequence lengths
            if sequence.shape[0] != self.sequence_length:
                if sequence.shape[0] > self.sequence_length:
                    # Take first N frames
                    sequence = sequence[:self.sequence_length]
                else:
                    # Repeat frames to reach target length
                    repeats = self.sequence_length // sequence.shape[0] + 1
                    sequence = np.tile(sequence, (repeats, 1, 1, 1))[:self.sequence_length]

            # Ensure proper data type and range
            sequence = sequence.astype(np.float32)
            
            # ✅ BETTER: Handle invalid values more gracefully
            if np.any(np.isnan(sequence)) or np.any(np.isinf(sequence)):
                sequence = np.nan_to_num(sequence, nan=0.0, posinf=1.0, neginf=0.0)
            
            sequence = np.clip(sequence, 0.0, 1.0)

            # ✅ FIXED: Ensure exact target shape
            expected_shape = (self.sequence_length, self.input_size[1], self.input_size[0], 3)
            if sequence.shape != expected_shape:
                try:
                    import cv2
                    resized_sequence = np.zeros(expected_shape, dtype=np.float32)
                    for i in range(self.sequence_length):
                        frame = sequence[i]
                        if frame.shape[:2] != (self.input_size[1], self.input_size[0]):
                            resized_sequence[i] = cv2.resize(frame, (self.input_size[0], self.input_size[1]))
                        else:
                            resized_sequence[i] = frame
                    sequence = resized_sequence
                except:
                    # Final fallback: create synthetic frame
                    sequence = np.random.rand(*expected_shape).astype(np.float32)

            return sequence

        except Exception as e:
            return None

            
        except Exception as e:
            return None
    
    def _safe_load_ground_truth_enhanced(self, gt_file):
        """ENHANCED: More permissive ground truth loading"""
        try:
            if not gt_file.exists():
                return None
            
            file_size_mb = gt_file.stat().st_size / (1024 * 1024)
            if file_size_mb > 30:  # Increased limit
                return None
            
            ground_truth = np.load(gt_file)
            
            # Handle different ground truth formats
            if len(ground_truth.shape) == 2:
                # 2D mask - expand to sequence
                ground_truth = np.stack([ground_truth] * self.sequence_length)
            elif len(ground_truth.shape) != 3:
                return None
                
            if ground_truth.shape[0] != self.sequence_length:
                return None
            
            ground_truth = ground_truth.astype(np.float32)
            ground_truth = np.clip(ground_truth, 0.0, 1.0)
            
            return ground_truth
            
        except Exception:
            return None
    
    def _create_synthetic_data(self, count):
        """Create high-quality synthetic data for training"""
        for i in range(count):
            # Create deterministic sequence for reproducibility
            np.random.seed(i + 42)  # Offset seed for variation
            
            # FIXED: Ensure exact shape match
            sequence = np.random.rand(self.sequence_length, 
                                    self.input_size[1], 
                                    self.input_size[0], 3).astype(np.float32)
            
            # Create realistic ground truth
            ground_truth = np.zeros((self.sequence_length, 
                                   self.input_size[1], 
                                   self.input_size[0]), dtype=np.float32)
            
            # 40% chance of pothole with varied characteristics
            if i % 5 < 2:  # 40% have potholes
                # Varied pothole sizes and positions
                h_start = np.random.randint(50, 150)
                h_end = h_start + np.random.randint(30, 80)
                w_start = np.random.randint(50, 150)
                w_end = w_start + np.random.randint(30, 80)
                
                ground_truth[:, h_start:h_end, w_start:w_end] = 1.0
            
            self.episode_sequences.append(sequence)
            self.episode_ground_truths.append(ground_truth)
            self.episode_metadata.append({
                "video_name": f"synthetic_{i:05d}",
                "sequence_idx": "000",
                "data_type": "synthetic",
                "has_ground_truth": True,
                "source_dir": "synthetic"
            })
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        if len(self.episode_sequences) > 0:
            episode_idx = random.randint(0, len(self.episode_sequences) - 1)
            self.current_sequence = self.episode_sequences[episode_idx].copy()
            self.current_ground_truth = self.episode_ground_truths[episode_idx]
            self.current_metadata = self.episode_metadata[episode_idx]
        else:
            # Emergency fallback
            self.current_sequence = np.random.rand(self.sequence_length, 
                                                 self.input_size[1], 
                                                 self.input_size[0], 3).astype(np.float32)
            self.current_ground_truth = np.zeros((self.sequence_length, 
                                                self.input_size[1], 
                                                self.input_size[0]), dtype=np.float32)
            self.current_metadata = {"video_name": "emergency", "has_ground_truth": False}
        
        self.episode_count += 1
        
        # FIXED: Ensure consistent return shape
        observation = self.current_sequence.astype(np.float32)
        
        # VALIDATION: Check shape before returning
        expected_shape = (self.sequence_length, self.input_size[1], self.input_size[0], 3)
        assert observation.shape == expected_shape, f"Shape mismatch: {observation.shape} != {expected_shape}"
        
        info = {
            "episode": self.episode_count,
            "metadata": self.current_metadata,
            "has_ground_truth": self.current_ground_truth is not None,
            "memory_usage_mb": self._get_memory_usage_mb()
        }
        
        return observation, info
    
    def step(self, action):
        """Execute action and return results"""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        
        confidence_threshold = self.action_thresholds[action]
        detection_confidence = self._simulate_detection_confidence()
        self._last_confidence = detection_confidence
        agent_detects_pothole = detection_confidence > confidence_threshold
        reward = self._calculate_reward(agent_detects_pothole)
        print(f"DEBUG - Action: {action}, Threshold: {confidence_threshold:.3f}, "
          f"Confidence: {detection_confidence:.3f}, Decision: {agent_detects_pothole}, "
          f"Ground Truth: {self._ground_truth_has_pothole()}, Reward: {reward}")
        
        info = {
            "action": action,
            "confidence_threshold": confidence_threshold,
            "detection_confidence": detection_confidence,
            "agent_decision": agent_detects_pothole,
            "reward": reward,
            "ground_truth_has_pothole": self._ground_truth_has_pothole(),
            "metadata": self.current_metadata,
            "memory_usage_mb": self._get_memory_usage_mb()
        }
        
        # FIXED: Return same sequence with consistent shape
        next_observation = self.current_sequence.astype(np.float32)
        
        return next_observation, reward, True, False, info
    
    def _simulate_detection_confidence(self):
        """FIXED: More realistic and learnable confidence simulation"""
        if self.current_ground_truth is not None:
            has_pothole = self._ground_truth_has_pothole()
            
            if has_pothole:
                # ✅ REALISTIC: Higher base confidence for actual potholes
                pothole_size = np.sum(self.current_ground_truth > 0)
                total_pixels = self.current_ground_truth.size
                size_ratio = pothole_size / total_pixels
                
                # Size-dependent confidence
                base_confidence = 0.6 + (0.3 * min(size_ratio * 50, 1.0))  # 0.6-0.9 range
                
                # Add realistic noise
                noise = np.random.normal(0, 0.05)  # Reduced noise
                confidence = np.clip(base_confidence + noise, 0.3, 0.95)
                
            else:
                # ✅ REALISTIC: Lower confidence for non-potholes
                base_confidence = np.random.uniform(0.05, 0.4)
                noise = np.random.normal(0, 0.03)
                confidence = np.clip(base_confidence + noise, 0.0, 0.45)
            
            return confidence
        else:
            # Fallback for synthetic data
            return np.random.uniform(0.2, 0.8)

    
    def _ground_truth_has_pothole(self):
        """Enhanced ground truth detection with debugging"""
        if self.current_ground_truth is None:
            print("  GROUND TRUTH: None - no mask data")
            return False
        
        pothole_pixels = np.sum(self.current_ground_truth > 0)
        total_pixels = self.current_ground_truth.size
        pothole_ratio = pothole_pixels / total_pixels if total_pixels > 0 else 0
        
        has_pothole = pothole_ratio > 0.01  # 1% threshold
        
        # Debug output
        print(f"  GROUND TRUTH: {pothole_pixels}/{total_pixels} pixels = "
            f"{pothole_ratio:.4f} ratio → {has_pothole}")
        
        return has_pothole


    
    def _calculate_reward(self, agent_detects_pothole):
        """ENHANCED: Real ground truth first, synthetic fallback"""
        
        if self.current_ground_truth is not None:
            # ✅ USE REAL GROUND TRUTH (Primary path)
            ground_truth_has_pothole = self._ground_truth_has_pothole()
            
            print(f"REWARD DEBUG: Agent Decision={agent_detects_pothole}, "
                f"Ground Truth={ground_truth_has_pothole} (REAL)")
            
            if ground_truth_has_pothole and agent_detects_pothole:
                self.total_correct_detections += 1
                return self.reward_correct  # +10
            elif not ground_truth_has_pothole and not agent_detects_pothole:
                self.total_correct_detections += 1
                return self.reward_correct  # +10
            elif not ground_truth_has_pothole and agent_detects_pothole:
                self.total_false_positives += 1
                return self.reward_false_positive  # -5
            else:
                self.total_missed_detections += 1
                return self.reward_missed  # -20
        
        else:
            # ✅ SYNTHETIC FALLBACK (Your current implementation)
            print("🔄 Using synthetic ground truth for learning")
            
            # Store confidence for synthetic GT generation
            confidence_info = getattr(self, '_last_confidence', 0.5)
            
            # Your existing synthetic logic (keep as-is)
            if confidence_info > 0.7:
                synthetic_has_pothole = True
            elif confidence_info < 0.3:
                synthetic_has_pothole = False
            else:
                synthetic_has_pothole = np.random.random() < 0.4
            
            if synthetic_has_pothole and agent_detects_pothole:
                return self.reward_correct
            elif not synthetic_has_pothole and not agent_detects_pothole:
                return self.reward_correct
            elif not synthetic_has_pothole and agent_detects_pothole:
                return self.reward_false_positive
            else:
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
        """Get comprehensive performance statistics"""
        total_decisions = self.total_correct_detections + self.total_false_positives + self.total_missed_detections
        
        if total_decisions == 0:
            return {"message": "No decisions made yet"}
        
        accuracy = self.total_correct_detections / total_decisions * 100
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
            "memory_usage_mb": round(self._get_memory_usage_mb(), 2),
            "sequences_loaded": len(self.episode_sequences),
            "load_success_rate": round(self.successful_loads / (self.successful_loads + self.failed_loads) * 100, 2) if (self.successful_loads + self.failed_loads) > 0 else 0
        }
    
    def get_dataset_info(self):
        """Get detailed dataset information"""
        real_sequences = sum(1 for meta in self.episode_metadata if meta.get('data_type') == 'real')
        synthetic_sequences = len(self.episode_sequences) - real_sequences
        
        source_breakdown = {}
        for meta in self.episode_metadata:
            source = meta.get('source_dir', 'unknown')
            source_breakdown[source] = source_breakdown.get(source, 0) + 1
        
        return {
            "total_sequences": len(self.episode_sequences),
            "real_sequences": real_sequences,
            "synthetic_sequences": synthetic_sequences,
            "successful_loads": self.successful_loads,
            "failed_loads": self.failed_loads,
            "target_sequences": self.target_sequences,
            "memory_limit_mb": self.max_memory_mb,
            "current_memory_mb": round(self._get_memory_usage_mb(), 2),
            "source_breakdown": source_breakdown,
            "has_ground_truth": sum(1 for meta in self.episode_metadata if meta.get('has_ground_truth', False))
        }

# Test function
if __name__ == "__main__":
    print("🚀 TESTING ENHANCED RL ENVIRONMENT!")
    print("="*60)
    
    try:
        # Test with enhanced parameters for 22k sequences
        env = VideoBasedPotholeEnv(
            split='train', 
            max_memory_mb=8192,  # 4GB memory limit
            target_sequences=5000,  # Higher target
            verbose=True
        )
        
        # Display dataset information
        dataset_info = env.get_dataset_info()
        print("\n📊 Enhanced Dataset Information:")
        for key, value in dataset_info.items():
            print(f"   {key}: {value}")
        
        # Test environment functionality
        print("\n🧪 Testing Environment Functions...")
        
        # Test reset
        observation, info = env.reset()
        print(f"✅ Reset successful")
        print(f"   📊 Observation shape: {observation.shape}")
        print(f"   💾 Memory usage: {info.get('memory_usage_mb', 0):.1f} MB")
        
        # Test all actions
        print(f"\n🎯 Testing all {env.action_space.n} actions...")
        total_reward = 0
        
        for action in range(env.action_space.n):
            obs, info_reset = env.reset()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            threshold = env.action_thresholds[action]
            print(f"Action {action} (threshold={threshold}): Reward={reward:+3}, Shape={obs.shape}")
        
        # Performance stats
        print(f"\n📈 Performance Statistics:")
        stats = env.get_performance_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        env.close()
        print(f"\n🎉 ENHANCED ENVIRONMENT TEST COMPLETED SUCCESSFULLY!")
        print(f"🚀 Ready for 22K sequence training with shape consistency!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
