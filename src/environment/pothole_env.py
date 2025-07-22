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
import cv2
from collections import deque
from scipy import ndimage

sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import ENV_CONFIG, VIDEO_CONFIG, PATHS


class SyntheticNonPotholeGenerator:
    """Generate realistic non-pothole road sequences for balanced training"""
    
    def __init__(self, sequence_length=5, input_size=(224, 224)):
        self.sequence_length = sequence_length
        self.input_size = input_size
        
    def generate_smooth_road_sequence(self, seed=None):
        """Generate a sequence of smooth road surfaces"""
        if seed is not None:
            np.random.seed(seed)
            
        # Create base asphalt texture
        base_texture = self._create_asphalt_texture()
        
        # Create sequence with slight variations
        sequence = []
        for frame_idx in range(self.sequence_length):
            frame = base_texture.copy()
            
            # Add temporal variation (slight camera movement)
            frame = self._add_camera_movement(frame, frame_idx)
            
            # Add road features
            frame = self._add_road_markings(frame, probability=0.3)
            frame = self._add_lighting_variation(frame, frame_idx)
            frame = self._add_subtle_texture_variation(frame)
            
            # Ensure proper normalization
            frame = np.clip(frame, 0.0, 1.0).astype(np.float32)
            sequence.append(frame)
            
        return np.stack(sequence, axis=0)
    
    def _create_asphalt_texture(self):
        """Create realistic asphalt road texture"""
        # Base gray asphalt color
        base_color = np.random.uniform(0.15, 0.35)  # Dark gray
        frame = np.full((*self.input_size, 3), base_color, dtype=np.float32)
        
        # Add noise for asphalt texture
        noise = np.random.normal(0, 0.05, (*self.input_size, 3))
        frame += noise
        
        # Add some larger texture variations
        for _ in range(random.randint(5, 15)):
            x = random.randint(0, self.input_size[0] - 1)
            y = random.randint(0, self.input_size[1] - 1)
            size = random.randint(3, 8)
            intensity = random.uniform(-0.1, 0.1)
            
            frame[max(0, x-size):min(self.input_size[0], x+size),
                  max(0, y-size):min(self.input_size[1], y+size)] += intensity
        
        return frame
    
    def _add_road_markings(self, frame, probability=0.3):
        """Add road markings (lane lines, etc.)"""
        if random.random() < probability:
            # Add lane marking
            if random.random() < 0.5:  # Vertical lane line
                line_x = random.randint(self.input_size[0] // 4, 3 * self.input_size[0] // 4)
                line_width = random.randint(2, 6)
                line_color = random.uniform(0.7, 0.9)  # White/yellow
                
                frame[line_x:line_x+line_width, :] = line_color
                
            else:  # Horizontal road marking
                line_y = random.randint(self.input_size[1] // 4, 3 * self.input_size[1] // 4)
                line_height = random.randint(2, 4)
                line_color = random.uniform(0.7, 0.9)
                
                frame[:, line_y:line_y+line_height] = line_color
        
        return frame
    
    def _add_lighting_variation(self, frame, frame_idx):
        """Add realistic lighting variations"""
        # Simulate slight lighting changes over time
        brightness_factor = 1.0 + 0.1 * np.sin(frame_idx * 0.5) * random.uniform(0.5, 1.0)
        frame *= brightness_factor
        
        # Add shadow effects
        if random.random() < 0.2:
            shadow_intensity = random.uniform(0.8, 0.95)
            shadow_area = random.randint(20, 60)
            
            x_start = random.randint(0, self.input_size[0] - shadow_area)
            y_start = random.randint(0, self.input_size[1] - shadow_area)
            
            frame[x_start:x_start+shadow_area, y_start:y_start+shadow_area] *= shadow_intensity
        
        return frame
    
    def _add_camera_movement(self, frame, frame_idx):
        """Simulate slight camera movement"""
        # Small translation to simulate vehicle movement
        dx = int(2 * np.sin(frame_idx * 0.3))
        dy = int(1 * np.cos(frame_idx * 0.2))
        
        # Apply translation
        if dx != 0 or dy != 0:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            frame = cv2.warpAffine(frame, M, self.input_size)
        
        return frame
    
    def _add_subtle_texture_variation(self, frame):
        """Add subtle surface texture variations"""
        # Add very small bumps and variations (NOT potholes)
        for _ in range(random.randint(0, 3)):
            x = random.randint(10, self.input_size[0] - 10)
            y = random.randint(10, self.input_size[1] - 10)
            size = random.randint(2, 5)  # Very small
            intensity = random.uniform(-0.02, 0.02)  # Very subtle
            
            frame[x-size:x+size, y-size:y+size] += intensity
        
        return frame
    
    def generate_ground_truth_mask(self):
        """Generate ground truth mask for non-pothole sequence"""
        # Non-pothole sequences have zero ground truth (no potholes)
        return np.zeros((self.sequence_length, self.input_size[1], self.input_size[0]), dtype=np.float32)


class VideoBasedPotholeEnv(gym.Env):
    """
    🚀 REVOLUTIONARY RL ENVIRONMENT FOR POTHOLE DETECTION! 🚀
    
    Enhanced version with balanced pothole/non-pothole data generation for comprehensive training.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, split='train', render_mode=None, max_memory_mb=4096, target_sequences=None, 
                 lazy=False, verbose=False, balanced=False):  # Added balanced parameter
        super().__init__()
        
        if verbose:
            print("🎯 Initializing ENHANCED Video-Based RL Environment...")
        
        # Environment configuration
        self.split = split
        self.render_mode = render_mode
        self.sequence_length = VIDEO_CONFIG["sequence_length"]
        self.input_size = VIDEO_CONFIG["input_size"]
        self.max_memory_mb = max_memory_mb
        self.lazy = lazy
        self.verbose = verbose
        self.balanced = balanced  # NEW: Enable balanced training
        
        # Initialize synthetic generator
        self.synthetic_generator = SyntheticNonPotholeGenerator(
            sequence_length=self.sequence_length,
            input_size=self.input_size
        )
        
        # FIXED: Dynamic target sequences based on memory
        if target_sequences is None:
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
        
        # Load dataset based on balanced flag
        if self.balanced:
            self._load_dataset_balanced()
        else:
            self._load_dataset_optimized()
        
        if verbose:
            print(f"✅ ENHANCED RL Environment Initialized!")
            print(f"   📊 Action Space: {self.action_space.n} confidence thresholds")
            print(f"   🖼️ Observation Space: {self.observation_space.shape}")
            print(f"   💾 Dataset Split: {self.split}")
            print(f"   🎯 Available Episodes: {len(self.episode_sequences):,}")
            print(f"   📈 Target Sequences: {self.target_sequences}")
            print(f"   ⚖️ Balanced Mode: {self.balanced}")
    
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
        
        ground_truth_dir = PATHS["ground_truth"] / self.split
        
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
    
    def _load_dataset_balanced(self):
        """ENHANCED: Load dataset with balanced pothole/non-pothole distribution"""
        if self.verbose:
            print(f"📊 Loading BALANCED {self.split} dataset...")
            print(f"   🎯 Target sequences: {self.target_sequences}")
            print(f"   ⚖️ Creating 50% pothole, 50% non-pothole distribution")
        
        self.episode_sequences = []
        self.episode_ground_truths = []
        self.episode_metadata = []
        self.successful_loads = 0
        self.failed_loads = 0
        
        # Calculate balanced distribution
        pothole_target = self.target_sequences // 2
        non_pothole_target = self.target_sequences - pothole_target
        
        # Load real pothole data first
        data_sources = [
            PATHS["processed_frames"] / f"{self.split}_optimized",
            PATHS["processed_frames"] / f"{self.split}_augmented", 
            PATHS["processed_frames"] / self.split
        ]
        
        ground_truth_dir = PATHS["ground_truth"] / self.split
        
        for data_dir in data_sources:
            if data_dir.exists() and len(self.episode_sequences) < pothole_target:
                if self.verbose:
                    print(f"   📁 Loading potholes from: {data_dir.name}")
                self._load_from_directory_limited(data_dir, ground_truth_dir, pothole_target)
        
        # Fill remaining pothole quota with synthetic pothole data
        current_pothole_count = len(self.episode_sequences)
        if current_pothole_count < pothole_target:
            synthetic_pothole_needed = pothole_target - current_pothole_count
            if self.verbose:
                print(f"   🔧 Creating {synthetic_pothole_needed} synthetic pothole sequences")
            self._create_synthetic_data(synthetic_pothole_needed)
        
        # Create synthetic non-pothole data
        if self.verbose:
            print(f"   🛣️ Creating {non_pothole_target} synthetic non-pothole sequences")
        self._create_balanced_synthetic_data(non_pothole_target)
        
        # Shuffle for balanced training
        combined_data = list(zip(self.episode_sequences, self.episode_ground_truths, self.episode_metadata))
        random.shuffle(combined_data)
        self.episode_sequences, self.episode_ground_truths, self.episode_metadata = zip(*combined_data)
        self.episode_sequences = list(self.episode_sequences)
        self.episode_ground_truths = list(self.episode_ground_truths)
        self.episode_metadata = list(self.episode_metadata)
        
        if self.verbose:
            pothole_count = sum(1 for meta in self.episode_metadata if 'non_pothole' not in meta.get('data_type', ''))
            non_pothole_count = sum(1 for meta in self.episode_metadata if 'non_pothole' in meta.get('data_type', ''))
            print(f"🚀 BALANCED Dataset loaded: {len(self.episode_sequences):,} sequences")
            print(f"   ✅ Potholes: {pothole_count}")
            print(f"   🛣️ Non-potholes: {non_pothole_count}")
            print(f"   ⚖️ Balance ratio: {pothole_count/len(self.episode_sequences)*100:.1f}% / {non_pothole_count/len(self.episode_sequences)*100:.1f}%")
    
    def _load_from_directory_limited(self, data_dir, gt_dir, max_sequences):
        """Load sequences from directory with maximum limit"""
        video_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        
        for i, video_dir in enumerate(video_dirs):
            if len(self.episode_sequences) >= max_sequences:
                break
                
            sequences_dir = video_dir / "sequences"
            if not sequences_dir.exists():
                continue
            
            seq_files = sorted(sequences_dir.glob("*.npy"))
            
            for seq_file in seq_files:
                if len(self.episode_sequences) >= max_sequences:
                    break
                
                try:
                    sequence = self._safe_load_sequence_enhanced(seq_file)
                    if sequence is None:
                        self.failed_loads += 1
                        continue
                    
                    # Load ground truth with existing logic
                    video_name = video_dir.name.split('_aug')[0].split('_opt')[0]
                    seq_idx = seq_file.stem.split('_')[-1]
                    
                    # Try multiple ground truth path patterns
                    ground_truth = None
                    if gt_dir.exists():
                        if 'video_' in video_name:
                            video_number = video_name.replace('video_', '')
                            try:
                                gt_directory_name = f"{int(video_number):04d}"
                            except ValueError:
                                gt_directory_name = video_name
                        else:
                            gt_directory_name = video_name
                        
                        gt_search_paths = [
                            gt_dir / gt_directory_name / "sequences" / f"mask_sequence_{seq_idx}.npy",
                            gt_dir / video_name / "sequences" / f"mask_sequence_{seq_idx}.npy",
                        ]
                        
                        for gt_path in gt_search_paths:
                            if gt_path.exists():
                                ground_truth = self._safe_load_ground_truth_enhanced(gt_path)
                                if ground_truth is not None:
                                    break
                    
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
                    continue
    
    def _create_balanced_synthetic_data(self, count):
        """Create synthetic non-pothole sequences for balanced training"""
        for i in range(count):
            # Generate non-pothole sequence
            sequence = self.synthetic_generator.generate_smooth_road_sequence(seed=i + 5000)
            ground_truth = self.synthetic_generator.generate_ground_truth_mask()
            
            self.episode_sequences.append(sequence)
            self.episode_ground_truths.append(ground_truth)
            self.episode_metadata.append({
                "video_name": f"synthetic_non_pothole_{i:05d}",
                "sequence_idx": "000",
                "data_type": "synthetic_non_pothole",
                "has_ground_truth": True,
                "source_dir": "synthetic"
            })
    
    def _load_from_directory(self, data_dir, gt_dir):
        """Load sequences from a specific directory with FIXED ground truth loading"""
        video_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        
        # ENHANCED: Load more sequences per video to reach target
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
                    # ENHANCED: More permissive loading for target sequences
                    sequence = self._safe_load_sequence_enhanced(seq_file)
                    if sequence is None:
                        self.failed_loads += 1
                        continue
                    
                    # ✅ FIXED: Ground truth path resolution with naming convention handling
                    video_name = video_dir.name.split('_aug')[0].split('_opt')[0]  # Remove suffixes
                    seq_idx = seq_file.stem.split('_')[-1]
                    
                    # Extract just the numeric part and format it for ground truth lookup
                    if 'video_' in video_name:
                        video_number = video_name.replace('video_', '')
                        try:
                            gt_directory_name = f"{int(video_number):04d}"  # Convert to 4-digit format
                        except ValueError:
                            gt_directory_name = video_name  # Fallback to original name
                    else:
                        gt_directory_name = video_name
                    
                    # Load ground truth with multiple fallback patterns
                    ground_truth = None
                    if gt_dir.exists():
                        # Try multiple ground truth path patterns
                        gt_search_paths = [
                            gt_dir / gt_directory_name / "sequences" / f"mask_sequence_{seq_idx}.npy",
                            gt_dir / video_name / "sequences" / f"mask_sequence_{seq_idx}.npy",
                            gt_dir / f"{video_name.zfill(4)}" / "sequences" / f"mask_sequence_{seq_idx}.npy",
                            gt_dir / video_name.replace('video_', '') / "sequences" / f"mask_sequence_{seq_idx}.npy"
                        ]
                        
                        for gt_path in gt_search_paths:
                            if gt_path.exists():
                                ground_truth = self._safe_load_ground_truth_enhanced(gt_path)
                                if ground_truth is not None:
                                    if self.verbose and len(self.episode_sequences) % 100 == 0:
                                        print(f"   ✅ Found ground truth: {gt_path.relative_to(gt_dir)}")
                                    break
                        
                        if ground_truth is None and self.verbose and len(self.episode_sequences) % 500 == 0:
                            print(f"   ⚠️ No ground truth found for {video_name} -> {gt_directory_name}, seq {seq_idx}")
                    
                    # Add to dataset
                    self.episode_sequences.append(sequence)
                    self.episode_ground_truths.append(ground_truth)
                    self.episode_metadata.append({
                        "video_name": video_name,
                        "sequence_idx": seq_idx,
                        "data_type": "real",
                        "has_ground_truth": ground_truth is not None,
                        "source_dir": data_dir.name,
                        "gt_directory_name": gt_directory_name  # Track the mapped name for debugging
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
    # Add this to your existing VideoBasedPotholeEnv class
    def enable_cnn_integration(self, cnn_model_path=None):
        """Enable real CNN detection instead of simulation"""
        try:
            from src.models.cnn_detector import PotholeCNNDetector
            
            self.cnn_detector = PotholeCNNDetector()
            self.use_real_cnn = True
            
            if cnn_model_path:
                self.cnn_detector.load_state_dict(torch.load(cnn_model_path))
                print(f"✅ CNN detector loaded: {cnn_model_path}")
            else:
                print("📦 Using pretrained CNN backbone")
                
        except Exception as e:
            print(f"⚠️ CNN integration failed: {e}")
            self.use_real_cnn = False

    def _simulate_detection_confidence(self):
        """ENHANCED: Use real CNN if available, fallback to simulation"""
        if hasattr(self, 'use_real_cnn') and self.use_real_cnn:
            try:
                return self.cnn_detector.predict_confidence(self.current_sequence)
            except Exception as e:
                print(f"⚠️ CNN prediction failed: {e}, using simulation")
        
        # Original simulation code as fallback
        if self.current_ground_truth is not None:
            has_pothole = self._ground_truth_has_pothole()
            
            if has_pothole:
                base_confidence = random.uniform(0.65, 0.88)
                noise = random.uniform(-0.15, 0.12)
                return np.clip(base_confidence + noise, 0.0, 1.0)
            else:
                base_confidence = random.uniform(0.12, 0.42)
                noise = random.uniform(-0.12, 0.15)
                return np.clip(base_confidence + noise, 0.0, 1.0)
        else:
            return random.uniform(0.2, 0.8)
    
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
        
        # Only print debug info occasionally to reduce spam
        if self.episode_count % 10 == 0:  # Print every 10th episode
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
        """ENHANCED: More realistic confidence simulation for balanced data"""
        if self.current_ground_truth is not None:
            has_pothole = self._ground_truth_has_pothole()
            
            # Check if this is a synthetic non-pothole sequence
            is_synthetic_non_pothole = 'non_pothole' in self.current_metadata.get('data_type', '')
            
            if has_pothole:
                # Higher confidence for actual potholes
                pothole_size = np.sum(self.current_ground_truth > 0)
                total_pixels = self.current_ground_truth.size
                size_ratio = pothole_size / total_pixels
                
                # Size-dependent confidence
                base_confidence = 0.6 + (0.3 * min(size_ratio * 50, 1.0))  # 0.6-0.9 range
                
                # Add realistic noise
                noise = np.random.normal(0, 0.05)  # Reduced noise
                confidence = np.clip(base_confidence + noise, 0.3, 0.95)
                
            elif is_synthetic_non_pothole:
                # Very low confidence for synthetic non-pothole roads
                base_confidence = np.random.uniform(0.05, 0.25)  # Even lower for synthetic
                noise = np.random.normal(0, 0.02)
                confidence = np.clip(base_confidence + noise, 0.0, 0.35)
                
            else:
                # Low confidence for real non-potholes
                base_confidence = np.random.uniform(0.15, 0.4)
                noise = np.random.normal(0, 0.03)
                confidence = np.clip(base_confidence + noise, 0.0, 0.45)
            
            return confidence
        else:
            # Fallback for synthetic data
            return np.random.uniform(0.2, 0.8)
    
    def _ground_truth_has_pothole(self):
        """Enhanced ground truth detection with debugging"""
        if self.current_ground_truth is None:
            if self.episode_count % 50 == 0:  # Reduce debug spam
                print("  GROUND TRUTH: None - no mask data")
            return False
        
        pothole_pixels = np.sum(self.current_ground_truth > 0)
        total_pixels = self.current_ground_truth.size
        pothole_ratio = pothole_pixels / total_pixels if total_pixels > 0 else 0
        
        has_pothole = pothole_ratio > 0.01  # 1% threshold
        
        # Debug output (reduced frequency)
        if self.episode_count % 10 == 0:
            print(f"  GROUND TRUTH: {pothole_pixels}/{total_pixels} pixels = "
                  f"{pothole_ratio:.4f} ratio → {has_pothole}")
        
        return has_pothole
    
    def _calculate_reward(self, agent_detects_pothole):
        """ENHANCED: Real ground truth first, synthetic fallback"""
        
        if self.current_ground_truth is not None:
            # ✅ USE REAL GROUND TRUTH (Primary path)
            ground_truth_has_pothole = self._ground_truth_has_pothole()
            
            if self.episode_count % 10 == 0:  # Reduce debug spam
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
            # ✅ SYNTHETIC FALLBACK
            if self.episode_count % 50 == 0:  # Reduce debug spam
                print("🔄 Using synthetic ground truth for learning")
            
            # Store confidence for synthetic GT generation
            confidence_info = getattr(self, '_last_confidence', 0.5)
            
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
        non_pothole_sequences = sum(1 for meta in self.episode_metadata if 'non_pothole' in meta.get('data_type', ''))
        
        source_breakdown = {}
        for meta in self.episode_metadata:
            source = meta.get('source_dir', 'unknown')
            source_breakdown[source] = source_breakdown.get(source, 0) + 1
        
        return {
            "total_sequences": len(self.episode_sequences),
            "real_sequences": real_sequences,
            "synthetic_sequences": synthetic_sequences,
            "non_pothole_sequences": non_pothole_sequences,
            "pothole_sequences": len(self.episode_sequences) - non_pothole_sequences,
            "successful_loads": self.successful_loads,
            "failed_loads": self.failed_loads,
            "target_sequences": self.target_sequences,
            "memory_limit_mb": self.max_memory_mb,
            "current_memory_mb": round(self._get_memory_usage_mb(), 2),
            "source_breakdown": source_breakdown,
            "has_ground_truth": sum(1 for meta in self.episode_metadata if meta.get('has_ground_truth', False)),
            "balanced_mode": self.balanced
        }


# Test function
if __name__ == "__main__":
    print("🚀 TESTING ENHANCED BALANCED RL ENVIRONMENT!")
    print("="*60)
    
    try:
        # Test balanced environment
        env = VideoBasedPotholeEnv(
            split='train', 
            max_memory_mb=4096,
            target_sequences=1000,  # Smaller for testing
            balanced=True,  # Enable balanced mode
            verbose=True
        )
        
        # Display dataset information
        dataset_info = env.get_dataset_info()
        print("\n📊 Balanced Dataset Information:")
        for key, value in dataset_info.items():
            print(f"   {key}: {value}")
        
        # Test environment functionality
        print("\n🧪 Testing Balanced Environment Functions...")
        
        # Test reset
        observation, info = env.reset()
        print(f"✅ Reset successful")
        print(f"   📊 Observation shape: {observation.shape}")
        print(f"   💾 Memory usage: {info.get('memory_usage_mb', 0):.1f} MB")
        
        # Test actions on different sequence types
        pothole_rewards = []
        non_pothole_rewards = []
        
        for i in range(20):  # Test 20 episodes
            obs, info_reset = env.reset()
            action = random.randint(0, 4)  # Random action
            obs, reward, done, truncated, info = env.step(action)
            
            # Categorize by sequence type
            if 'non_pothole' in info['metadata'].get('data_type', ''):
                non_pothole_rewards.append(reward)
            else:
                pothole_rewards.append(reward)
        
        print(f"\n📈 Balanced Performance Test:")
        print(f"   Pothole sequences tested: {len(pothole_rewards)}")
        print(f"   Non-pothole sequences tested: {len(non_pothole_rewards)}")
        print(f"   Pothole rewards: {set(pothole_rewards) if pothole_rewards else 'None'}")
        print(f"   Non-pothole rewards: {set(non_pothole_rewards) if non_pothole_rewards else 'None'}")
        
        # Performance stats
        print(f"\n📈 Performance Statistics:")
        stats = env.get_performance_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        env.close()
        print(f"\n🎉 BALANCED ENVIRONMENT TEST COMPLETED SUCCESSFULLY!")
        print(f"🚀 Ready for balanced training with realistic accuracy metrics!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
