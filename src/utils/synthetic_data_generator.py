import numpy as np
import cv2
from pathlib import Path
import random
from scipy import ndimage
from sklearn.preprocessing import normalize
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import VIDEO_CONFIG, ENV_CONFIG

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
    
    def create_non_pothole_dataset(self, count=2500, output_dir=None):
        """Create a complete dataset of non-pothole sequences"""
        sequences = []
        ground_truths = []
        metadata = []
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(count):
            # Generate sequence
            sequence = self.generate_smooth_road_sequence(seed=i + 1000)
            ground_truth = self.generate_ground_truth_mask()
            
            sequences.append(sequence)
            ground_truths.append(ground_truth)
            metadata.append({
                "video_name": f"synthetic_non_pothole_{i:05d}",
                "sequence_idx": "000",
                "data_type": "synthetic_non_pothole",
                "has_ground_truth": True,
                "source_dir": "synthetic"
            })
            
            # Save to disk if output directory specified
            if output_dir:
                seq_dir = output_dir / f"synthetic_non_pothole_{i:05d}" / "sequences"
                seq_dir.mkdir(parents=True, exist_ok=True)
                
                np.save(seq_dir / "sequence_000.npy", sequence)
                np.save(seq_dir / "mask_sequence_000.npy", ground_truth)
        
        return sequences, ground_truths, metadata

# Factory function for easy integration
def create_synthetic_generator():
    """Factory function to create generator with config parameters"""
    return SyntheticNonPotholeGenerator(
        sequence_length=VIDEO_CONFIG["sequence_length"],
        input_size=VIDEO_CONFIG["input_size"]
    )
