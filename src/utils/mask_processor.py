import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import PATHS
from tqdm import tqdm
import json

class MaskProcessor:
    """Process ground truth segmentation masks for RL training"""
    
    def __init__(self):
        self.input_size = (224, 224)  # Match video processing
        
    def load_mask(self, mask_path):
        """Load and preprocess segmentation mask"""
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        
        # Resize to match frame size
        mask_resized = cv2.resize(mask, self.input_size)
        
        # Binarize mask (pothole = 1, background = 0)
        _, binary_mask = cv2.threshold(mask_resized, 127, 1, cv2.THRESH_BINARY)
        
        return binary_mask.astype(np.float32)
    
    def extract_pothole_info(self, mask):
        """Extract pothole information from mask"""
        if mask is None:
            return {"has_pothole": False, "pothole_area": 0, "pothole_ratio": 0}
        
        pothole_pixels = np.sum(mask)
        total_pixels = mask.size
        pothole_ratio = pothole_pixels / total_pixels
        
        return {
            "has_pothole": pothole_pixels > 0,
            "pothole_area": int(pothole_pixels),
            "pothole_ratio": float(pothole_ratio),
            "total_pixels": total_pixels
        }
    
    def process_video_masks(self, video_name, split_name):
        """Process all masks for a video"""
        mask_dir = PATHS["raw_videos"] / split_name / "mask"
        
        # Find corresponding mask file
        mask_file = None
        for ext in ['.mp4', '.avi']:
            potential_mask = mask_dir / f"{video_name}{ext}"
            if potential_mask.exists():
                mask_file = potential_mask
                break
        
        if not mask_file:
            print(f"Warning: No mask found for {video_name}")
            return None
        
        # Extract frames from mask video
        cap = cv2.VideoCapture(str(mask_file))
        masks = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale and process
            gray_mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed_mask = cv2.resize(gray_mask, self.input_size)
            _, binary_mask = cv2.threshold(processed_mask, 127, 1, cv2.THRESH_BINARY)
            
            masks.append(binary_mask.astype(np.float32))
        
        cap.release()
        return masks
    
    def create_ground_truth_sequences(self, masks, sequence_length=5, overlap=2):
        """Create temporal mask sequences matching video sequences"""
        if len(masks) < sequence_length:
            return []
        
        sequences = []
        for i in range(0, len(masks) - sequence_length + 1, sequence_length - overlap):
            sequence = masks[i:i + sequence_length]
            sequences.append(np.array(sequence))
        
        return sequences
    
    def validate_rgb_mask_correspondence(self, split_name):
        """Validate RGB videos have corresponding masks"""
        rgb_dir = PATHS["raw_videos"] / split_name / "rgb"
        mask_dir = PATHS["raw_videos"] / split_name / "mask"
        
        rgb_files = {f.stem for f in rgb_dir.glob("*.mp4")}
        mask_files = {f.stem for f in mask_dir.glob("*.mp4")}
        
        missing_masks = rgb_files - mask_files
        extra_masks = mask_files - rgb_files
        
        print(f"\n{split_name.upper()} RGB-MASK CORRESPONDENCE:")
        print(f"  RGB videos: {len(rgb_files)}")
        print(f"  Mask videos: {len(mask_files)}")
        print(f"  Missing masks: {len(missing_masks)}")
        print(f"  Extra masks: {len(extra_masks)}")
        
        if missing_masks:
            print(f"  Missing mask files: {list(missing_masks)[:5]}...")
        
        return len(missing_masks) == 0

if __name__ == "__main__":
    processor = MaskProcessor()
    for split in ["train", "val", "test"]:
        processor.validate_rgb_mask_correspondence(split)
