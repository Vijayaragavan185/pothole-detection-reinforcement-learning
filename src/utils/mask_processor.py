import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import PATHS, VIDEO_CONFIG
from tqdm import tqdm
import json

class MaskProcessor:
    """Process ground truth segmentation masks for RL training"""
    
    def __init__(self):
        self.input_size = VIDEO_CONFIG["input_size"]  # (224, 224)
        self.sequence_length = VIDEO_CONFIG["sequence_length"]  # 5
        self.overlap = VIDEO_CONFIG["overlap"]  # 2
        
    def load_mask_video(self, mask_path):
        """Load mask video and return frames"""
        cap = cv2.VideoCapture(str(mask_path))
        mask_frames = []
        
        if not cap.isOpened():
            print(f"Error: Cannot open mask video {mask_path}")
            return None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame
            
            mask_frames.append(gray_frame)
        
        cap.release()
        return mask_frames
    
    def preprocess_mask(self, mask_frame):
        """Preprocess single mask frame"""
        # Resize to model input size
        resized_mask = cv2.resize(mask_frame, self.input_size)
        
        # Binarize mask (pothole = 1, background = 0)
        _, binary_mask = cv2.threshold(resized_mask, 127, 1, cv2.THRESH_BINARY)
        
        return binary_mask.astype(np.float32)
    
    def extract_pothole_info(self, mask):
        """Extract detailed pothole information from mask"""
        if mask is None:
            return self._empty_pothole_info()
        
        pothole_pixels = np.sum(mask)
        total_pixels = mask.size
        
        if pothole_pixels == 0:
            return self._empty_pothole_info()
        
        # Find contours to get bounding box
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self._empty_pothole_info()
        
        # Get largest contour (main pothole)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate additional metrics
        pothole_ratio = pothole_pixels / total_pixels
        aspect_ratio = w / h if h > 0 else 1.0
        center_x, center_y = x + w/2, y + h/2
        
        return {
            "has_pothole": True,
            "pothole_area": int(pothole_pixels),
            "pothole_ratio": float(pothole_ratio),
            "bbox": [int(x), int(y), int(w), int(h)],
            "center": [float(center_x), float(center_y)],
            "aspect_ratio": float(aspect_ratio),
            "total_pixels": int(total_pixels)
        }
    
    def _empty_pothole_info(self):
        """Return empty pothole info structure"""
        return {
            "has_pothole": False,
            "pothole_area": 0,
            "pothole_ratio": 0.0,
            "bbox": [0, 0, 0, 0],
            "center": [0.0, 0.0],
            "aspect_ratio": 1.0,
            "total_pixels": self.input_size[0] * self.input_size[1]
        }
    
    def create_mask_sequences(self, mask_frames):
        """Create temporal mask sequences matching video sequences"""
        if len(mask_frames) < self.sequence_length:
            print(f"Warning: Mask has {len(mask_frames)} frames, less than required {self.sequence_length}")
            return []
        
        sequences = []
        sequence_info = []
        
        # Create overlapping sequences
        for i in range(0, len(mask_frames) - self.sequence_length + 1, self.sequence_length - self.overlap):
            sequence_masks = mask_frames[i:i + self.sequence_length]
            sequence_array = np.array(sequence_masks)
            sequences.append(sequence_array)
            
            # Extract info for each frame in sequence
            frame_info = []
            for mask in sequence_masks:
                info = self.extract_pothole_info(mask)
                frame_info.append(info)
            
            # Sequence-level statistics
            pothole_frames = sum(1 for info in frame_info if info["has_pothole"])
            avg_pothole_ratio = np.mean([info["pothole_ratio"] for info in frame_info])
            
            sequence_info.append({
                "sequence_index": len(sequences) - 1,
                "pothole_frames": pothole_frames,
                "total_frames": len(sequence_masks),
                "pothole_frame_ratio": pothole_frames / len(sequence_masks),
                "avg_pothole_ratio": float(avg_pothole_ratio),
                "frame_info": frame_info
            })
        
        return sequences, sequence_info
    
    def process_video_masks(self, video_name, split_name):
        """Process masks for a specific video"""
        # Find corresponding mask file
        mask_dir = PATHS["raw_videos"] / split_name / "mask"
        mask_file = None
        
        for ext in ['.mp4', '.avi']:
            potential_mask = mask_dir / f"{video_name}{ext}"
            if potential_mask.exists():
                mask_file = potential_mask
                break
        
        if not mask_file:
            print(f"Warning: No mask found for {video_name} in {split_name}")
            return None, None
        
        # Load and process mask video
        mask_frames = self.load_mask_video(mask_file)
        if mask_frames is None:
            return None, None
        
        # Preprocess all frames
        processed_masks = []
        for mask_frame in mask_frames:
            processed_mask = self.preprocess_mask(mask_frame)
            processed_masks.append(processed_mask)
        
        # Create sequences
        mask_sequences, sequence_info = self.create_mask_sequences(processed_masks)
        
        return mask_sequences, sequence_info
    
    def process_split_masks(self, split_name):
        """Process all masks in a dataset split"""
        print(f"\nProcessing masks for {split_name} split...")
        
        rgb_dir = PATHS["raw_videos"] / split_name / "rgb"
        output_dir = PATHS["ground_truth"] / split_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all RGB video files to find corresponding masks
        rgb_files = list(rgb_dir.glob("*.mp4"))
        if not rgb_files:
            rgb_files = list(rgb_dir.glob("*.avi"))
        
        print(f"Processing masks for {len(rgb_files)} videos...")
        
        successful = 0
        failed = 0
        total_sequences = 0
        
        for rgb_file in tqdm(rgb_files, desc=f"Processing {split_name} masks"):
            video_name = rgb_file.stem
            
            try:
                mask_sequences, sequence_info = self.process_video_masks(video_name, split_name)
                
                if mask_sequences is not None:
                    # Save processed masks
                    video_output_dir = output_dir / video_name
                    video_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save sequences
                    sequences_dir = video_output_dir / "sequences"
                    sequences_dir.mkdir(exist_ok=True)
                    
                    for i, sequence in enumerate(mask_sequences):
                        sequence_path = sequences_dir / f"mask_sequence_{i:03d}.npy"
                        np.save(sequence_path, sequence)
                    
                    # Save sequence info
                    info_path = video_output_dir / "sequence_info.json"
                    with open(info_path, 'w') as f:
                        json.dump(sequence_info, f, indent=2)
                    
                    # Save metadata
                    metadata = {
                        "video_name": video_name,
                        "total_mask_frames": len(mask_sequences[0]) if mask_sequences else 0,
                        "total_sequences": len(mask_sequences),
                        "sequence_length": self.sequence_length,
                        "overlap": self.overlap,
                        "input_size": self.input_size
                    }
                    
                    metadata_path = video_output_dir / "mask_metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    successful += 1
                    total_sequences += len(mask_sequences)
                else:
                    failed += 1
                    
            except Exception as e:
                print(f"Error processing {video_name}: {e}")
                failed += 1
        
        print(f"\n{split_name} mask processing complete:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total mask sequences: {total_sequences}")
        
        return successful, failed, total_sequences
    
    def validate_rgb_mask_correspondence(self, split_name):
        """Validate RGB videos have corresponding masks"""
        rgb_dir = PATHS["raw_videos"] / split_name / "rgb"
        mask_dir = PATHS["raw_videos"] / split_name / "mask"
        
        if not rgb_dir.exists() or not mask_dir.exists():
            print(f"Missing directories for {split_name}")
            return False
        
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
    
    def process_all_splits(self):
        """Process masks for all dataset splits"""
        print("Starting mask processing pipeline...")
        
        splits = ["train", "val", "test"]
        total_successful = 0
        total_failed = 0
        total_sequences = 0
        
        # First validate correspondence
        for split in splits:
            self.validate_rgb_mask_correspondence(split)
        
        # Process each split
        for split in splits:
            successful, failed, sequences = self.process_split_masks(split)
            total_successful += successful
            total_failed += failed
            total_sequences += sequences
        
        print(f"\nMask Processing Summary:")
        print(f"Total successful: {total_successful}")
        print(f"Total failed: {total_failed}")
        print(f"Total mask sequences: {total_sequences}")
        print(f"Success rate: {total_successful/(total_successful+total_failed)*100:.1f}%")
        
        return total_successful, total_failed, total_sequences

if __name__ == "__main__":
    processor = MaskProcessor()
    processor.process_all_splits()
