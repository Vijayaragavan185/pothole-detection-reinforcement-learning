import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import VIDEO_CONFIG, PATHS
from tqdm import tqdm
import json

class VideoProcessor:
    """Video processing utilities for pothole detection dataset"""
    
    def __init__(self):
        self.input_size = VIDEO_CONFIG["input_size"]
        self.sequence_length = VIDEO_CONFIG["sequence_length"]
        self.overlap = VIDEO_CONFIG["overlap"]
        
    def load_video(self, video_path):
        """Load video and return frames"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return frames
    
    def preprocess_frame(self, frame):
        """Preprocess single frame for model input"""
        # Resize to model input size
        resized = cv2.resize(frame, self.input_size)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb_frame.astype(np.float32) / 255.0
        
        return normalized
    
    def create_temporal_sequences(self, frames):
        """Create temporal sequences for RL processing"""
        sequences = []
        
        if len(frames) < self.sequence_length:
            print(f"Warning: Video has {len(frames)} frames, less than required {self.sequence_length}")
            return sequences
        
        # Create overlapping sequences
        for i in range(0, len(frames) - self.sequence_length + 1, self.sequence_length - self.overlap):
            sequence = frames[i:i + self.sequence_length]
            sequences.append(sequence)
        
        return sequences
    
    def process_video_file(self, video_path, output_dir):
        """Process single video file and save preprocessed frames"""
        # Load video
        frames = self.load_video(video_path)
        if frames is None:
            return False
        
        # Create output directory
        video_name = video_path.stem
        video_output_dir = output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process frames
        processed_frames = []
        for i, frame in enumerate(frames):
            processed_frame = self.preprocess_frame(frame)
            processed_frames.append(processed_frame)
            
            # Save individual frame
            frame_path = video_output_dir / f"frame_{i:03d}.npy"
            np.save(frame_path, processed_frame)
        
        # Create temporal sequences
        sequences = self.create_temporal_sequences(processed_frames)
        
        # Save sequences
        sequences_dir = video_output_dir / "sequences"
        sequences_dir.mkdir(exist_ok=True)
        
        for i, sequence in enumerate(sequences):
            sequence_array = np.array(sequence)
            sequence_path = sequences_dir / f"sequence_{i:03d}.npy"
            np.save(sequence_path, sequence_array)
        
        # Save metadata
        metadata = {
            "original_frames": len(frames),
            "processed_frames": len(processed_frames),
            "sequences": len(sequences),
            "input_size": self.input_size,
            "sequence_length": self.sequence_length,
            "overlap": self.overlap
        }
        
        metadata_path = video_output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True
    
    def process_dataset_split(self, split_name):
        """Process entire dataset split (train/val/test)"""
        print(f"\nProcessing {split_name} split...")
        
        # Input and output paths
        input_dir = PATHS["raw_videos"] / split_name / "rgb"
        output_dir = PATHS["processed_frames"] / split_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all video files
        video_files = list(input_dir.glob("*.mp4"))
        if not video_files:
            video_files = list(input_dir.glob("*.avi"))
        
        print(f"Found {len(video_files)} videos in {split_name}")
        
        # Process each video
        successful = 0
        failed = 0
        
        for video_file in tqdm(video_files, desc=f"Processing {split_name}"):
            try:
                success = self.process_video_file(video_file, output_dir)
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
                failed += 1
        
        print(f"{split_name} processing complete:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        
        return successful, failed
    
    def process_all_splits(self):
        """Process all dataset splits"""
        print("Starting video processing pipeline...")
        
        splits = ["train", "val", "test"]
        total_successful = 0
        total_failed = 0
        
        for split in splits:
            successful, failed = self.process_dataset_split(split)
            total_successful += successful
            total_failed += failed
        
        print(f"\nProcessing Summary:")
        print(f"Total successful: {total_successful}")
        print(f"Total failed: {total_failed}")
        print(f"Success rate: {total_successful/(total_successful+total_failed)*100:.1f}%")
        
        return total_successful, total_failed

if __name__ == "__main__":
    processor = VideoProcessor()
    processor.process_all_splits()
