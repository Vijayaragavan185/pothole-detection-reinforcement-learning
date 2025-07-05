import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import PATHS, VIDEO_CONFIG
from tqdm import tqdm
import argparse

class FrameExtractor:
    """Extract and save individual frames from videos"""
    
    def __init__(self):
        self.input_size = VIDEO_CONFIG["input_size"]
        
    def extract_frames_from_video(self, video_path, output_dir, max_frames=None):
        """Extract frames from a single video"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error: Cannot open {video_path}")
            return 0
        
        # Create output directory
        video_name = video_path.stem
        video_output_dir = output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Limit frames if specified
            if max_frames and extracted_count >= max_frames:
                break
            
            # Save original frame
            frame_filename = f"frame_{frame_count:03d}.jpg"
            frame_path = video_output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            
            # Save resized frame for model input
            resized_frame = cv2.resize(frame, self.input_size)
            resized_filename = f"frame_{frame_count:03d}_resized.jpg"
            resized_path = video_output_dir / resized_filename
            cv2.imwrite(str(resized_path), resized_frame)
            
            frame_count += 1
            extracted_count += 1
        
        cap.release()
        print(f"Extracted {extracted_count} frames from {video_name}")
        return extracted_count
    
    def extract_frames_from_split(self, split_name, max_videos=None, max_frames_per_video=None):
        """Extract frames from all videos in a split"""
        print(f"\nExtracting frames from {split_name} split...")
        
        # Input and output paths
        input_dir = PATHS["raw_videos"] / split_name / "rgb"
        output_dir = PATHS["processed_frames"] / split_name / "extracted_frames"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get video files
        video_files = list(input_dir.glob("*.mp4"))
        if not video_files:
            video_files = list(input_dir.glob("*.avi"))
        
        if max_videos:
            video_files = video_files[:max_videos]
        
        print(f"Processing {len(video_files)} videos...")
        
        total_frames = 0
        successful_videos = 0
        
        for video_file in tqdm(video_files, desc=f"Extracting {split_name}"):
            try:
                frames_extracted = self.extract_frames_from_video(
                    video_file, output_dir, max_frames_per_video
                )
                total_frames += frames_extracted
                successful_videos += 1
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
        
        print(f"\n{split_name} extraction complete:")
        print(f"  Videos processed: {successful_videos}/{len(video_files)}")
        print(f"  Total frames extracted: {total_frames}")
        
        return total_frames
    
    def extract_sample_frames(self, num_videos_per_split=5, max_frames_per_video=10):
        """Extract sample frames for quick testing"""
        print("Extracting sample frames for testing...")
        
        splits = ["train", "val", "test"]
        total_frames = 0
        
        for split in splits:
            frames = self.extract_frames_from_split(
                split, 
                max_videos=num_videos_per_split,
                max_frames_per_video=max_frames_per_video
            )
            total_frames += frames
        
        print(f"\nSample extraction complete. Total frames: {total_frames}")
        return total_frames

def main():
    parser = argparse.ArgumentParser(description="Extract frames from pothole videos")
    parser.add_argument("--split", choices=["train", "val", "test", "all", "sample"], 
                       default="sample", help="Which split to process")
    parser.add_argument("--max-videos", type=int, help="Maximum number of videos to process")
    parser.add_argument("--max-frames", type=int, help="Maximum frames per video")
    
    args = parser.parse_args()
    
    extractor = FrameExtractor()
    
    if args.split == "sample":
        extractor.extract_sample_frames()
    elif args.split == "all":
        for split in ["train", "val", "test"]:
            extractor.extract_frames_from_split(split, args.max_videos, args.max_frames)
    else:
        extractor.extract_frames_from_split(args.split, args.max_videos, args.max_frames)

if __name__ == "__main__":
    main()
