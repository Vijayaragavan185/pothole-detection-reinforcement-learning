import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import PATHS
import json
from collections import defaultdict

class VideoAnalyzer:
    """Analyze video dataset properties and statistics"""
    
    def __init__(self):
        self.stats = defaultdict(list)
        
    def analyze_video_properties(self, video_path):
        """Analyze properties of a single video"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return None
        
        properties = {
            "filename": video_path.name,
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": 0,
            "avg_brightness": 0,
            "motion_score": 0
        }
        
        if properties["fps"] > 0:
            properties["duration"] = properties["frame_count"] / properties["fps"]
        
        # Analyze first few frames for quality metrics
        frame_brightnesses = []
        prev_frame = None
        motion_scores = []
        
        for i in range(min(10, properties["frame_count"])):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Brightness analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            frame_brightnesses.append(brightness)
            
            # Motion analysis (simple frame difference)
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                motion = np.mean(diff)
                motion_scores.append(motion)
            
            prev_frame = gray
        
        if frame_brightnesses:
            properties["avg_brightness"] = np.mean(frame_brightnesses)
        if motion_scores:
            properties["motion_score"] = np.mean(motion_scores)
        
        cap.release()
        return properties
    
    def analyze_split(self, split_name):
        """Analyze all videos in a split"""
        print(f"\nAnalyzing {split_name} split...")
        
        rgb_dir = PATHS["raw_videos"] / split_name / "rgb"
        if not rgb_dir.exists():
            print(f"Directory {rgb_dir} does not exist")
            return {}
        
        video_files = list(rgb_dir.glob("*.mp4")) + list(rgb_dir.glob("*.avi"))
        
        split_stats = {
            "total_videos": len(video_files),
            "properties": [],
            "summary": {}
        }
        
        for video_file in video_files[:20]:  # Analyze first 20 videos
            properties = self.analyze_video_properties(video_file)
            if properties:
                split_stats["properties"].append(properties)
        
        # Calculate summary statistics
        if split_stats["properties"]:
            props = split_stats["properties"]
            split_stats["summary"] = {
                "avg_frame_count": np.mean([p["frame_count"] for p in props]),
                "avg_fps": np.mean([p["fps"] for p in props]),
                "avg_duration": np.mean([p["duration"] for p in props]),
                "avg_width": np.mean([p["width"] for p in props]),
                "avg_height": np.mean([p["height"] for p in props]),
                "avg_brightness": np.mean([p["avg_brightness"] for p in props]),
                "avg_motion": np.mean([p["motion_score"] for p in props])
            }
        
        return split_stats
    
    def generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        print("="*50)
        print("VIDEO DATASET ANALYSIS REPORT")
        print("="*50)
        
        all_stats = {}
        splits = ["train", "val", "test"]
        
        for split in splits:
            all_stats[split] = self.analyze_split(split)
            
            if all_stats[split]["properties"]:
                summary = all_stats[split]["summary"]
                print(f"\n{split.upper()} SPLIT ANALYSIS:")
                print(f"  Videos analyzed: {len(all_stats[split]['properties'])}")
                print(f"  Avg frame count: {summary['avg_frame_count']:.1f}")
                print(f"  Avg FPS: {summary['avg_fps']:.1f}")
                print(f"  Avg duration: {summary['avg_duration']:.1f}s")
                print(f"  Avg resolution: {summary['avg_width']:.0f}x{summary['avg_height']:.0f}")
                print(f"  Avg brightness: {summary['avg_brightness']:.1f}")
                print(f"  Avg motion score: {summary['avg_motion']:.1f}")
        
        # Save detailed report
        report_path = PATHS["raw_videos"].parent / "analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(all_stats, f, indent=2, default=str)
        
        print(f"\nDetailed report saved to: {report_path}")
        return all_stats
    
    def create_visualizations(self):
        """Create visualization plots"""
        print("\nCreating visualization plots...")
        
        # This would create plots - placeholder for now
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Video Dataset Analysis")
        
        # Placeholder plots
        axes[0,0].set_title("Frame Count Distribution")
        axes[0,1].set_title("Duration Distribution")
        axes[1,0].set_title("Resolution Distribution")
        axes[1,1].set_title("Brightness Distribution")
        
        plt.tight_layout()
        
        # Save plot
        plot_path = PATHS["raw_videos"].parent / "analysis_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plots saved to: {plot_path}")
        plt.close()

if __name__ == "__main__":
    analyzer = VideoAnalyzer()
    analyzer.generate_analysis_report()
    analyzer.create_visualizations()
