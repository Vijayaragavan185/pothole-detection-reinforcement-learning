import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set backend before importing pyplot
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
    
    def create_visualizations(self, all_stats):
        """Create visualization plots with actual data"""
        print("\nCreating visualization plots...")
        
        # Extract data from all_stats
        splits = ['train', 'val', 'test']
        
        # Prepare data
        brightness_data = []
        motion_data = []
        frame_counts = []
        durations = []
        
        for split in splits:
            if split in all_stats and 'properties' in all_stats[split]:
                props = all_stats[split]['properties']
                brightness_data.extend([p['avg_brightness'] for p in props])
                motion_data.extend([p['motion_score'] for p in props])
                frame_counts.extend([p['frame_count'] for p in props])
                durations.extend([p['duration'] for p in props])
        
        # Create figure with actual data
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Pothole Video Dataset Analysis", fontsize=16, fontweight='bold')
        
        # Plot 1: Frame Count Distribution
        if frame_counts:
            axes[0,0].hist(frame_counts, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0,0].set_title(f"Frame Count Distribution\n(Mean: {np.mean(frame_counts):.1f})")
            axes[0,0].set_xlabel("Frame Count")
            axes[0,0].set_ylabel("Frequency")
        
        # Plot 2: Duration Distribution  
        if durations:
            axes[0,1].hist(durations, bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[0,1].set_title(f"Duration Distribution\n(Mean: {np.mean(durations):.2f}s)")
            axes[0,1].set_xlabel("Duration (seconds)")
            axes[0,1].set_ylabel("Frequency")
        
        # Plot 3: Brightness by Split
        brightness_by_split = []
        split_labels = []
        for split in splits:
            if split in all_stats and 'properties' in all_stats[split]:
                props = all_stats[split]['properties']
                split_brightness = [p['avg_brightness'] for p in props]
                if split_brightness:
                    brightness_by_split.append(split_brightness)
                    split_labels.append(split.capitalize())
        
        if brightness_by_split:
            axes[1,0].boxplot(brightness_by_split, labels=split_labels)
            axes[1,0].set_title("Brightness Distribution by Split")
            axes[1,0].set_ylabel("Average Brightness")
        
        # Plot 4: Motion Score by Split
        motion_by_split = []
        for split in splits:
            if split in all_stats and 'properties' in all_stats[split]:
                props = all_stats[split]['properties']
                split_motion = [p['motion_score'] for p in props]
                if split_motion:
                    motion_by_split.append(split_motion)
        
        if motion_by_split:
            axes[1,1].boxplot(motion_by_split, labels=split_labels)
            axes[1,1].set_title("Motion Score Distribution by Split")
            axes[1,1].set_ylabel("Motion Score")
        
        plt.tight_layout()
        
        # Save plot
        plot_path = PATHS["raw_videos"].parent / "analysis_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Plots saved to: {plot_path}")
        
        # Try to display
        plt.show()
        
        return fig
    
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
        
        # Create visualizations with data
        self.create_visualizations(all_stats)
        
        return all_stats

if __name__ == "__main__":
    analyzer = VideoAnalyzer()
    analyzer.generate_analysis_report()
