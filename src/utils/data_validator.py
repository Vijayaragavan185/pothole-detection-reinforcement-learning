import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import PATHS, DATASET_CONFIG
import json
from collections import defaultdict

class DatasetValidator:
    """Validate dataset integrity and structure"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov']
        
    def validate_video_file(self, video_path):
        """Validate individual video file"""
        if not video_path.exists():
            return False, "File does not exist"
        
        if video_path.suffix.lower() not in self.supported_formats:
            return False, f"Unsupported format: {video_path.suffix}"
        
        # Try to open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, "Cannot open video file"
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        # Validate properties
        if frame_count < 10:  # Minimum frames
            return False, f"Too few frames: {frame_count}"
        
        if width < 100 or height < 100:  # Minimum resolution
            return False, f"Resolution too low: {width}x{height}"
        
        return True, {
            "frame_count": frame_count,
            "fps": fps,
            "resolution": (width, height),
            "duration": frame_count / fps if fps > 0 else 0
        }
    
    def validate_dataset_structure(self):
        """Validate overall dataset structure"""
        print("Validating dataset structure...")
        
        issues = []
        
        # Check main directories exist
        if not PATHS["raw_videos"].exists():
            issues.append("raw_videos directory missing")
            return issues
        
        # Check split directories
        splits = ["train", "val", "test"]
        for split in splits:
            split_dir = PATHS["raw_videos"] / split
            if not split_dir.exists():
                issues.append(f"{split} directory missing")
                continue
            
            # Check rgb and mask subdirectories
            rgb_dir = split_dir / "rgb"
            mask_dir = split_dir / "mask"
            
            if not rgb_dir.exists():
                issues.append(f"{split}/rgb directory missing")
            if not mask_dir.exists():
                issues.append(f"{split}/mask directory missing")
        
        return issues
    
    def validate_split_data(self, split_name):
        """Validate data in a specific split"""
        print(f"\nValidating {split_name} split...")
        
        rgb_dir = PATHS["raw_videos"] / split_name / "rgb"
        mask_dir = PATHS["raw_videos"] / split_name / "mask"
        
        if not rgb_dir.exists() or not mask_dir.exists():
            print(f"Directories missing for {split_name}")
            return {}
        
        # Get video files
        rgb_files = list(rgb_dir.glob("*.mp4")) + list(rgb_dir.glob("*.avi"))
        mask_files = list(mask_dir.glob("*.mp4")) + list(mask_dir.glob("*.avi"))
        
        stats = {
            "total_rgb_videos": len(rgb_files),
            "total_mask_videos": len(mask_files),
            "valid_videos": 0,
            "invalid_videos": 0,
            "issues": [],
            "video_properties": []
        }
        
        # Validate each RGB video
        for video_file in rgb_files:
            is_valid, result = self.validate_video_file(video_file)
            
            if is_valid:
                stats["valid_videos"] += 1
                stats["video_properties"].append({
                    "filename": video_file.name,
                    **result
                })
            else:
                stats["invalid_videos"] += 1
                stats["issues"].append(f"{video_file.name}: {result}")
        
        # Check for matching mask files
        rgb_names = {f.stem for f in rgb_files}
        mask_names = {f.stem for f in mask_files}
        
        missing_masks = rgb_names - mask_names
        extra_masks = mask_names - rgb_names
        
        if missing_masks:
            stats["issues"].append(f"Missing masks for: {missing_masks}")
        if extra_masks:
            stats["issues"].append(f"Extra masks without RGB: {extra_masks}")
        
        return stats
    
    def validate_complete_dataset(self):
        """Validate entire dataset"""
        print("="*50)
        print("DATASET VALIDATION REPORT")
        print("="*50)
        
        # Structure validation
        structure_issues = self.validate_dataset_structure()
        if structure_issues:
            print("STRUCTURE ISSUES:")
            for issue in structure_issues:
                print(f"  ❌ {issue}")
            return
        else:
            print("✅ Dataset structure is valid")
        
        # Split validation
        splits = ["train", "val", "test"]
        total_stats = defaultdict(int)
        
        for split in splits:
            stats = self.validate_split_data(split)
            
            print(f"\n{split.upper()} SPLIT:")
            print(f"  RGB videos: {stats.get('total_rgb_videos', 0)}")
            print(f"  Mask videos: {stats.get('total_mask_videos', 0)}")
            print(f"  Valid videos: {stats.get('valid_videos', 0)}")
            print(f"  Invalid videos: {stats.get('invalid_videos', 0)}")
            
            if stats.get('issues'):
                print(f"  Issues:")
                for issue in stats['issues'][:5]:  # Show first 5 issues
                    print(f"    ❌ {issue}")
                if len(stats['issues']) > 5:
                    print(f"    ... and {len(stats['issues']) - 5} more issues")
            
            # Aggregate stats
            for key in ['total_rgb_videos', 'valid_videos', 'invalid_videos']:
                total_stats[key] += stats.get(key, 0)
        
        # Summary
        print(f"\nOVERALL SUMMARY:")
        print(f"  Total RGB videos: {total_stats['total_rgb_videos']}")
        print(f"  Valid videos: {total_stats['valid_videos']}")
        print(f"  Invalid videos: {total_stats['invalid_videos']}")
        
        expected_total = sum(DATASET_CONFIG["splits"].values())
        print(f"  Expected total: {expected_total}")
        
        if total_stats['total_rgb_videos'] == expected_total:
            print("  ✅ Video count matches expected")
        else:
            print(f"  ❌ Video count mismatch")
        
        # Validation score
        if total_stats['total_rgb_videos'] > 0:
            success_rate = total_stats['valid_videos'] / total_stats['total_rgb_videos'] * 100
            print(f"  Validation success rate: {success_rate:.1f}%")
            
            if success_rate >= 95:
                print("  ✅ Dataset is ready for processing!")
            elif success_rate >= 80:
                print("  ⚠️  Dataset has some issues but is usable")
            else:
                print("  ❌ Dataset has significant issues")
        
        return total_stats

if __name__ == "__main__":
    validator = DatasetValidator()
    validator.validate_complete_dataset()
