import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import PATHS, VIDEO_CONFIG
import json
from tqdm import tqdm
from collections import defaultdict

class SequenceOptimizer:
    """Optimize video sequences for RL training efficiency"""
    
    def __init__(self):
        self.sequence_length = VIDEO_CONFIG["sequence_length"]
        self.overlap = VIDEO_CONFIG["overlap"]
        
    def calculate_sequence_quality(self, sequence):
        """Calculate quality metrics for a sequence"""
        # Motion variation (temporal consistency)
        motion_scores = []
        for i in range(1, len(sequence)):
            frame_diff = np.mean(np.abs(sequence[i] - sequence[i-1]))
            motion_scores.append(frame_diff)
        
        motion_variation = np.std(motion_scores) if motion_scores else 0
        
        # Brightness variation
        brightness_scores = [np.mean(frame) for frame in sequence]
        brightness_variation = np.std(brightness_scores)
        
        # Contrast quality
        contrast_scores = [np.std(frame) for frame in sequence]
        avg_contrast = np.mean(contrast_scores)
        
        # Overall quality score
        quality_score = (
            0.4 * (1 - motion_variation) +  # Prefer stable motion
            0.3 * avg_contrast +            # Prefer good contrast
            0.3 * (1 - brightness_variation) # Prefer consistent brightness
        )
        
        return {
            "quality_score": float(quality_score),
            "motion_variation": float(motion_variation),
            "brightness_variation": float(brightness_variation),
            "avg_contrast": float(avg_contrast),
            "motion_scores": [float(s) for s in motion_scores]
        }
    
    # def filter_sequences_by_quality(self, sequences, quality_threshold=0.1):
    #     """Filter sequences based on quality metrics"""
    #     quality_sequences = []
    #     quality_metrics = []
        
    #     for sequence in sequences:
    #         metrics = self.calculate_sequence_quality(sequence)
            
    #         if metrics["quality_score"] >= quality_threshold:
    #             quality_sequences.append(sequence)
    #             quality_metrics.append(metrics)
        
    #     return quality_sequences, quality_metrics
    def filter_sequences_by_quality(self, sequences, quality_threshold=0.1):
        """Filter sequences based on quality metrics"""
        quality_sequences = []
        quality_metrics = []
        
        # Add debugging
        quality_scores = []
        
        for sequence in sequences:
            metrics = self.calculate_sequence_quality(sequence)
            quality_scores.append(metrics["quality_score"])
            
            if metrics["quality_score"] >= quality_threshold:
                quality_sequences.append(sequence)
                quality_metrics.append(metrics)
        
        # Debug output
        if quality_scores:
            print(f"🔍 Quality scores - Min: {min(quality_scores):.4f}, Max: {max(quality_scores):.4f}, Avg: {sum(quality_scores)/len(quality_scores):.4f}")
            print(f"🔍 Sequences above threshold {quality_threshold}: {len(quality_sequences)}/{len(sequences)}")
        
        return quality_sequences, quality_metrics
    
    def balance_dataset(self, sequences_with_masks, target_ratio=0.5):
        """Balance sequences with and without potholes"""
        pothole_sequences = []
        no_pothole_sequences = []
        
        for seq_data in sequences_with_masks:
            sequence, mask_sequence, info = seq_data
            
            # Check if sequence contains potholes
            has_pothole = info.get("pothole_frames", 0) > 0
            
            if has_pothole:
                pothole_sequences.append(seq_data)
            else:
                no_pothole_sequences.append(seq_data)
        
        # Balance the dataset
        min_count = min(len(pothole_sequences), len(no_pothole_sequences))
        target_pothole_count = int(min_count / target_ratio)
        target_no_pothole_count = int(min_count / (1 - target_ratio))
        
        # Sample sequences to achieve balance
        balanced_sequences = []
        
        if len(pothole_sequences) >= target_pothole_count:
            balanced_sequences.extend(pothole_sequences[:target_pothole_count])
        else:
            balanced_sequences.extend(pothole_sequences)
        
        if len(no_pothole_sequences) >= target_no_pothole_count:
            balanced_sequences.extend(no_pothole_sequences[:target_no_pothole_count])
        else:
            balanced_sequences.extend(no_pothole_sequences)
        
        return balanced_sequences
    
    def create_training_batches(self, sequences, batch_size=32):
        """Create optimized training batches"""
        # Sort sequences by quality score
        sequence_qualities = []
        for i, sequence in enumerate(sequences):
            quality = self.calculate_sequence_quality(sequence)
            sequence_qualities.append((i, quality["quality_score"]))
        
        # Sort by quality (best first)
        sequence_qualities.sort(key=lambda x: x[1], reverse=True)
        
        # Create batches with mixed quality levels
        batches = []
        high_quality_indices = [idx for idx, score in sequence_qualities[:len(sequences)//2]]
        low_quality_indices = [idx for idx, score in sequence_qualities[len(sequences)//2:]]
        
        # Mix high and low quality sequences in each batch
        for i in range(0, len(sequences), batch_size):
            batch_indices = []
            
            # Add high quality sequences
            high_start = (i // batch_size) * (batch_size // 2)
            high_end = min(high_start + batch_size // 2, len(high_quality_indices))
            batch_indices.extend(high_quality_indices[high_start:high_end])
            
            # Add low quality sequences
            low_start = (i // batch_size) * (batch_size // 2)
            low_end = min(low_start + batch_size // 2, len(low_quality_indices))
            batch_indices.extend(low_quality_indices[low_start:low_end])
            
            if batch_indices:
                batch_sequences = [sequences[idx] for idx in batch_indices]
                batches.append(batch_sequences)
        
        return batches
    
    def optimize_split_sequences(self, split_name):
        """Optimize sequences for a specific split"""
        print(f"\nOptimizing sequences for {split_name} split...")
        
        processed_dir = PATHS["processed_frames"] / split_name
        ground_truth_dir = PATHS["ground_truth"] / split_name
        output_dir = PATHS["processed_frames"] / f"{split_name}_optimized"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not processed_dir.exists():
            print(f"Processed directory {processed_dir} does not exist")
            return 0
        
        video_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
        optimized_count = 0
        
        for video_dir in tqdm(video_dirs, desc=f"Optimizing {split_name}"):
            sequences_dir = video_dir / "sequences"
            if not sequences_dir.exists():
                continue
            
            # Load video sequences
            sequence_files = list(sequences_dir.glob("sequence_*.npy"))
            sequences = []
            
            for seq_file in sequence_files:
                try:
                    sequence = np.load(seq_file)
                    if sequence.size > 0:  # Valid sequence
                        sequences.append(sequence)
                except (ValueError, OSError) as e:
                    print(f"⚠️  Skipping corrupted file: {seq_file.name}")
                    continue
            
            if not sequences:
                continue
            
            # Filter by quality
            quality_sequences, quality_metrics = self.filter_sequences_by_quality(
                sequences, quality_threshold=0.1
            )
            
            # Load corresponding masks if available
            mask_dir = ground_truth_dir / video_dir.name / "sequences"
            sequences_with_masks = []
            
            if mask_dir.exists():
                for i, sequence in enumerate(quality_sequences):
                    mask_file = mask_dir / f"mask_sequence_{i:03d}.npy"
                    if mask_file.exists():
                        mask_sequence = np.load(mask_file)
                        
                        # Load sequence info
                        info_file = ground_truth_dir / video_dir.name / "sequence_info.json"
                        if info_file.exists():
                            with open(info_file, 'r') as f:
                                info_data = json.load(f)
                            info = info_data[i] if i < len(info_data) else {}
                        else:
                            info = {}
                        
                        sequences_with_masks.append((sequence, mask_sequence, info))
                    else:
                        # No mask available
                        sequences_with_masks.append((sequence, None, {}))
            else:
                # No masks available
                for sequence in quality_sequences:
                    sequences_with_masks.append((sequence, None, {}))
            
            # Balance dataset (only for train split)
            if split_name == "train":
                # balanced_sequences = self.balance_dataset(sequences_with_masks)  # Comment out
                balanced_sequences = sequences_with_masks  # Use all sequences
                print(f"🎯 Bypassing balance for train: {len(balanced_sequences)} sequences")
            else:
                balanced_sequences = sequences_with_masks
            
            # Save optimized sequences
            optimized_video_dir = output_dir / video_dir.name
            optimized_sequences_dir = optimized_video_dir / "sequences"
            optimized_sequences_dir.mkdir(parents=True, exist_ok=True)
            
            optimization_stats = {
                "original_sequences": len(sequences),
                "quality_filtered": len(quality_sequences),
                "final_optimized": len(balanced_sequences),
                "quality_threshold": 0.3,
                "split_name": split_name
            }
            
            for i, (sequence, mask_sequence, info) in enumerate(balanced_sequences):
                # Save optimized sequence
                opt_seq_path = optimized_sequences_dir / f"optimized_sequence_{i:03d}.npy"
                np.save(opt_seq_path, sequence)
                
                # Save corresponding mask if available
                if mask_sequence is not None:
                    opt_mask_path = optimized_sequences_dir / f"optimized_mask_{i:03d}.npy"
                    np.save(opt_mask_path, mask_sequence)
                
                optimized_count += 1
            
            # Save optimization statistics
            stats_path = optimized_video_dir / "optimization_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(optimization_stats, f, indent=2)
        
        print(f"Optimized {optimized_count} sequences for {split_name}")
        return optimized_count
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization report"""
        print("="*50)
        print("SEQUENCE OPTIMIZATION REPORT")
        print("="*50)
        
        splits = ["train", "val", "test"]
        total_stats = defaultdict(int)
        
        for split in splits:
            optimized_count = self.optimize_split_sequences(split)
            total_stats[split] = optimized_count
        
        print(f"\nOptimization Summary:")
        for split in splits:
            print(f"  {split}: {total_stats[split]} optimized sequences")
        
        print(f"  Total optimized sequences: {sum(total_stats.values())}")
        
        return total_stats

if __name__ == "__main__":
    optimizer = SequenceOptimizer()
    optimizer.generate_optimization_report()
