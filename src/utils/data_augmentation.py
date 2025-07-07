import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import VIDEO_CONFIG, PATHS
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

class VideoAugmentation:
    """Advanced data augmentation for video sequences"""
    
    def __init__(self, augment_probability=0.5):
        self.augment_prob = augment_probability
        self.input_size = VIDEO_CONFIG["input_size"]
        
        # Define augmentation pipeline
        self.spatial_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            A.CLAHE(clip_limit=2, tile_grid_size=(8, 8), p=0.3),
            A.RandomRotate90(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
        ])
        
        # Temporal augmentations
        self.temporal_transforms = {
            'time_reverse': 0.1,
            'frame_dropout': 0.1,
            'temporal_noise': 0.2
        }
    
    def apply_spatial_augmentation(self, frame):
        """Apply spatial augmentations to a single frame"""
        if random.random() < self.augment_prob:
            # Convert to uint8 for albumentations
            frame_uint8 = (frame * 255).astype(np.uint8)
            
            # Apply augmentations
            augmented = self.spatial_transforms(image=frame_uint8)
            
            # Convert back to float32
            return augmented['image'].astype(np.float32) / 255.0
        
        return frame
    
    def apply_temporal_augmentation(self, sequence):
        """Apply temporal augmentations to video sequence"""
        augmented_sequence = sequence.copy()
        
        # Time reversal
        if random.random() < self.temporal_transforms['time_reverse']:
            augmented_sequence = augmented_sequence[::-1]
        
        # Frame dropout and duplication
        if random.random() < self.temporal_transforms['frame_dropout']:
            augmented_sequence = self._frame_dropout(augmented_sequence)
        
        # Temporal noise
        if random.random() < self.temporal_transforms['temporal_noise']:
            augmented_sequence = self._add_temporal_noise(augmented_sequence)
        
        return augmented_sequence
    
    def _frame_dropout(self, sequence):
        """Randomly drop and duplicate frames"""
        if len(sequence) < 3:
            return sequence
        
        # Drop one random frame and duplicate another
        drop_idx = random.randint(1, len(sequence) - 2)  # Don't drop first/last
        dup_idx = random.randint(0, len(sequence) - 1)
        
        new_sequence = []
        for i, frame in enumerate(sequence):
            if i == drop_idx:
                continue  # Skip this frame
            new_sequence.append(frame)
            if i == dup_idx:
                new_sequence.append(frame)  # Duplicate this frame
        
        return np.array(new_sequence)
    
    def _add_temporal_noise(self, sequence):
        """Add subtle temporal noise"""
        noise_intensity = random.uniform(0.01, 0.05)
        noise = np.random.normal(0, noise_intensity, sequence.shape)
        return np.clip(sequence + noise, 0, 1)
    
    def augment_sequence(self, frames, masks=None):
        """Apply both spatial and temporal augmentations"""
        # Apply temporal augmentation first
        augmented_frames = self.apply_temporal_augmentation(frames)
        
        # Apply spatial augmentation to each frame
        final_frames = []
        final_masks = []
        
        for i, frame in enumerate(augmented_frames):
            aug_frame = self.apply_spatial_augmentation(frame)
            final_frames.append(aug_frame)
            
            # Apply same spatial transform to mask if provided
            if masks is not None and i < len(masks):
                if random.random() < self.augment_prob:
                    mask_uint8 = (masks[i] * 255).astype(np.uint8)
                    aug_mask = self.spatial_transforms(image=mask_uint8)['image']
                    final_masks.append(aug_mask.astype(np.float32) / 255.0)
                else:
                    final_masks.append(masks[i])
        
        if masks is not None:
            return np.array(final_frames), np.array(final_masks)
        
        return np.array(final_frames)
    
    def create_augmented_dataset(self, split_name, augmentation_factor=2):
        """Create augmented versions of dataset split"""
        print(f"\nCreating augmented dataset for {split_name} split...")
        
        # Input and output paths
        input_dir = PATHS["processed_frames"] / split_name
        output_dir = PATHS["processed_frames"] / f"{split_name}_augmented"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not input_dir.exists():
            print(f"Input directory {input_dir} does not exist")
            return 0
        
        video_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        augmented_count = 0
        
        for video_dir in video_dirs:
            sequences_dir = video_dir / "sequences"
            if not sequences_dir.exists():
                continue
            
            # Load original sequences
            sequence_files = list(sequences_dir.glob("sequence_*.npy"))
            
            for seq_file in sequence_files:
                original_sequence = np.load(seq_file)
                
                # Create multiple augmented versions
                for aug_idx in range(augmentation_factor):
                    augmented_seq = self.augment_sequence(original_sequence)
                    
                    # Save augmented sequence
                    aug_video_dir = output_dir / f"{video_dir.name}_aug{aug_idx}"
                    aug_sequences_dir = aug_video_dir / "sequences"
                    aug_sequences_dir.mkdir(parents=True, exist_ok=True)
                    
                    aug_file_path = aug_sequences_dir / f"{seq_file.stem}_aug{aug_idx}.npy"
                    np.save(aug_file_path, augmented_seq)
                    augmented_count += 1
        
        print(f"Created {augmented_count} augmented sequences for {split_name}")
        return augmented_count

if __name__ == "__main__":
    augmenter = VideoAugmentation()
    
    # Test on sample data
    for split in ["train", "val"]:  # Don't augment test set
        augmenter.create_augmented_dataset(split, augmentation_factor=2)
