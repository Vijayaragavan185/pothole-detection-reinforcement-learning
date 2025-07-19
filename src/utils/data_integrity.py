import sys
from pathlib import Path
sys.path.append('.')
from configs.config import PATHS

print('=== TRAIN SPLIT DIAGNOSTIC ===')
print(f'Raw videos path: {PATHS[\"raw_videos\"]}')
print(f'Ground truth path: {PATHS[\"ground_truth\"]}')

# Check directories
train_processed = PATHS['processed_frames'] / 'train'
train_gt = PATHS['ground_truth'] / 'train'

print(f'Processed frames exist: {train_processed.exists()}')
print(f'Ground truth exist: {train_gt.exists()}')

if train_processed.exists():
    video_dirs = [d for d in train_processed.iterdir() if d.is_dir()]
    print(f'Processed video dirs: {len(video_dirs)}')

if train_gt.exists():
    gt_dirs = [d for d in train_gt.iterdir() if d.is_dir()]
    print(f'Ground truth dirs: {len(gt_dirs)}')
else:
    print('❌ Ground truth missing - run mask_processor.py first!')