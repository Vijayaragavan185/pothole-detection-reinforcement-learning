import numpy as np
from pathlib import Path
data_dir = Path('data/processed_frames/train_optimized')
if data_dir.exists():
    seq_files = list(data_dir.rglob('*.npy'))
    print(f'Found {len(seq_files)} sequence files')
    if seq_files:
        sample = np.load(seq_files[0])
        print(f'Sample shape: {sample.shape}')
        print(f'Sample size: {sample.size} elements')
        print(f'File size: {seq_files[0].stat().st_size / 1024 / 1024:.1f} MB')
else:
    print('❌ Data directory not found')