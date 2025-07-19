import numpy as np
from pathlib import Path
train_dir = Path('data/processed_frames/train')
valid_sequences = 0
corrupted_sequences = 0
for seq_file in train_dir.rglob('sequence_*.npy'):
    try:
        seq = np.load(seq_file)
        if seq.size > 0:
            valid_sequences += 1
        else:
            corrupted_sequences += 1
    except:
        corrupted_sequences += 1
print(f'Valid sequences: {valid_sequences}')
print(f'Corrupted sequences: {corrupted_sequences}')