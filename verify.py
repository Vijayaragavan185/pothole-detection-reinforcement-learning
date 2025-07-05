import os

def count_files(directory, extension='.mp4'):
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count

base_dir = "data/raw_videos"
print(f"Train videos: {count_files(os.path.join(base_dir, 'train/rgb'))}")
print(f"Validation videos: {count_files(os.path.join(base_dir, 'val/rgb'))}")
print(f"Test videos: {count_files(os.path.join(base_dir, 'test/rgb'))}")
