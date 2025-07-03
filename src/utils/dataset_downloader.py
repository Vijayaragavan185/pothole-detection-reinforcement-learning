import os
import requests
from pathlib import Path
import zipfile
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import PATHS, DATASET_CONFIG

class MendeleyDatasetDownloader:
    def __init__(self):
        # Correct Mendeley Data URL
        self.dataset_url = "https://data.mendeley.com/datasets/5bwfg4v4cd"
        self.doi = "10.17632/5bwfg4v4cd.3"
        self.version = "3"
        
    def download_dataset(self):
        """Download complete dataset"""
        print("Starting Pothole Video Dataset Download...")
        print(f"Dataset: {DATASET_CONFIG['name']}")
        print(f"DOI: {self.doi}")
        print(f"Version: {self.version}")
        print(f"Total videos: {DATASET_CONFIG['total_videos']}")
        
        # Create download directory
        download_dir = PATHS["raw_videos"].parent / "downloads"
        download_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("MANUAL DOWNLOAD REQUIRED")
        print("="*60)
        print("Please download the dataset manually from:")
        print(f"🔗 {self.dataset_url}")
        print(f"📄 DOI: {self.doi}")
        
        print("\nDataset Information:")
        print("• 619 high-resolution videos (1080×1080)")
        print("• 48 frames per video (2 seconds each)")
        print("• Train/Val/Test: 372/124/123 videos")
        print("• RGB videos + ground truth masks")
        
        print("\nDownload Instructions:")
        print("1. Click 'Download all files' on the Mendeley page")
        print("2. Extract all ZIP files")
        print("3. Place extracted folders in the downloads directory")
        
        print(f"\nSave all files to: {download_dir}")
        print("Then run: python src/utils/extract_dataset.py")
        
        return download_dir

if __name__ == "__main__":
    downloader = MendeleyDatasetDownloader()
    download_path = downloader.download_dataset()
    print(f"\nDownload directory: {download_path}")
