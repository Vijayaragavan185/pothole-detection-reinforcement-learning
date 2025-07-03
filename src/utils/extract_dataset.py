import zipfile
import shutil
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import PATHS

def extract_downloaded_dataset():
    """Extract manually downloaded dataset files"""
    download_dir = PATHS["raw_videos"].parent / "downloads"
    
    if not download_dir.exists():
        print(f"Downloads directory not found: {download_dir}")
        print("Please download dataset first using dataset_downloader.py")
        return
    
    # Check for zip files
    zip_files = list(download_dir.glob("*.zip"))
    if not zip_files:
        print("No zip files found in downloads directory")
        print(f"Expected location: {download_dir}")
        return
    
    print(f"Found {len(zip_files)} zip files")
    
    # Extract each zip file
    for zip_file in zip_files:
        print(f"\nExtracting {zip_file.name}...")
        
        if "train" in zip_file.name.lower():
            extract_to = PATHS["raw_videos"] / "train"
        elif "val" in zip_file.name.lower():
            extract_to = PATHS["raw_videos"] / "val"  
        elif "test" in zip_file.name.lower():
            extract_to = PATHS["raw_videos"] / "test"
        elif "ground_truth" in zip_file.name.lower():
            extract_to = PATHS["ground_truth"]
        else:
            extract_to = PATHS["raw_videos"] / zip_file.stem
        
        extract_to.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"✓ Extracted to {extract_to}")
        except Exception as e:
            print(f"✗ Failed to extract {zip_file.name}: {e}")
    
    print("\n" + "="*50)
    print("EXTRACTION COMPLETE")
    print("="*50)
    print("Verify your data structure:")
    print(f"Raw videos: {PATHS['raw_videos']}")
    print(f"Ground truth: {PATHS['ground_truth']}")

if __name__ == "__main__":
    extract_downloaded_dataset()
