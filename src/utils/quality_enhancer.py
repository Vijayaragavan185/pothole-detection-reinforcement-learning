import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import PATHS, VIDEO_CONFIG
from tqdm import tqdm
from skimage import restoration, filters, exposure
import json

class VideoQualityEnhancer:
    """Enhance video quality for better RL training"""
    
    def __init__(self):
        self.input_size = VIDEO_CONFIG["input_size"]
        
    def denoise_frame(self, frame):
        """Remove noise from video frame"""
        # Non-local means denoising
        if len(frame.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(
                (frame * 255).astype(np.uint8), None, 10, 10, 7, 21
            )
        else:
            denoised = cv2.fastNlMeansDenoising(
                (frame * 255).astype(np.uint8), None, 10, 7, 21
            )
        
        return denoised.astype(np.float32) / 255.0
    
    def enhance_contrast(self, frame):
        """Enhance frame contrast using CLAHE"""
        frame_uint8 = (frame * 255).astype(np.uint8)
        
        if len(frame.shape) == 3:
            # Convert to LAB for better contrast enhancement
            lab = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(frame_uint8)
        
        return enhanced.astype(np.float32) / 255.0
    
    def adjust_brightness_contrast(self, frame, target_brightness=0.5):
        """Automatically adjust brightness and contrast"""
        # Calculate current brightness
        if len(frame.shape) == 3:
            current_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        else:
            current_brightness = np.mean(frame)
        
        # Adjust brightness
        brightness_factor = target_brightness / (current_brightness + 1e-6)
        brightness_factor = np.clip(brightness_factor, 0.5, 2.0)
        
        adjusted = frame * brightness_factor
        
        # Enhance contrast
        adjusted = exposure.rescale_intensity(adjusted, out_range=(0, 1))
        
        return adjusted.astype(np.float32)
    
    def sharpen_frame(self, frame):
        """Apply unsharp masking for better edge definition"""
        # Convert to uint8 for OpenCV operations
        frame_uint8 = (frame * 255).astype(np.uint8)
        
        # Create unsharp mask
        gaussian = cv2.GaussianBlur(frame_uint8, (0, 0), 1.0)
        unsharp_mask = cv2.addWeighted(frame_uint8, 1.5, gaussian, -0.5, 0)
        
        return np.clip(unsharp_mask.astype(np.float32) / 255.0, 0, 1)
    
    def stabilize_sequence(self, sequence):
        """Apply basic video stabilization to sequence"""
        if len(sequence) < 2:
            return sequence
        
        stabilized = [sequence[0]]  # First frame as reference
        
        for i in range(1, len(sequence)):
            # Convert frames to grayscale for motion estimation
            prev_gray = cv2.cvtColor((sequence[i-1] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor((sequence[i] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Estimate motion using optical flow
            try:
                # Calculate sparse optical flow
                corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, 
                                                qualityLevel=0.01, minDistance=10)
                
                if corners is not None and len(corners) > 10:
                    # Track corners
                    new_corners, status, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray, curr_gray, corners, None
                    )
                    
                    # Filter good corners
                    good_old = corners[status == 1]
                    good_new = new_corners[status == 1]
                    
                    if len(good_old) > 10:
                        # Estimate transformation matrix
                        transform = cv2.estimateAffinePartial2D(good_old, good_new)[0]
                        
                        if transform is not None:
                            # Apply stabilization transform
                            h, w = sequence[i].shape[:2]
                            stabilized_frame = cv2.warpAffine(
                                (sequence[i] * 255).astype(np.uint8), 
                                transform, (w, h)
                            )
                            stabilized.append(stabilized_frame.astype(np.float32) / 255.0)
                        else:
                            stabilized.append(sequence[i])
                    else:
                        stabilized.append(sequence[i])
                else:
                    stabilized.append(sequence[i])
                    
            except Exception as e:
                # If stabilization fails, use original frame
                stabilized.append(sequence[i])
        
        return np.array(stabilized)
    
    def enhance_sequence(self, sequence, enhancement_level="medium"):
        """Apply comprehensive enhancement to video sequence"""
        enhanced_frames = []
        
        enhancement_configs = {
            "light": {
                "denoise": False,
                "contrast": True,
                "brightness": True,
                "sharpen": False,
                "stabilize": False
            },
            "medium": {
                "denoise": True,
                "contrast": True,
                "brightness": True,
                "sharpen": True,
                "stabilize": False
            },
            "heavy": {
                "denoise": True,
                "contrast": True,
                "brightness": True,
                "sharpen": True,
                "stabilize": True
            }
        }
        
        config = enhancement_configs.get(enhancement_level, enhancement_configs["medium"])
        
        # Apply frame-wise enhancements
        for frame in sequence:
            enhanced_frame = frame.copy()
            
            if config["denoise"]:
                enhanced_frame = self.denoise_frame(enhanced_frame)
            
            if config["brightness"]:
                enhanced_frame = self.adjust_brightness_contrast(enhanced_frame)
            
            if config["contrast"]:
                enhanced_frame = self.enhance_contrast(enhanced_frame)
            
            if config["sharpen"]:
                enhanced_frame = self.sharpen_frame(enhanced_frame)
            
            enhanced_frames.append(enhanced_frame)
        
        enhanced_sequence = np.array(enhanced_frames)
        
        # Apply sequence-wise enhancements
        if config["stabilize"]:
            enhanced_sequence = self.stabilize_sequence(enhanced_sequence)
        
        return enhanced_sequence
    
    def enhance_processed_data(self, split_name, enhancement_level="medium"):
        """Enhance all processed sequences in a split"""
        print(f"\nEnhancing video quality for {split_name} split...")
        
        input_dir = PATHS["processed_frames"] / split_name
        output_dir = PATHS["processed_frames"] / f"{split_name}_enhanced"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not input_dir.exists():
            print(f"Input directory {input_dir} does not exist")
            return 0
        
        video_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        enhanced_count = 0
        
        for video_dir in tqdm(video_dirs, desc=f"Enhancing {split_name}"):
            sequences_dir = video_dir / "sequences"
            if not sequences_dir.exists():
                continue
            
            # Create output directory
            enhanced_video_dir = output_dir / video_dir.name
            enhanced_sequences_dir = enhanced_video_dir / "sequences"
            enhanced_sequences_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each sequence
            sequence_files = list(sequences_dir.glob("sequence_*.npy"))
            
            for seq_file in sequence_files:
                try:
                    # Load original sequence
                    original_sequence = np.load(seq_file)
                    
                    # Apply enhancement
                    enhanced_sequence = self.enhance_sequence(original_sequence, enhancement_level)
                    
                    # Save enhanced sequence
                    enhanced_file_path = enhanced_sequences_dir / f"{seq_file.stem}_enhanced.npy"
                    np.save(enhanced_file_path, enhanced_sequence)
                    enhanced_count += 1
                    
                except Exception as e:
                    print(f"Error enhancing {seq_file}: {e}")
            
            # Copy metadata
            metadata_file = video_dir / "metadata.json"
            if metadata_file.exists():
                enhanced_metadata_file = enhanced_video_dir / "metadata.json"
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                metadata["enhancement_level"] = enhancement_level
                metadata["enhanced"] = True
                
                with open(enhanced_metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        print(f"Enhanced {enhanced_count} sequences for {split_name}")
        return enhanced_count
    
    def analyze_enhancement_quality(self, original_sequence, enhanced_sequence):
        """Analyze quality improvement metrics"""
        metrics = {}
        
        # Calculate brightness statistics
        orig_brightness = np.mean(original_sequence)
        enh_brightness = np.mean(enhanced_sequence)
        
        # Calculate contrast (standard deviation)
        orig_contrast = np.std(original_sequence)
        enh_contrast = np.std(enhanced_sequence)
        
        # Calculate sharpness (Laplacian variance)
        orig_sharpness = []
        enh_sharpness = []
        
        for i in range(len(original_sequence)):
            orig_gray = cv2.cvtColor((original_sequence[i] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            enh_gray = cv2.cvtColor((enhanced_sequence[i] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            orig_sharpness.append(cv2.Laplacian(orig_gray, cv2.CV_64F).var())
            enh_sharpness.append(cv2.Laplacian(enh_gray, cv2.CV_64F).var())
        
        metrics = {
            "brightness_improvement": float(enh_brightness - orig_brightness),
            "contrast_improvement": float(enh_contrast - orig_contrast),
            "sharpness_improvement": float(np.mean(enh_sharpness) - np.mean(orig_sharpness)),
            "original_quality": {
                "brightness": float(orig_brightness),
                "contrast": float(orig_contrast),
                "sharpness": float(np.mean(orig_sharpness))
            },
            "enhanced_quality": {
                "brightness": float(enh_brightness),
                "contrast": float(enh_contrast),
                "sharpness": float(np.mean(enh_sharpness))
            }
        }
        
        return metrics

if __name__ == "__main__":
    enhancer = VideoQualityEnhancer()
    
    # Test enhancement on train split only (validation/test should not be enhanced)
    enhanced_count = enhancer.enhance_processed_data("train", enhancement_level="medium")
    print(f"Quality enhancement completed: {enhanced_count} sequences enhanced")
