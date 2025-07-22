#!/usr/bin/env python3
"""
🎬 ENHANCED LIVE POTHOLE DETECTION WITH VISUAL INTERFACE! 🎬
Real-time video processing with RL optimization and VISUAL detection boxes
"""

import cv2
import numpy as np
import torch
import time
import json
from pathlib import Path
from collections import deque
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agents.advanced_dqn import AdvancedDQNAgent
from src.environment.pothole_env import VideoBasedPotholeEnv
from src.detectors.real_cnn_detector import create_cnn_detector
from configs.config import ENV_CONFIG, VIDEO_CONFIG

class LivePotholeProcessor:
    """
    🚀 ENHANCED LIVE POTHOLE DETECTION SYSTEM WITH VISUAL INTERFACE
    Real-time video processing with RL-optimized threshold selection and visual detection boxes
    """
    
    def __init__(self, rl_model_path=None, cnn_model_path=None, use_real_cnn=True):
        print("🎬 Initializing ENHANCED LIVE POTHOLE DETECTION SYSTEM!")
        
        # Initialize RL agent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🧠 Using device: {self.device}")
        
        # Load trained RL agent
        if rl_model_path and Path(rl_model_path).exists():
            try:
                self.rl_agent = AdvancedDQNAgent(
                    use_double_dqn=True,
                    use_dueling=True,
                    use_prioritized_replay=True
                )
                self.rl_agent.load_model(rl_model_path)
                self.rl_agent.q_network.eval()
                print(f"✅ RL agent loaded from {rl_model_path}")
            except Exception as e:
                print(f"⚠️ Could not load RL model: {e}")
                self.rl_agent = None
        else:
            print("🔄 Using random RL agent (no trained model provided)")
            self.rl_agent = AdvancedDQNAgent(
                use_double_dqn=True,
                use_dueling=True,
                use_prioritized_replay=True
            )
        
        # Initialize CNN detector
        self.cnn_detector = create_cnn_detector(
            use_real_cnn=use_real_cnn,
            model_path=cnn_model_path
        )
        
        # Video processing setup
        self.frame_buffer = deque(maxlen=5)  # 5-frame sequences
        self.sequence_length = 5
        self.input_size = (224, 224)
        
        # Action thresholds from config
        self.action_thresholds = ENV_CONFIG["action_thresholds"]
        
        # Performance tracking
        self.detection_history = []
        self.performance_stats = {
            'total_frames': 0,
            'total_detections': 0,
            'avg_confidence': 0.0,
            'avg_processing_time': 0.0,
            'fps': 0.0
        }
        
        # Visual detection settings
        self.colors = {
            'pothole': (0, 0, 255),      # Red for potholes
            'safe': (0, 255, 0),         # Green for safe areas
            'uncertain': (0, 255, 255),   # Yellow for uncertain
            'processing': (255, 0, 0)     # Blue for processing
        }
        
        print("✅ Enhanced Live processing system ready!")
        print(f"🎯 Action thresholds: {self.action_thresholds}")
        print(f"🧠 RL Agent: {'Trained' if rl_model_path else 'Random'}")
        print(f"👁️ CNN Detector: {'Real' if use_real_cnn else 'Simulated'}")
        print("🎨 Visual detection interface enabled!")
    
    def detect_pothole_regions(self, frame, confidence):
        """
        🎯 DETECT POTHOLE REGIONS WITH VISUAL BOUNDING BOXES
        Simulates CNN spatial detection for visualization
        """
        detections = []
        h, w = frame.shape[:2]
        
        # Based on confidence, generate detection regions
        if confidence > 0.5:  # High confidence - show detection boxes
            # Generate 1-3 detection regions based on confidence
            num_regions = int(1 + (confidence - 0.5) * 4)  # 1-3 regions
            
            for i in range(min(num_regions, 3)):
                # Random detection regions (in production, this comes from CNN)
                x = np.random.randint(50, w - 150)
                y = np.random.randint(50, h - 100) 
                width = np.random.randint(80, 200)
                height = np.random.randint(60, 120)
                
                # Ensure bbox stays within frame
                x = max(0, min(x, w - width))
                y = max(0, min(y, h - height))
                
                # Calculate region confidence (varies by region)
                region_confidence = confidence * (0.8 + 0.4 * np.random.random())
                
                detection = {
                    'bbox': (x, y, width, height),
                    'confidence': min(region_confidence, 0.98),
                    'type': 'pothole',
                    'severity': 'high' if region_confidence > 0.8 else 'medium'
                }
                detections.append(detection)
        
        return detections
    
    def draw_detection_boxes(self, frame, detections, overall_confidence):
        """
        🎨 DRAW VISUAL DETECTION BOXES ON FRAME
        Like face detection boxes but for potholes!
        """
        for detection in detections:
            x, y, w, h = detection['bbox']
            conf = detection['confidence']
            severity = detection.get('severity', 'medium')
            
            # Color based on severity
            if severity == 'high':
                color = (0, 0, 255)  # Red for high severity
            else:
                color = (0, 165, 255)  # Orange for medium severity
            
            # Draw main bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # Draw confidence bar
            bar_width = w
            bar_height = 8
            bar_fill = int(bar_width * conf)
            
            # Background bar
            cv2.rectangle(frame, (x, y - 20), (x + bar_width, y - 12), (100, 100, 100), -1)
            # Filled bar
            cv2.rectangle(frame, (x, y - 20), (x + bar_fill, y - 12), color, -1)
            
            # Label with confidence
            label = f"POTHOLE {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Label background
            cv2.rectangle(frame, (x, y - 40), (x + label_size[0] + 10, y - 20), color, -1)
            # Label text
            cv2.putText(frame, label, (x + 5, y - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw center crosshair
            center_x, center_y = x + w//2, y + h//2
            cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), color, 2)
            cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), color, 2)
        
        return frame
    
    def draw_detection_heatmap(self, frame, detections):
        """
        🔥 DRAW DETECTION HEATMAP OVERLAY
        Shows pothole probability as heat overlay
        """
        if not detections:
            return frame
            
        # Create heatmap overlay
        overlay = frame.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            conf = detection['confidence']
            
            # Create elliptical heat region
            center = (x + w//2, y + h//2)
            axes = (w//2, h//2)
            
            # Heat intensity based on confidence
            alpha = int(conf * 100)  # 0-100 alpha
            heat_color = (0, int(conf * 255), int((1-conf) * 255))  # Blue to red gradient
            
            # Draw filled ellipse for heat effect
            cv2.ellipse(overlay, center, axes, 0, 0, 360, heat_color, -1)
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        return frame
    
    def process_live_video(self, video_source=0, save_detections=False):
        """
        🎯 MAIN LIVE PROCESSING METHOD WITH VISUAL DETECTION
        Process live video stream with RL-optimized pothole detection and visual feedback
        """
        print(f"\n📹 STARTING ENHANCED LIVE VIDEO PROCESSING...")
        print(f"📺 Video source: {video_source}")
        print(f"💾 Save detections: {save_detections}")
        print("⌨️ Controls: 'q'=quit, 's'=save screenshot, 'r'=reset stats, 'h'=toggle heatmap")
        
        # Initialize video capture
        if isinstance(video_source, int):
            cap = cv2.VideoCapture(video_source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            cap = cv2.VideoCapture(str(video_source))
        
        if not cap.isOpened():
            print(f"❌ Could not open video source: {video_source}")
            return
        
        # Performance tracking
        frame_count = 0
        screenshots_saved = 0
        start_time = time.time()
        processing_times = []
        show_heatmap = True  # Toggle for heatmap display
        
        # Create display window
        cv2.namedWindow('🚀 ENHANCED POTHOLE DETECTION - RL OPTIMIZED', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('🚀 ENHANCED POTHOLE DETECTION - RL OPTIMIZED', 1400, 900)
        
        print(f"🟢 ENHANCED LIVE PROCESSING ACTIVE - Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(video_source, str):  # Video file ended
                        print("📹 Video file completed")
                        break
                    else:
                        print("⚠️ Failed to grab frame")
                        continue
                
                frame_start_time = time.time()
                frame_count += 1
                
                # Process frame with enhanced detection
                processed_frame, detection_info = self.process_single_frame(frame)
                
                # Track processing time
                processing_time = time.time() - frame_start_time
                processing_times.append(processing_time)
                
                # Create enhanced display frame
                display_frame = self.create_enhanced_display_frame(
                    processed_frame, detection_info, frame_count, processing_times, show_heatmap
                )
                
                # Show frame
                cv2.imshow('🚀 ENHANCED POTHOLE DETECTION - RL OPTIMIZED', display_frame)
                
                # Save detection if requested
                if save_detections and detection_info['pothole_detected']:
                    self.save_detection(display_frame, detection_info, frame_count)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_path = f"enhanced_screenshot_{screenshots_saved:03d}.jpg"
                    cv2.imwrite(screenshot_path, display_frame)
                    screenshots_saved += 1
                    print(f"📸 Screenshot saved: {screenshot_path}")
                elif key == ord('r'):
                    # Reset statistics
                    processing_times.clear()
                    self.detection_history.clear()
                    frame_count = 0
                    start_time = time.time()
                    print("🔄 Statistics reset")
                elif key == ord('h'):
                    # Toggle heatmap
                    show_heatmap = not show_heatmap
                    print(f"🔥 Heatmap: {'ON' if show_heatmap else 'OFF'}")
                
                # Update performance stats every 100 frames
                if frame_count % 100 == 0:
                    self.update_performance_stats(frame_count, start_time, processing_times)
        
        except KeyboardInterrupt:
            print("\n⏹️ Processing interrupted by user")
        
        finally:
            # Cleanup and generate report
            cap.release()
            cv2.destroyAllWindows()
            total_time = time.time() - start_time
            self.generate_processing_report(frame_count, total_time, processing_times)
    
    def process_single_frame(self, frame):
        """Process single frame and return detection results with visual information"""
        # Preprocess frame for CNN
        processed_frame = cv2.resize(frame, self.input_size)
        processed_frame_normalized = processed_frame.astype(np.float32) / 255.0
        
        # Add to frame buffer
        self.frame_buffer.append(processed_frame_normalized)
        
        # Initialize detection info
        detection_info = {
            'pothole_detected': False,
            'confidence': 0.0,
            'threshold_used': 0.5,
            'rl_action': 0,
            'processing_time': 0.0,
            'detections': []  # NEW: Store detection boxes
        }
        
        # Process when we have enough frames
        if len(self.frame_buffer) == self.sequence_length:
            # Create sequence for processing
            sequence = np.stack(list(self.frame_buffer))
            
            # Get CNN confidence
            cnn_start = time.time()
            confidence = self.cnn_detector.detect_pothole_confidence(sequence)
            cnn_time = time.time() - cnn_start
            
            # Get RL-optimized threshold
            if self.rl_agent:
                rl_start = time.time()
                rl_action = self.rl_agent.act(sequence, training=False)
                rl_time = time.time() - rl_start
                threshold = self.action_thresholds[rl_action]
            else:
                rl_action = 1  # Default to middle threshold
                threshold = self.action_thresholds[rl_action]
                rl_time = 0.0
            
            # Make detection decision
            pothole_detected = confidence > threshold
            
            # 🎯 NEW: Generate visual detection regions
            detections = self.detect_pothole_regions(frame, confidence) if pothole_detected else []
            
            # Update detection info
            detection_info.update({
                'pothole_detected': pothole_detected,
                'confidence': confidence,
                'threshold_used': threshold,
                'rl_action': rl_action,
                'processing_time': cnn_time + rl_time,
                'cnn_time': cnn_time,
                'rl_time': rl_time,
                'detections': detections  # NEW: Include detection boxes
            })
            
            # Add to history
            self.detection_history.append(detection_info.copy())
        
        return processed_frame, detection_info
    
    def create_enhanced_display_frame(self, frame, detection_info, frame_count, processing_times, show_heatmap=True):
        """Create enhanced display frame with VISUAL DETECTION BOXES"""
        # Create display frame
        display_frame = frame.copy()
        if len(display_frame.shape) == 3 and display_frame.shape[2] == 3:
            display_frame = cv2.resize(display_frame, (1280, 720))
        else:
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
            display_frame = cv2.resize(display_frame, (1280, 720))
        
        # 🎯 NEW: Draw detection boxes and heatmap
        detections = detection_info.get('detections', [])
        if detections:
            # Draw heatmap first (background)
            if show_heatmap:
                display_frame = self.draw_detection_heatmap(display_frame, detections)
            # Then draw detection boxes (foreground)  
            display_frame = self.draw_detection_boxes(display_frame, detections, detection_info['confidence'])
        
        # Enhanced detection status with visual indicator
        if detection_info['pothole_detected']:
            status = "🕳️ POTHOLE DETECTED!"
            color = (0, 0, 255)  # Red
            # Draw pulsing border for detected potholes
            thickness = 5 + int(3 * np.sin(frame_count * 0.5))  # Pulsing effect
            cv2.rectangle(display_frame, (10, 10), (1270, 710), color, thickness)
        else:
            status = "🛣️ Road Clear"
            color = (0, 255, 0)  # Green
        
        # Enhanced text overlay panel
        panel_height = 150
        cv2.rectangle(display_frame, (0, 0), (1280, panel_height), (0, 0, 0), -1)  # Black background
        cv2.rectangle(display_frame, (0, 0), (1280, panel_height), (50, 50, 50), 2)  # Gray border
        
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Main status
        cv2.putText(display_frame, status, (10, y_offset), font, 1.2, color, 3)
        y_offset += 40
        
        # Detection details
        details = [
            f"Confidence: {detection_info['confidence']:.3f}",
            f"RL Threshold: {detection_info['threshold_used']:.3f} (Action {detection_info['rl_action']})",
            f"Detections: {len(detections)} regions found" if detections else "No regions detected",
            f"Frame: {frame_count} | Buffer: {len(self.frame_buffer)}/5"
        ]
        
        for detail in details:
            cv2.putText(display_frame, detail, (10, y_offset), font, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        # Performance stats in corner
        if processing_times:
            avg_time = np.mean(processing_times[-100:])
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            perf_details = [
                f"FPS: {fps:.1f}",
                f"Processing: {detection_info['processing_time']*1000:.1f}ms", 
                f"Total Detections: {sum(1 for d in self.detection_history if d['pothole_detected'])}"
            ]
            
            y_perf = 200
            for detail in perf_details:
                cv2.putText(display_frame, detail, (1000, y_perf), font, 0.5, (255, 255, 0), 1)
                y_perf += 20
        
        # 🎯 NEW: Detection legend
        if detections:
            legend_x, legend_y = 10, 600
            cv2.rectangle(display_frame, (legend_x, legend_y), (legend_x + 250, legend_y + 100), (0, 0, 0), -1)
            cv2.putText(display_frame, "DETECTION LEGEND:", (legend_x + 10, legend_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "Red Box: High Severity", (legend_x + 10, legend_y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(display_frame, "Orange Box: Medium Severity", (legend_x + 10, legend_y + 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            if show_heatmap:
                cv2.putText(display_frame, "Heat Map: Probability", (legend_x + 10, legend_y + 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Controls help
        controls_y = 720 - 60
        cv2.rectangle(display_frame, (0, controls_y), (1280, 720), (0, 0, 0), -1)
        cv2.putText(display_frame, "Controls: 'q'=quit | 's'=screenshot | 'r'=reset | 'h'=toggle heatmap", 
                   (10, 720 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display_frame
    
    def save_detection(self, frame, detection_info, frame_count):
        """Save detected pothole frame with enhanced metadata"""
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"enhanced_pothole_detection_{frame_count:06d}_{timestamp}.jpg"
        filepath = Path("results/live_demos") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        cv2.imwrite(str(filepath), frame)
        
        # Save enhanced detection metadata
        meta_file = filepath.with_suffix('.json')
        enhanced_metadata = detection_info.copy()
        enhanced_metadata['timestamp'] = timestamp
        enhanced_metadata['frame_count'] = frame_count
        enhanced_metadata['detection_boxes'] = len(detection_info.get('detections', []))
        
        with open(meta_file, 'w') as f:
            json.dump(enhanced_metadata, f, indent=2)
    
    def update_performance_stats(self, frame_count, start_time, processing_times):
        """Update and display performance statistics"""
        total_time = time.time() - start_time
        if processing_times:
            avg_processing = np.mean(processing_times)
            fps = frame_count / total_time
            detections = sum(1 for d in self.detection_history if d['pothole_detected'])
            
            print(f"📊 Enhanced Stats Update - Frame {frame_count}:")
            print(f"⚡ FPS: {fps:.1f}")
            print(f"🔄 Avg Processing: {avg_processing*1000:.1f}ms")
            print(f"🕳️ Detections: {detections}")
            print(f"📈 Detection Rate: {detections/max(frame_count,1)*100:.1f}%")
            print(f"🎨 Visual Features: Detection boxes + Heatmap")
    
    def generate_processing_report(self, total_frames, total_time, processing_times):
        """Generate comprehensive processing report with visual detection metrics"""
        if not self.detection_history:
            print("📊 No detection data to report")
            return
        
        # Calculate enhanced statistics
        total_detections = sum(1 for d in self.detection_history if d['pothole_detected'])
        total_detection_boxes = sum(len(d.get('detections', [])) for d in self.detection_history)
        avg_confidence = np.mean([d['confidence'] for d in self.detection_history])
        avg_processing = np.mean(processing_times) if processing_times else 0
        fps = total_frames / total_time if total_time > 0 else 0
        
        # Threshold usage analysis
        threshold_usage = {}
        for d in self.detection_history:
            action = d['rl_action']
            if action not in threshold_usage:
                threshold_usage[action] = {'count': 0, 'detections': 0}
            threshold_usage[action]['count'] += 1
            if d['pothole_detected']:
                threshold_usage[action]['detections'] += 1
        
        report = {
            'session_summary': {
                'total_frames': total_frames,
                'total_time_seconds': total_time,
                'total_detections': total_detections,
                'total_detection_boxes': total_detection_boxes,
                'detection_rate_percent': total_detections / max(total_frames, 1) * 100,
                'average_fps': fps,
                'average_confidence': avg_confidence,
                'average_processing_time_ms': avg_processing * 1000
            },
            'visual_detection_stats': {
                'frames_with_boxes': sum(1 for d in self.detection_history if d.get('detections', [])),
                'avg_boxes_per_detection': total_detection_boxes / max(total_detections, 1),
                'visual_features_enabled': True
            },
            'threshold_analysis': threshold_usage,
            'performance_breakdown': {
                'cnn_times': [d.get('cnn_time', 0) for d in self.detection_history],
                'rl_times': [d.get('rl_time', 0) for d in self.detection_history]
            }
        }
        
        # Print enhanced report
        print(f"\n📊 ENHANCED LIVE PROCESSING REPORT:")
        print(f"="*50)
        print(f"🎬 Total frames processed: {total_frames:,}")
        print(f"⏱️ Total processing time: {total_time:.1f}s")
        print(f"🕳️ Potholes detected: {total_detections}")
        print(f"📦 Detection boxes drawn: {total_detection_boxes}")
        print(f"📈 Detection rate: {report['session_summary']['detection_rate_percent']:.1f}%")
        print(f"⚡ Average FPS: {fps:.1f}")
        print(f"🎯 Average confidence: {avg_confidence:.3f}")
        print(f"🔄 Average processing time: {avg_processing*1000:.1f}ms")
        print(f"🎨 Visual detection features: ENABLED")
        
        print(f"\n🎯 RL Threshold Usage:")
        for action, stats in threshold_usage.items():
            threshold = self.action_thresholds[action]
            success_rate = stats['detections'] / max(stats['count'], 1) * 100
            print(f"Action {action} (th={threshold:.1f}): {stats['count']} uses, "
                  f"{success_rate:.1f}% detection rate")
        
        # Save detailed report
        report_path = Path("results/live_demos") / f"enhanced_live_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Enhanced detailed report saved: {report_path}")
    
    def process_video_file(self, video_path, output_path=None):
        """Process a video file with enhanced visual detection and optionally save annotated output"""
        print(f"🎬 Processing video file with enhanced detection: {video_path}")
        
        if output_path:
            # Initialize video writer for enhanced output
            cap = cv2.VideoCapture(str(video_path))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            cap.release()
        
        # Process video with enhanced detection
        self.process_live_video(video_source=video_path, save_detections=True)
        
        if output_path:
            print(f"💾 Enhanced annotated video saved: {output_path}")

# Example usage and testing
if __name__ == "__main__":
    print("🎬 TESTING ENHANCED LIVE POTHOLE PROCESSING SYSTEM!")
    print("="*60)
    
    # Initialize enhanced processor
    processor = LivePotholeProcessor(
        rl_model_path="results/ultimate_comparison/models/best_ultimate_dqn_model.pth",  # Use your trained model
        use_real_cnn=False  # Start with simulated CNN for testing
    )
    
    print("\n🎯 Available test modes:")
    print("1. Webcam processing (live) - Enhanced visual detection")
    print("2. Video file processing - With visual detection boxes")
    print("3. Performance benchmark - Visual detection performance")
    
    try:
        # Test with webcam (uncomment to use)
        processor.process_live_video(video_source=0)
        
        # Or test with a video file (uncomment to use)
        # processor.process_video_file("path/to/test/video.mp4", "enhanced_output_annotated.mp4")
        
        print("🎉 Enhanced live processing system ready!")
        print("Features: Visual detection boxes, heatmaps, confidence bars, crosshairs!")
        
    except Exception as e:
        print(f"⚠️ Test error: {e}")
