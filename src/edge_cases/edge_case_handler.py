#!/usr/bin/env python3
"""
🛡️ EDGE CASE HANDLER FOR POTHOLE DETECTION 🛡️
Robust handling of challenging scenarios and edge cases
"""

import numpy as np
import torch
import cv2
from pathlib import Path
import json
from datetime import datetime

class EdgeCaseDetector:
    """
    🔍 EDGE CASE DETECTION AND HANDLING SYSTEM
    
    Identifies and handles challenging scenarios:
    - Low lighting conditions
    - Motion blur
    - Occlusions
    - Weather conditions
    - Surface variations
    """
    
    def __init__(self, confidence_threshold=0.3):
        self.confidence_threshold = confidence_threshold
        self.edge_case_history = []
        
    def analyze_sequence_quality(self, sequence):
        """Analyze video sequence for edge case conditions"""
        edge_cases = {
            'low_lighting': False,
            'motion_blur': False,
            'occlusion': False,
            'weather_effects': False,
            'surface_anomaly': False,
            'confidence_score': 1.0
        }
        
        # Analyze each frame in sequence
        brightness_scores = []
        blur_scores = []
        
        for frame_idx in range(sequence.shape[0]):
            frame = sequence[frame_idx]
            
            # Convert to grayscale for analysis
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray_frame = (frame * 255).astype(np.uint8)
            
            # Brightness analysis
            brightness = np.mean(gray_frame) / 255.0
            brightness_scores.append(brightness)
            
            # Blur detection using Laplacian variance
            blur_score = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
            blur_scores.append(blur_score)
        
        # Edge case detection
        avg_brightness = np.mean(brightness_scores)
        avg_blur = np.mean(blur_scores)
        
        # Low lighting detection
        if avg_brightness < 0.3:
            edge_cases['low_lighting'] = True
            edge_cases['confidence_score'] *= 0.7
        
        # Motion blur detection
        if avg_blur < 100:  # Low variance indicates blur
            edge_cases['motion_blur'] = True
            edge_cases['confidence_score'] *= 0.8
        
        # Weather effects (detect through brightness variation)
        brightness_std = np.std(brightness_scores)
        if brightness_std > 0.2:
            edge_cases['weather_effects'] = True
            edge_cases['confidence_score'] *= 0.85
        
        # Surface anomaly detection (high contrast variations)
        contrast_scores = []
        for frame_idx in range(sequence.shape[0]):
            frame = sequence[frame_idx]
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                contrast = np.std(gray_frame) / 255.0
                contrast_scores.append(contrast)
        
        avg_contrast = np.mean(contrast_scores)
        if avg_contrast > 0.3:
            edge_cases['surface_anomaly'] = True
            edge_cases['confidence_score'] *= 0.9
        
        return edge_cases
    
    def adjust_detection_strategy(self, edge_cases, base_confidence):
        """Adjust detection strategy based on edge case analysis"""
        adjusted_confidence = base_confidence * edge_cases['confidence_score']
        
        recommendations = {
            'adjusted_confidence': adjusted_confidence,
            'threshold_adjustment': 0,
            'preprocessing_needed': [],
            'reliability_score': edge_cases['confidence_score']
        }
        
        # Threshold adjustments for different edge cases
        if edge_cases['low_lighting']:
            recommendations['threshold_adjustment'] -= 0.1  # Lower threshold
            recommendations['preprocessing_needed'].append('brightness_enhancement')
        
        if edge_cases['motion_blur']:
            recommendations['threshold_adjustment'] -= 0.05  # Slightly lower threshold
            recommendations['preprocessing_needed'].append('deblur_filter')
        
        if edge_cases['weather_effects']:
            recommendations['threshold_adjustment'] -= 0.08
            recommendations['preprocessing_needed'].append('weather_compensation')
        
        return recommendations
    
    def preprocess_sequence(self, sequence, preprocessing_needed):
        """Apply preprocessing based on edge case analysis"""
        processed_sequence = sequence.copy()
        
        for preprocessing in preprocessing_needed:
            if preprocessing == 'brightness_enhancement':
                processed_sequence = self._enhance_brightness(processed_sequence)
            elif preprocessing == 'deblur_filter':
                processed_sequence = self._apply_deblur(processed_sequence)
            elif preprocessing == 'weather_compensation':
                processed_sequence = self._compensate_weather(processed_sequence)
        
        return processed_sequence
    
    def _enhance_brightness(self, sequence):
        """Enhance brightness for low-light conditions"""
        enhanced = sequence.copy()
        
        for frame_idx in range(sequence.shape[0]):
            frame = enhanced[frame_idx]
            
            # Gamma correction for brightness enhancement
            gamma = 1.5
            enhanced[frame_idx] = np.power(frame, 1/gamma)
        
        return np.clip(enhanced, 0, 1)
    
    def _apply_deblur(self, sequence):
        """Apply deblurring filter"""
        deblurred = sequence.copy()
        
        # Simple sharpening kernel
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        
        for frame_idx in range(sequence.shape[0]):
            frame = (deblurred[frame_idx] * 255).astype(np.uint8)
            
            if len(frame.shape) == 3:
                for channel in range(3):
                    frame[:,:,channel] = cv2.filter2D(frame[:,:,channel], -1, kernel)
            else:
                frame = cv2.filter2D(frame, -1, kernel)
            
            deblurred[frame_idx] = frame / 255.0
        
        return np.clip(deblurred, 0, 1)
    
    def _compensate_weather(self, sequence):
        """Compensate for weather effects"""
        compensated = sequence.copy()
        
        for frame_idx in range(sequence.shape[0]):
            frame = compensated[frame_idx]
            
            # Histogram equalization for weather compensation
            if len(frame.shape) == 3:
                frame_uint8 = (frame * 255).astype(np.uint8)
                
                # Convert to LAB color space
                lab = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                
                # Merge and convert back
                lab = cv2.merge([l, a, b])
                compensated[frame_idx] = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) / 255.0
        
        return np.clip(compensated, 0, 1)

class RobustPotholeDetector:
    """
    🛡️ ROBUST POTHOLE DETECTION WITH EDGE CASE HANDLING
    
    Integrates edge case detection with robust detection strategies
    """
    
    def __init__(self, base_agent, edge_case_detector=None):
        self.base_agent = base_agent
        self.edge_case_detector = edge_case_detector or EdgeCaseDetector()
        self.detection_history = []
        
    def robust_detect(self, sequence, ground_truth=None):
        """Perform robust detection with edge case handling"""
        # Analyze sequence for edge cases
        edge_case_analysis = self.edge_case_detector.analyze_sequence_quality(sequence)
        
        # Get base prediction
        base_prediction = self.base_agent.act(sequence, training=False)
        base_confidence = getattr(self.base_agent, '_last_confidence', 0.5)
        
        # Adjust strategy based on edge cases
        adjustments = self.edge_case_detector.adjust_detection_strategy(
            edge_case_analysis, base_confidence
        )
        
        # Apply preprocessing if needed
        if adjustments['preprocessing_needed']:
            processed_sequence = self.edge_case_detector.preprocess_sequence(
                sequence, adjustments['preprocessing_needed']
            )
            # Re-evaluate with processed sequence
            enhanced_prediction = self.base_agent.act(processed_sequence, training=False)
        else:
            enhanced_prediction = base_prediction
            processed_sequence = sequence
        
        # Final decision with confidence adjustment
        final_confidence = adjustments['adjusted_confidence']
        threshold_adjustment = adjustments['threshold_adjustment']
        
        # Store detection result
        detection_result = {
            'timestamp': datetime.now().isoformat(),
            'base_prediction': base_prediction,
            'enhanced_prediction': enhanced_prediction,
            'final_confidence': final_confidence,
            'edge_cases': edge_case_analysis,
            'adjustments': adjustments,
            'reliability_score': adjustments['reliability_score']
        }
        
        self.detection_history.append(detection_result)
        
        return {
            'prediction': enhanced_prediction,
            'confidence': final_confidence,
            'reliability': adjustments['reliability_score'],
            'edge_cases': edge_case_analysis,
            'preprocessing_applied': adjustments['preprocessing_needed']
        }
    
    def get_robustness_statistics(self):
        """Get statistics on edge case handling performance"""
        if not self.detection_history:
            return {}
        
        edge_case_counts = {
            'low_lighting': 0,
            'motion_blur': 0,
            'occlusion': 0,
            'weather_effects': 0,
            'surface_anomaly': 0
        }
        
        reliability_scores = []
        
        for detection in self.detection_history:
            edge_cases = detection['edge_cases']
            reliability_scores.append(detection['reliability_score'])
            
            for case_type in edge_case_counts:
                if edge_cases.get(case_type, False):
                    edge_case_counts[case_type] += 1
        
        total_detections = len(self.detection_history)
        
        return {
            'total_detections': total_detections,
            'edge_case_percentages': {
                case: (count / total_detections) * 100 
                for case, count in edge_case_counts.items()
            },
            'average_reliability': np.mean(reliability_scores),
            'reliability_std': np.std(reliability_scores),
            'robust_detections': sum(1 for score in reliability_scores if score > 0.7)
        }
