#!/usr/bin/env python3
"""
🌦️ WEATHER-ADAPTIVE POTHOLE DETECTION SYSTEM 🌦️
Advanced adaptation for various weather conditions
"""

import numpy as np
import torch
import torch.nn as nn
from enum import Enum
import cv2

class WeatherCondition(Enum):
    CLEAR = "clear"
    RAIN = "rain"
    FOG = "fog"
    SNOW = "snow"
    OVERCAST = "overcast"
    TWILIGHT = "twilight"

class WeatherDetector:
    """
    🌤️ WEATHER CONDITION DETECTOR
    
    Automatically detects weather conditions from video sequences
    """
    
    def __init__(self):
        self.weather_features = {
            WeatherCondition.CLEAR: {'brightness': (0.4, 0.8), 'contrast': (0.3, 0.7), 'saturation': (0.4, 0.8)},
            WeatherCondition.RAIN: {'brightness': (0.2, 0.5), 'contrast': (0.2, 0.5), 'saturation': (0.3, 0.6)},
            WeatherCondition.FOG: {'brightness': (0.5, 0.8), 'contrast': (0.1, 0.3), 'saturation': (0.2, 0.4)},
            WeatherCondition.SNOW: {'brightness': (0.6, 0.9), 'contrast': (0.1, 0.4), 'saturation': (0.1, 0.3)},
            WeatherCondition.OVERCAST: {'brightness': (0.3, 0.6), 'contrast': (0.2, 0.5), 'saturation': (0.2, 0.5)},
            WeatherCondition.TWILIGHT: {'brightness': (0.1, 0.4), 'contrast': (0.3, 0.6), 'saturation': (0.3, 0.7)}
        }
    
    def detect_weather(self, sequence):
        """Detect weather condition from video sequence"""
        features = self._extract_weather_features(sequence)
        
        # Score each weather condition
        weather_scores = {}
        for condition, ranges in self.weather_features.items():
            score = 1.0
            
            for feature, (min_val, max_val) in ranges.items():
                feature_val = features[feature]
                if min_val <= feature_val <= max_val:
                    score *= 1.0  # Perfect match
                else:
                    # Penalty for being outside range
                    distance = min(abs(feature_val - min_val), abs(feature_val - max_val))
                    score *= max(0.1, 1.0 - distance)
            
            weather_scores[condition] = score
        
        # Return most likely weather condition
        best_condition = max(weather_scores, key=weather_scores.get)
        confidence = weather_scores[best_condition]
        
        return {
            'condition': best_condition,
            'confidence': confidence,
            'features': features,
            'all_scores': weather_scores
        }
    
    def _extract_weather_features(self, sequence):
        """Extract weather-related features from sequence"""
        brightness_values = []
        contrast_values = []
        saturation_values = []
        
        for frame_idx in range(sequence.shape[0]):
            frame = sequence[frame_idx]
            
            if len(frame.shape) == 3:
                # Convert to different color spaces for analysis
                frame_uint8 = (frame * 255).astype(np.uint8)
                
                # Brightness (average of all pixels)
                brightness = np.mean(frame)
                brightness_values.append(brightness)
                
                # Contrast (standard deviation)
                gray = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2GRAY)
                contrast = np.std(gray) / 255.0
                contrast_values.append(contrast)
                
                # Saturation (from HSV)
                hsv = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2HSV)
                saturation = np.mean(hsv[:, :, 1]) / 255.0
                saturation_values.append(saturation)
        
        return {
            'brightness': np.mean(brightness_values),
            'contrast': np.mean(contrast_values),
            'saturation': np.mean(saturation_values)
        }

class WeatherAdaptiveAgent:
    """
    🌦️ WEATHER-ADAPTIVE RL AGENT
    
    Adapts detection strategies based on weather conditions
    """
    
    def __init__(self, base_agent):
        self.base_agent = base_agent
        self.weather_detector = WeatherDetector()
        
        # Weather-specific configurations
        self.weather_configs = {
            WeatherCondition.CLEAR: {
                'confidence_boost': 1.0,
                'threshold_adjustment': 0.0,
                'preprocessing': []
            },
            WeatherCondition.RAIN: {
                'confidence_boost': 0.8,
                'threshold_adjustment': -0.1,
                'preprocessing': ['rain_removal', 'contrast_enhancement']
            },
            WeatherCondition.FOG: {
                'confidence_boost': 0.7,
                'threshold_adjustment': -0.15,
                'preprocessing': ['fog_removal', 'brightness_enhancement']
            },
            WeatherCondition.SNOW: {
                'confidence_boost': 0.75,
                'threshold_adjustment': -0.12,
                'preprocessing': ['snow_removal', 'contrast_enhancement']
            },
            WeatherCondition.OVERCAST: {
                'confidence_boost': 0.85,
                'threshold_adjustment': -0.05,
                'preprocessing': ['brightness_enhancement']
            },
            WeatherCondition.TWILIGHT: {
                'confidence_boost': 0.7,
                'threshold_adjustment': -0.2,
                'preprocessing': ['low_light_enhancement', 'noise_reduction']
            }
        }
        
        self.detection_history = []
    
    def weather_adaptive_detect(self, sequence):
        """Perform weather-adaptive detection"""
        # Detect weather condition
        weather_info = self.weather_detector.detect_weather(sequence)
        condition = weather_info['condition']
        weather_confidence = weather_info['confidence']
        
        # Get weather-specific configuration
        config = self.weather_configs[condition]
        
        # Apply preprocessing
        processed_sequence = self._apply_weather_preprocessing(
            sequence, config['preprocessing']
        )
        
        # Get base prediction
        prediction = self.base_agent.act(processed_sequence, training=False)
        base_confidence = getattr(self.base_agent, '_last_confidence', 0.5)
        
        # Apply weather adaptations
        adapted_confidence = base_confidence * config['confidence_boost']
        threshold_adjustment = config['threshold_adjustment']
        
        # Store detection result
        detection_result = {
            'weather_condition': condition.value,
            'weather_confidence': weather_confidence,
            'base_prediction': prediction,
            'adapted_confidence': adapted_confidence,
            'threshold_adjustment': threshold_adjustment,
            'preprocessing_applied': config['preprocessing']
        }
        
        self.detection_history.append(detection_result)
        
        return {
            'prediction': prediction,
            'confidence': adapted_confidence,
            'weather_condition': condition.value,
            'weather_confidence': weather_confidence,
            'adaptations_applied': config['preprocessing']
        }
    
    def _apply_weather_preprocessing(self, sequence, preprocessing_list):
        """Apply weather-specific preprocessing"""
        processed = sequence.copy()
        
        for preprocessing in preprocessing_list:
            if preprocessing == 'rain_removal':
                processed = self._remove_rain_effects(processed)
            elif preprocessing == 'fog_removal':
                processed = self._remove_fog_effects(processed)
            elif preprocessing == 'snow_removal':
                processed = self._remove_snow_effects(processed)
            elif preprocessing == 'brightness_enhancement':
                processed = self._enhance_brightness(processed)
            elif preprocessing == 'contrast_enhancement':
                processed = self._enhance_contrast(processed)
            elif preprocessing == 'low_light_enhancement':
                processed = self._enhance_low_light(processed)
            elif preprocessing == 'noise_reduction':
                processed = self._reduce_noise(processed)
        
        return processed
    
    def _remove_rain_effects(self, sequence):
        """Remove rain streaks and effects"""
        # Simplified rain removal using median filtering
        cleaned = sequence.copy()
        
        for frame_idx in range(sequence.shape[0]):
            frame = (cleaned[frame_idx] * 255).astype(np.uint8)
            
            if len(frame.shape) == 3:
                for channel in range(3):
                    frame[:,:,channel] = cv2.medianBlur(frame[:,:,channel], 3)
            else:
                frame = cv2.medianBlur(frame, 3)
            
            cleaned[frame_idx] = frame / 255.0
        
        return cleaned
    
    def _remove_fog_effects(self, sequence):
        """Remove fog effects using dark channel prior"""
        cleaned = sequence.copy()
        
        for frame_idx in range(sequence.shape[0]):
            frame = cleaned[frame_idx]
            
            # Simple fog removal using histogram stretching
            if len(frame.shape) == 3:
                for channel in range(3):
                    channel_data = frame[:,:,channel]
                    p2, p98 = np.percentile(channel_data, (2, 98))
                    frame[:,:,channel] = np.clip((channel_data - p2) / (p98 - p2), 0, 1)
            
            cleaned[frame_idx] = frame
        
        return cleaned
    
    def _remove_snow_effects(self, sequence):
        """Remove snow effects"""
        cleaned = sequence.copy()
        
        for frame_idx in range(sequence.shape[0]):
            frame = (cleaned[frame_idx] * 255).astype(np.uint8)
            
            # Snow removal using morphological operations
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                # Detect bright spots (snow)
                _, snow_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                
                # Remove small snow particles
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                snow_mask = cv2.morphologyEx(snow_mask, cv2.MORPH_OPENING, kernel)
                
                # Inpaint snow regions
                frame = cv2.inpaint(frame, snow_mask, 3, cv2.INPAINT_TELEA)
            
            cleaned[frame_idx] = frame / 255.0
        
        return cleaned
    
    def _enhance_brightness(self, sequence):
        """Enhance brightness for low-light conditions"""
        enhanced = sequence.copy()
        
        for frame_idx in range(sequence.shape[0]):
            frame = enhanced[frame_idx]
            
            # Gamma correction
            gamma = 1.3
            enhanced[frame_idx] = np.power(frame, 1/gamma)
        
        return np.clip(enhanced, 0, 1)
    
    def _enhance_contrast(self, sequence):
        """Enhance contrast"""
        enhanced = sequence.copy()
        
        for frame_idx in range(sequence.shape[0]):
            frame = (enhanced[frame_idx] * 255).astype(np.uint8)
            
            if len(frame.shape) == 3:
                lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                
                enhanced[frame_idx] = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB) / 255.0
        
        return enhanced
    
    def _enhance_low_light(self, sequence):
        """Enhanced low-light processing"""
        enhanced = sequence.copy()
        
        for frame_idx in range(sequence.shape[0]):
            frame = enhanced[frame_idx]
            
            # Adaptive gamma correction based on brightness
            avg_brightness = np.mean(frame)
            if avg_brightness < 0.3:
                gamma = 0.6  # Strong enhancement
            elif avg_brightness < 0.5:
                gamma = 0.8  # Moderate enhancement
            else:
                gamma = 1.0  # No enhancement needed
            
            enhanced[frame_idx] = np.power(frame, gamma)
        
        return np.clip(enhanced, 0, 1)
    
    def _reduce_noise(self, sequence):
        """Reduce noise in low-light conditions"""
        denoised = sequence.copy()
        
        for frame_idx in range(sequence.shape[0]):
            frame = (denoised[frame_idx] * 255).astype(np.uint8)
            
            # Non-local means denoising
            if len(frame.shape) == 3:
                frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
            else:
                frame = cv2.fastNlMeansDenoising(frame, None, 10, 7, 21)
            
            denoised[frame_idx] = frame / 255.0
        
        return denoised
    
    def get_weather_statistics(self):
        """Get weather adaptation statistics"""
        if not self.detection_history:
            return {}
        
        weather_counts = {}
        adaptations_used = {}
        
        for detection in self.detection_history:
            condition = detection['weather_condition']
            weather_counts[condition] = weather_counts.get(condition, 0) + 1
            
            for adaptation in detection['preprocessing_applied']:
                adaptations_used[adaptation] = adaptations_used.get(adaptation, 0) + 1
        
        total_detections = len(self.detection_history)
        
        return {
            'total_detections': total_detections,
            'weather_distribution': {
                condition: (count / total_detections) * 100
                for condition, count in weather_counts.items()
            },
            'adaptation_usage': {
                adaptation: (count / total_detections) * 100
                for adaptation, count in adaptations_used.items()
            },
            'most_common_weather': max(weather_counts, key=weather_counts.get),
            'most_used_adaptation': max(adaptations_used, key=adaptations_used.get) if adaptations_used else None
        }
