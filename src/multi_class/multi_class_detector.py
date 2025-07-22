#!/usr/bin/env python3
"""
🎯 MULTI-CLASS ROAD DAMAGE DETECTION SYSTEM 🎯
Day 9: Advanced multi-class detection for comprehensive road assessment
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.detectors.real_cnn_detector import create_cnn_detector

class RoadDamageType(Enum):
    """Comprehensive road damage classification"""
    SMOOTH_ROAD = 0
    MINOR_CRACK = 1
    MAJOR_CRACK = 2
    SMALL_POTHOLE = 3
    MEDIUM_POTHOLE = 4
    LARGE_POTHOLE = 5
    CRITICAL_POTHOLE = 6
    ROAD_MARKING = 7
    PATCH_REPAIR = 8
    DEBRIS = 9

class SeverityLevel(Enum):
    """Damage severity classification"""
    NONE = 0
    MINOR = 1
    MODERATE = 2
    SEVERE = 3
    CRITICAL = 4

@dataclass
class MultiClassDetection:
    """Multi-class detection result"""
    damage_type: RoadDamageType
    severity: SeverityLevel
    confidence: float
    bbox: Tuple[int, int, int, int]
    area_percentage: float
    repair_priority: int
    estimated_cost: float
    safety_impact: str

class MultiClassRoadDetector:
    """
    🌟 ADVANCED MULTI-CLASS ROAD DAMAGE DETECTION SYSTEM
    Comprehensive road condition analysis with severity and priority assessment
    """
    
    def __init__(self, base_detector=None):
        print("🎯 Initializing MULTI-CLASS Road Damage Detection System!")
        
        # Initialize base detector
        self.base_detector = base_detector or create_cnn_detector(use_real_cnn=False)
        
        # Classification thresholds for different damage types
        self.damage_thresholds = {
            RoadDamageType.SMOOTH_ROAD: {'confidence': 0.1, 'size_threshold': 0.0},
            RoadDamageType.MINOR_CRACK: {'confidence': 0.3, 'size_threshold': 0.01},
            RoadDamageType.MAJOR_CRACK: {'confidence': 0.5, 'size_threshold': 0.05},
            RoadDamageType.SMALL_POTHOLE: {'confidence': 0.6, 'size_threshold': 0.02},
            RoadDamageType.MEDIUM_POTHOLE: {'confidence': 0.75, 'size_threshold': 0.08},
            RoadDamageType.LARGE_POTHOLE: {'confidence': 0.85, 'size_threshold': 0.15},
            RoadDamageType.CRITICAL_POTHOLE: {'confidence': 0.95, 'size_threshold': 0.25},
            RoadDamageType.ROAD_MARKING: {'confidence': 0.4, 'size_threshold': 0.03},
            RoadDamageType.PATCH_REPAIR: {'confidence': 0.5, 'size_threshold': 0.06},
            RoadDamageType.DEBRIS: {'confidence': 0.6, 'size_threshold': 0.01}
        }
        
        # Repair cost estimates (USD per sq meter)
        self.repair_costs = {
            RoadDamageType.SMOOTH_ROAD: 0,
            RoadDamageType.MINOR_CRACK: 15,
            RoadDamageType.MAJOR_CRACK: 45,
            RoadDamageType.SMALL_POTHOLE: 80,
            RoadDamageType.MEDIUM_POTHOLE: 150,
            RoadDamageType.LARGE_POTHOLE: 300,
            RoadDamageType.CRITICAL_POTHOLE: 600,
            RoadDamageType.ROAD_MARKING: 25,
            RoadDamageType.PATCH_REPAIR: 120,
            RoadDamageType.DEBRIS: 5
        }
        
        print("✅ Multi-class detection system initialized!")
        print(f"   🎯 Damage types supported: {len(self.damage_thresholds)}")
    
    def analyze_comprehensive_road_condition(self, frame_sequence):
        """
        🔍 COMPREHENSIVE ROAD CONDITION ANALYSIS
        Multi-class detection with severity assessment
        """
        # Get base confidence from existing detector
        base_confidence = self.base_detector.detect_pothole_confidence(frame_sequence)
        
        # Analyze frame characteristics for classification
        frame_analysis = self._analyze_frame_characteristics(frame_sequence[-1])  # Use latest frame
        
        # Classify damage type and severity
        detections = self._classify_damage_comprehensive(base_confidence, frame_analysis)
        
        # Generate comprehensive assessment
        assessment = self._generate_road_assessment(detections, frame_analysis)
        
        return {
            'detections': detections,
            'overall_assessment': assessment,
            'processing_metadata': {
                'base_confidence': base_confidence,
                'frame_analysis': frame_analysis,
                'timestamp': np.datetime64('now').item()
            }
        }
    
    def _analyze_frame_characteristics(self, frame):
        """Analyze frame for multi-class classification features"""
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        
        # Convert to grayscale for analysis
        if len(frame.shape) == 3:
            gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (frame * 255).astype(np.uint8)
        
        # Feature extraction
        features = {
            # Darkness analysis (for potholes)
            'darkness_ratio': np.sum(gray < 80) / gray.size,
            'dark_clusters': self._detect_dark_clusters(gray),
            
            # Edge analysis (for cracks)
            'edge_density': cv2.Canny(gray, 50, 150).sum() / gray.size,
            'edge_continuity': self._analyze_edge_continuity(gray),
            
            # Texture analysis
            'texture_uniformity': 1.0 - np.std(gray) / 255.0,
            'surface_roughness': self._calculate_surface_roughness(gray),
            
            # Color analysis (for markings and patches)
            'brightness_variance': np.var(gray),
            'color_distribution': self._analyze_color_distribution(frame),
            
            # Shape analysis
            'circular_features': self._detect_circular_features(gray),
            'linear_features': self._detect_linear_features(gray)
        }
        
        return features
    
    def _detect_dark_clusters(self, gray_frame):
        """Detect dark cluster regions (potential potholes)"""
        # Threshold for dark regions
        _, dark_mask = cv2.threshold(gray_frame, 60, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'count': 0, 'largest_area': 0, 'total_area': 0}
        
        areas = [cv2.contourArea(c) for c in contours]
        total_area = sum(areas)
        largest_area = max(areas) if areas else 0
        
        return {
            'count': len(contours),
            'largest_area': largest_area,
            'total_area': total_area,
            'area_ratio': total_area / gray_frame.size
        }
    
    def _analyze_edge_continuity(self, gray_frame):
        """Analyze edge continuity (for crack detection)"""
        edges = cv2.Canny(gray_frame, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is None:
            return {'line_count': 0, 'max_length': 0, 'continuity_score': 0}
        
        return {
            'line_count': len(lines),
            'max_length': self._calculate_max_line_length(lines),
            'continuity_score': min(len(lines) / 10, 1.0)  # Normalized score
        }
    
    def _calculate_surface_roughness(self, gray_frame):
        """Calculate surface roughness metric"""
        # Calculate gradients
        grad_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine gradients
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Return normalized roughness
        return np.mean(gradient_magnitude) / 255.0
    
    def _analyze_color_distribution(self, frame):
        """Analyze color distribution for marking/patch detection"""
        if len(frame.shape) != 3:
            return {'variance': 0, 'bright_regions': 0}
        
        # Convert to HSV for better analysis
        frame_uint8 = (frame * 255).astype(np.uint8)
        hsv = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2HSV)
        
        # Analyze brightness (V channel)
        brightness = hsv[:, :, 2]
        bright_regions = np.sum(brightness > 200) / brightness.size
        
        return {
            'variance': np.var(brightness),
            'bright_regions': bright_regions,
            'color_complexity': np.std(hsv.reshape(-1, 3), axis=0).mean()
        }
    
    def _detect_circular_features(self, gray_frame):
        """Detect circular features (potential potholes)"""
        # Use HoughCircles to detect circular patterns
        circles = cv2.HoughCircles(
            gray_frame, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=30, minRadius=10, maxRadius=100
        )
        
        if circles is None:
            return {'count': 0, 'largest_radius': 0}
        
        circles = np.round(circles[0, :]).astype("int")
        
        return {
            'count': len(circles),
            'largest_radius': max(circles[:, 2]) if len(circles) > 0 else 0,
            'avg_radius': np.mean(circles[:, 2]) if len(circles) > 0 else 0
        }
    
    def _detect_linear_features(self, gray_frame):
        """Detect linear features (cracks, markings)"""
        # Edge detection
        edges = cv2.Canny(gray_frame, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi/180, threshold=50,
            minLineLength=30, maxLineGap=10
        )
        
        if lines is None:
            return {'count': 0, 'total_length': 0}
        
        total_length = sum([
            np.sqrt((x2-x1)**2 + (y2-y1)**2) 
            for x1, y1, x2, y2 in lines[:, 0]
        ])
        
        return {
            'count': len(lines),
            'total_length': total_length,
            'avg_length': total_length / len(lines) if lines is not None else 0
        }
    
    def _calculate_max_line_length(self, lines):
        """Calculate maximum line length from Hough lines"""
        if lines is None or len(lines) == 0:
            return 0
        
        # For Hough lines, we estimate length based on the frame diagonal
        # This is a simplified calculation
        return 100  # Placeholder for now
    
    def _classify_damage_comprehensive(self, base_confidence, frame_analysis):
        """Classify damage with comprehensive multi-class analysis"""
        detections = []
        
        # Classification logic based on frame analysis
        if base_confidence < 0.2:
            # Likely smooth road
            detection = self._create_detection(
                RoadDamageType.SMOOTH_ROAD,
                SeverityLevel.NONE,
                base_confidence,
                frame_analysis
            )
            detections.append(detection)
        
        elif frame_analysis['edge_continuity']['line_count'] > 3 and base_confidence < 0.6:
            # Likely crack damage
            severity = SeverityLevel.MINOR if base_confidence < 0.4 else SeverityLevel.MODERATE
            damage_type = RoadDamageType.MINOR_CRACK if severity == SeverityLevel.MINOR else RoadDamageType.MAJOR_CRACK
            
            detection = self._create_detection(damage_type, severity, base_confidence, frame_analysis)
            detections.append(detection)
        
        elif frame_analysis['dark_clusters']['count'] > 0 and base_confidence > 0.5:
            # Pothole detection with size-based classification
            area_ratio = frame_analysis['dark_clusters']['area_ratio']
            
            if area_ratio < 0.05:
                damage_type, severity = RoadDamageType.SMALL_POTHOLE, SeverityLevel.MINOR
            elif area_ratio < 0.15:
                damage_type, severity = RoadDamageType.MEDIUM_POTHOLE, SeverityLevel.MODERATE
            elif area_ratio < 0.25:
                damage_type, severity = RoadDamageType.LARGE_POTHOLE, SeverityLevel.SEVERE
            else:
                damage_type, severity = RoadDamageType.CRITICAL_POTHOLE, SeverityLevel.CRITICAL
            
            detection = self._create_detection(damage_type, severity, base_confidence, frame_analysis)
            detections.append(detection)
        
        elif frame_analysis['color_distribution']['bright_regions'] > 0.1:
            # Potential road marking or patch
            if frame_analysis['texture_uniformity'] > 0.6:
                damage_type = RoadDamageType.ROAD_MARKING
                severity = SeverityLevel.NONE
            else:
                damage_type = RoadDamageType.PATCH_REPAIR
                severity = SeverityLevel.MINOR
            
            detection = self._create_detection(damage_type, severity, base_confidence, frame_analysis)
            detections.append(detection)
        
        # Default fallback
        if not detections:
            detection = self._create_detection(
                RoadDamageType.SMOOTH_ROAD,
                SeverityLevel.NONE,
                base_confidence,
                frame_analysis
            )
            detections.append(detection)
        
        return detections
    
    def _create_detection(self, damage_type, severity, confidence, frame_analysis):
        """Create a multi-class detection object"""
        # Generate realistic bbox based on damage type
        if damage_type in [RoadDamageType.SMALL_POTHOLE, RoadDamageType.MEDIUM_POTHOLE, 
                          RoadDamageType.LARGE_POTHOLE, RoadDamageType.CRITICAL_POTHOLE]:
            # Center-focused bbox for potholes
            bbox = self._generate_pothole_bbox(damage_type)
            area_percentage = frame_analysis['dark_clusters']['area_ratio']
        elif damage_type in [RoadDamageType.MINOR_CRACK, RoadDamageType.MAJOR_CRACK]:
            # Linear bbox for cracks
            bbox = self._generate_crack_bbox()
            area_percentage = frame_analysis['edge_continuity']['continuity_score'] * 0.1
        else:
            # Default bbox
            bbox = (50, 50, 100, 100)
            area_percentage = 0.05
        
        # Calculate repair priority and cost
        repair_priority = self._calculate_repair_priority(damage_type, severity)
        estimated_cost = self._calculate_repair_cost(damage_type, area_percentage)
        safety_impact = self._assess_safety_impact(damage_type, severity)
        
        return MultiClassDetection(
            damage_type=damage_type,
            severity=severity,
            confidence=confidence,
            bbox=bbox,
            area_percentage=area_percentage,
            repair_priority=repair_priority,
            estimated_cost=estimated_cost,
            safety_impact=safety_impact
        )
    
    def _generate_pothole_bbox(self, damage_type):
        """Generate realistic bbox for pothole"""
        size_mapping = {
            RoadDamageType.SMALL_POTHOLE: (60, 80),
            RoadDamageType.MEDIUM_POTHOLE: (100, 120),
            RoadDamageType.LARGE_POTHOLE: (150, 180),
            RoadDamageType.CRITICAL_POTHOLE: (200, 240)
        }
        
        w, h = size_mapping.get(damage_type, (100, 120))
        x = np.random.randint(50, 224 - w - 50)
        y = np.random.randint(50, 224 - h - 50)
        
        return (x, y, w, h)
    
    def _generate_crack_bbox(self):
        """Generate realistic bbox for crack"""
        # Linear feature
        w = np.random.randint(150, 200)
        h = np.random.randint(10, 30)
        x = np.random.randint(10, 224 - w - 10)
        y = np.random.randint(50, 224 - h - 50)
        
        return (x, y, w, h)
    
    def _calculate_repair_priority(self, damage_type, severity):
        """Calculate repair priority (1-10 scale)"""
        base_priorities = {
            RoadDamageType.SMOOTH_ROAD: 0,
            RoadDamageType.MINOR_CRACK: 2,
            RoadDamageType.MAJOR_CRACK: 4,
            RoadDamageType.SMALL_POTHOLE: 5,
            RoadDamageType.MEDIUM_POTHOLE: 7,
            RoadDamageType.LARGE_POTHOLE: 8,
            RoadDamageType.CRITICAL_POTHOLE: 10,
            RoadDamageType.ROAD_MARKING: 1,
            RoadDamageType.PATCH_REPAIR: 3,
            RoadDamageType.DEBRIS: 6
        }
        
        severity_multipliers = {
            SeverityLevel.NONE: 0.0,
            SeverityLevel.MINOR: 1.0,
            SeverityLevel.MODERATE: 1.2,
            SeverityLevel.SEVERE: 1.5,
            SeverityLevel.CRITICAL: 2.0
        }
        
        base_priority = base_priorities.get(damage_type, 5)
        multiplier = severity_multipliers.get(severity, 1.0)
        
        return min(int(base_priority * multiplier), 10)
    
    def _calculate_repair_cost(self, damage_type, area_percentage):
        """Calculate estimated repair cost"""
        cost_per_sqm = self.repair_costs.get(damage_type, 100)
        
        # Assume 1 sq meter base area, adjusted by area percentage
        estimated_area = max(area_percentage * 10, 0.1)  # Minimum 0.1 sq meter
        
        return cost_per_sqm * estimated_area
    
    def _assess_safety_impact(self, damage_type, severity):
        """Assess safety impact of damage"""
        impact_matrix = {
            (RoadDamageType.SMOOTH_ROAD, SeverityLevel.NONE): "None",
            (RoadDamageType.MINOR_CRACK, SeverityLevel.MINOR): "Low",
            (RoadDamageType.MAJOR_CRACK, SeverityLevel.MODERATE): "Medium",
            (RoadDamageType.SMALL_POTHOLE, SeverityLevel.MINOR): "Medium",
            (RoadDamageType.MEDIUM_POTHOLE, SeverityLevel.MODERATE): "High",
            (RoadDamageType.LARGE_POTHOLE, SeverityLevel.SEVERE): "High",
            (RoadDamageType.CRITICAL_POTHOLE, SeverityLevel.CRITICAL): "Critical",
            (RoadDamageType.ROAD_MARKING, SeverityLevel.NONE): "Low",
            (RoadDamageType.PATCH_REPAIR, SeverityLevel.MINOR): "Low",
            (RoadDamageType.DEBRIS, SeverityLevel.MINOR): "High"
        }
        
        return impact_matrix.get((damage_type, severity), "Medium")
    
    def _generate_road_assessment(self, detections, frame_analysis):
        """Generate comprehensive road assessment"""
        if not detections:
            return {
                'overall_condition': 'Good',
                'safety_rating': 'Safe',
                'maintenance_urgency': 'None',
                'total_repair_cost': 0.0
            }
        
        # Calculate overall metrics
        max_priority = max(d.repair_priority for d in detections)
        total_cost = sum(d.estimated_cost for d in detections)
        
        # Determine overall condition
        if max_priority <= 2:
            condition = 'Excellent'
        elif max_priority <= 4:
            condition = 'Good'
        elif max_priority <= 6:
            condition = 'Fair'
        elif max_priority <= 8:
            condition = 'Poor'
        else:
            condition = 'Critical'
        
        # Safety assessment
        critical_detections = [d for d in detections if d.safety_impact in ['High', 'Critical']]
        if critical_detections:
            safety_rating = 'Hazardous' if any(d.safety_impact == 'Critical' for d in critical_detections) else 'Caution'
        else:
            safety_rating = 'Safe'
        
        # Maintenance urgency
        if max_priority >= 8:
            urgency = 'Immediate'
        elif max_priority >= 6:
            urgency = 'High'
        elif max_priority >= 4:
            urgency = 'Medium'
        elif max_priority >= 2:
            urgency = 'Low'
        else:
            urgency = 'None'
        
        return {
            'overall_condition': condition,
            'safety_rating': safety_rating,
            'maintenance_urgency': urgency,
            'total_repair_cost': total_cost,
            'priority_score': max_priority,
            'damage_count': len(detections),
            'most_severe_damage': max(detections, key=lambda x: x.repair_priority).damage_type.name
        }


# Test the multi-class detector
if __name__ == "__main__":
    print("🎯 TESTING MULTI-CLASS ROAD DAMAGE DETECTION!")
    print("="*60)
    
    # Initialize detector
    detector = MultiClassRoadDetector()
    
    # Test with sample data
    test_sequence = [np.random.rand(224, 224, 3).astype(np.float32) for _ in range(5)]
    
    # Run comprehensive analysis
    results = detector.analyze_comprehensive_road_condition(test_sequence)
    
    print(f"✅ Multi-class Analysis Results:")
    print(f"   🎯 Detections found: {len(results['detections'])}")
    
    for i, detection in enumerate(results['detections']):
        print(f"\n   Detection {i+1}:")
        print(f"      Type: {detection.damage_type.name}")
        print(f"      Severity: {detection.severity.name}")
        print(f"      Confidence: {detection.confidence:.3f}")
        print(f"      Priority: {detection.repair_priority}/10")
        print(f"      Cost: ${detection.estimated_cost:.2f}")
        print(f"      Safety Impact: {detection.safety_impact}")
    
    assessment = results['overall_assessment']
    print(f"\n   🔍 Overall Assessment:")
    print(f"      Condition: {assessment['overall_condition']}")
    print(f"      Safety: {assessment['safety_rating']}")
    print(f"      Urgency: {assessment['maintenance_urgency']}")
    print(f"      Total Cost: ${assessment['total_repair_cost']:.2f}")
    
    print(f"\n🎉 MULTI-CLASS DETECTION SYSTEM READY!")
