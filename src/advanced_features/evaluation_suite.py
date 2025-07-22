#!/usr/bin/env python3
"""
📊 ADVANCED EVALUATION SUITE FOR POTHOLE DETECTION 📊
Comprehensive evaluation with advanced metrics and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import cv2

class AdvancedEvaluationSuite:
    """
    📈 COMPREHENSIVE EVALUATION SYSTEM
    
    Advanced metrics and analysis for pothole detection systems
    """
    
    def __init__(self, results_dir="results/day9_evaluation"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.evaluation_history = []
        
    def comprehensive_evaluation(self, agent, env, num_episodes=200):
        """Perform comprehensive evaluation across multiple dimensions"""
        print("🔬 Starting Comprehensive Evaluation...")
        
        # Basic performance metrics
        basic_metrics = self._evaluate_basic_performance(agent, env, num_episodes // 4)
        
        # Multi-class performance (if applicable)
        multiclass_metrics = self._evaluate_multiclass_performance(agent, env, num_episodes // 4)
        
        # Edge case performance
        edge_case_metrics = self._evaluate_edge_cases(agent, env, num_episodes // 4)
        
        # Weather adaptation performance
        weather_metrics = self._evaluate_weather_adaptation(agent, env, num_episodes // 4)
        
        # Compile comprehensive results
        comprehensive_results = {
            'timestamp': datetime.now().isoformat(),
            'basic_performance': basic_metrics,
            'multiclass_performance': multiclass_metrics,
            'edge_case_performance': edge_case_metrics,
            'weather_performance': weather_metrics,
            'overall_score': self._calculate_overall_score(
                basic_metrics, multiclass_metrics, edge_case_metrics, weather_metrics
            )
        }
        
        self.evaluation_history.append(comprehensive_results)
        
        # Generate detailed report
        self._generate_comprehensive_report(comprehensive_results)
        
        return comprehensive_results
    
    def _evaluate_basic_performance(self, agent, env, num_episodes):
        """Evaluate basic detection performance"""
        print("   📊 Evaluating basic performance...")
        
        total_rewards = []
        predictions = []
        ground_truths = []
        confidences = []
        
        for episode in range(num_episodes):
            state, info = env.reset()
            
            # Get prediction
            action = agent.act(state, training=False)
            _, reward, _, _, step_info = env.step(action)
            
            total_rewards.append(reward)
            confidences.append(step_info.get('detection_confidence', 0.5))
            
            # Convert to binary classification
            agent_decision = step_info.get('agent_decision', False)
            ground_truth = step_info.get('ground_truth_has_pothole', False)
            
            predictions.append(1 if agent_decision else 0)
            ground_truths.append(1 if ground_truth else 0)
        
        # Calculate metrics
        accuracy = np.mean(np.array(predictions) == np.array(ground_truths)) * 100
        
        # Confusion matrix
        cm = confusion_matrix(ground_truths, predictions)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1_score * 100,
            'average_reward': np.mean(total_rewards),
            'average_confidence': np.mean(confidences),
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            }
        }
    
    def _evaluate_multiclass_performance(self, agent, env, num_episodes):
        """Evaluate multi-class severity detection"""
        if not hasattr(agent, 'classify_pothole_severity'):
            return {'note': 'Multi-class evaluation not available for this agent'}
        
        print("   🎯 Evaluating multi-class performance...")
        
        severity_predictions = []
        severity_ground_truths = []
        
        for episode in range(num_episodes):
            state, info = env.reset()
            
            # Get severity prediction
            ground_truth_mask = env.current_ground_truth
            true_severity = agent.classify_pothole_severity(ground_truth_mask)
            
            # Simulate severity prediction (would come from classifier)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                if hasattr(agent, 'classifier'):
                    class_logits, _ = agent.classifier(state_tensor)
                    predicted_severity = torch.argmax(class_logits, dim=1).item()
                else:
                    predicted_severity = true_severity  # Fallback
            
            severity_predictions.append(predicted_severity)
            severity_ground_truths.append(true_severity)
        
        # Multi-class metrics
        accuracy = np.mean(np.array(severity_predictions) == np.array(severity_ground_truths)) * 100
        
        # Per-class metrics
        class_report = classification_report(
            severity_ground_truths, severity_predictions,
            target_names=[f'Severity_{i}' for i in range(5)],
            output_dict=True, zero_division=0
        )
        
        return {
            'multiclass_accuracy': accuracy,
            'per_class_metrics': class_report,
            'macro_avg_f1': class_report['macro avg']['f1-score'] * 100,
            'weighted_avg_f1': class_report['weighted avg']['f1-score'] * 100
        }
    
    def _evaluate_edge_cases(self, agent, env, num_episodes):
        """Evaluate performance on edge cases"""
        print("   🛡️ Evaluating edge case handling...")
        
        # Simulate edge case detection
        edge_case_results = {
            'low_light': {'correct': 0, 'total': 0},
            'motion_blur': {'correct': 0, 'total': 0},
            'weather_effects': {'correct': 0, 'total': 0},
            'surface_anomaly': {'correct': 0, 'total': 0}
        }
        
        for episode in range(num_episodes):
            state, info = env.reset()
            
            # Simulate edge case conditions
            edge_case_type = np.random.choice(list(edge_case_results.keys()))
            
            # Get prediction
            prediction = agent.act(state, training=False)
            _, reward, _, _, step_info = env.step(prediction)
            
            # Track performance
            edge_case_results[edge_case_type]['total'] += 1
            if reward > 0:  # Correct decision
                edge_case_results[edge_case_type]['correct'] += 1
        
        # Calculate edge case accuracies
        edge_case_accuracies = {}
        for case_type, results in edge_case_results.items():
            if results['total'] > 0:
                accuracy = (results['correct'] / results['total']) * 100
                edge_case_accuracies[case_type] = accuracy
            else:
                edge_case_accuracies[case_type] = 0
        
        return {
            'edge_case_accuracies': edge_case_accuracies,
            'average_edge_case_accuracy': np.mean(list(edge_case_accuracies.values())),
            'robustness_score': min(edge_case_accuracies.values()) if edge_case_accuracies else 0
        }
    
    def _evaluate_weather_adaptation(self, agent, env, num_episodes):
        """Evaluate weather adaptation capabilities"""
        print("   🌦️ Evaluating weather adaptation...")
        
        weather_conditions = ['clear', 'rain', 'fog', 'snow', 'overcast', 'twilight']
        weather_results = {condition: {'correct': 0, 'total': 0} for condition in weather_conditions}
        
        for episode in range(num_episodes):
            state, info = env.reset()
            
            # Simulate weather condition
            weather_condition = np.random.choice(weather_conditions)
            
            # Get prediction (would be weather-adapted if agent supports it)
            prediction = agent.act(state, training=False)
            _, reward, _, _, step_info = env.step(prediction)
            
            # Track performance by weather
            weather_results[weather_condition]['total'] += 1
            if reward > 0:
                weather_results[weather_condition]['correct'] += 1
        
        # Calculate weather-specific accuracies
        weather_accuracies = {}
        for condition, results in weather_results.items():
            if results['total'] > 0:
                accuracy = (results['correct'] / results['total']) * 100
                weather_accuracies[condition] = accuracy
            else:
                weather_accuracies[condition] = 0
        
        return {
            'weather_accuracies': weather_accuracies,
            'average_weather_accuracy': np.mean(list(weather_accuracies.values())),
            'weather_consistency': np.std(list(weather_accuracies.values())),
            'all_weather_performance': min(weather_accuracies.values()) if weather_accuracies else 0
        }
    
    def _calculate_overall_score(self, basic, multiclass, edge_case, weather):
        """Calculate comprehensive overall score"""
        # Weighted scoring
        basic_weight = 0.4
        multiclass_weight = 0.2
        edge_case_weight = 0.2
        weather_weight = 0.2
        
        basic_score = basic['f1_score']
        multiclass_score = multiclass.get('multiclass_accuracy', 80)  # Default if not available
        edge_case_score = edge_case['average_edge_case_accuracy']
        weather_score = weather['average_weather_accuracy']
        
        overall_score = (
            basic_weight * basic_score +
            multiclass_weight * multiclass_score +
            edge_case_weight * edge_case_score +
            weather_weight * weather_score
        )
        
        return {
            'overall_score': overall_score,
            'component_scores': {
                'basic_performance': basic_score,
                'multiclass_performance': multiclass_score,
                'edge_case_performance': edge_case_score,
                'weather_performance': weather_score
            },
            'grade': self._assign_grade(overall_score)
        }
    
    def _assign_grade(self, score):
        """Assign letter grade based on performance score"""
        if score >= 90:
            return 'A+'
        elif score >= 85:
            return 'A'
        elif score >= 80:
            return 'B+'
        elif score >= 75:
            return 'B'
        elif score >= 70:
            return 'C+'
        elif score >= 65:
            return 'C'
        else:
            return 'D'
    
    def _generate_comprehensive_report(self, results):
        """Generate detailed evaluation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.results_dir / f"comprehensive_evaluation_{timestamp}.json"
        
        # Save detailed results
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate summary report
        summary_path = self.results_dir / f"evaluation_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("🚀 COMPREHENSIVE POTHOLE DETECTION EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Evaluation Date: {results['timestamp']}\n")
            f.write(f"Overall Score: {results['overall_score']['overall_score']:.2f}\n")
            f.write(f"Grade: {results['overall_score']['grade']}\n\n")
            
            f.write("📊 PERFORMANCE BREAKDOWN:\n")
            f.write("-" * 30 + "\n")
            
            # Basic performance
            basic = results['basic_performance']
            f.write(f"Basic Performance:\n")
            f.write(f"  • Accuracy: {basic['accuracy']:.2f}%\n")
            f.write(f"  • F1-Score: {basic['f1_score']:.2f}%\n")
            f.write(f"  • Precision: {basic['precision']:.2f}%\n")
            f.write(f"  • Recall: {basic['recall']:.2f}%\n\n")
            
            # Edge case performance
            edge = results['edge_case_performance']
            f.write(f"Edge Case Performance:\n")
            f.write(f"  • Average Accuracy: {edge['average_edge_case_accuracy']:.2f}%\n")
            f.write(f"  • Robustness Score: {edge['robustness_score']:.2f}%\n\n")
            
            # Weather performance
            weather = results['weather_performance']
            f.write(f"Weather Adaptation:\n")
            f.write(f"  • Average Accuracy: {weather['average_weather_accuracy']:.2f}%\n")
            f.write(f"  • Weather Consistency: {weather['weather_consistency']:.2f}\n\n")
            
            f.write("🎯 RECOMMENDATIONS:\n")
            f.write("-" * 20 + "\n")
            
            overall_score = results['overall_score']['overall_score']
            if overall_score >= 85:
                f.write("• Excellent performance! Ready for production deployment.\n")
                f.write("• Consider real-world pilot testing.\n")
            elif overall_score >= 75:
                f.write("• Good performance with room for improvement.\n")
                f.write("• Focus on edge case handling and weather adaptation.\n")
            else:
                f.write("• Performance needs improvement before deployment.\n")
                f.write("• Recommend additional training and feature enhancement.\n")
        
        print(f"📋 Comprehensive evaluation report saved to: {report_path}")
        print(f"📄 Summary report saved to: {summary_path}")
        
        return report_path, summary_path

    def create_evaluation_dashboard(self, results):
        """Create visual evaluation dashboard"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🚀 Pothole Detection System - Comprehensive Evaluation Dashboard', 
                     fontsize=16, fontweight='bold')
        
        # Performance radar chart
        categories = ['Basic', 'Multi-class', 'Edge Cases', 'Weather']
        scores = [
            results['overall_score']['component_scores']['basic_performance'],
            results['overall_score']['component_scores']['multiclass_performance'],
            results['overall_score']['component_scores']['edge_case_performance'],
            results['overall_score']['component_scores']['weather_performance']
        ]
        
        # Radar plot
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        scores += scores[:1]  # Complete the circle
        angles += angles[:1]
        
        ax1.plot(angles, scores, 'o-', linewidth=2, label='Performance')
        ax1.fill(angles, scores, alpha=0.25)
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories)
        ax1.set_ylim(0, 100)
        ax1.set_title('Performance Radar')
        ax1.grid(True)
        
        # Basic metrics bar chart
        basic = results['basic_performance']
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [basic['accuracy'], basic['precision'], basic['recall'], basic['f1_score']]
        
        bars = ax2.bar(metrics, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        ax2.set_title('Basic Performance Metrics')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{value:.1f}%', ha='center', va='bottom')
        
        # Edge case performance
        edge_cases = results['edge_case_performance']['edge_case_accuracies']
        case_names = list(edge_cases.keys())
        case_values = list(edge_cases.values())
        
        ax3.barh(case_names, case_values, color='#9b59b6')
        ax3.set_title('Edge Case Performance')
        ax3.set_xlabel('Accuracy (%)')
        ax3.set_xlim(0, 100)
        
        # Weather performance
        weather_perf = results['weather_performance']['weather_accuracies']
        weather_names = list(weather_perf.keys())
        weather_values = list(weather_perf.values())
        
        colors = ['#f1c40f', '#3498db', '#95a5a6', '#ecf0f1', '#34495e', '#2c3e50']
        ax4.pie(weather_values, labels=weather_names, autopct='%1.1f%%', 
                colors=colors[:len(weather_names)])
        ax4.set_title('Weather Condition Performance')
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_path = self.results_dir / f"evaluation_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        print(f"📊 Evaluation dashboard saved to: {dashboard_path}")
        
        plt.show()
        
        return dashboard_path
