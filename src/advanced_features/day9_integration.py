#!/usr/bin/env python3
"""
🚀 DAY 9 INTEGRATION SCRIPT 🚀
Integrate all advanced features for comprehensive testing
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.multi_class.multi_class_detector import MultiClassRLAgent
from src.edge_cases.edge_case_handler import RobustPotholeDetector
from src.weather_adaptation.weather_adaptive_agent import WeatherAdaptiveAgent
from src.advanced_features.evaluation_suite import AdvancedEvaluationSuite
from src.environment.pothole_env import VideoBasedPotholeEnv
from src.agents.advanced_dqn import AdvancedDQNAgent

def main():
    """Day 9 comprehensive testing pipeline"""
    print("🚀 DAY 9: ADVANCED FEATURES INTEGRATION")
    print("=" * 60)
    
    # Create environment
    env = VideoBasedPotholeEnv(
        split='train',
        balanced=True,
        target_sequences=2000,
        verbose=True
    )
    
    # Create base agent
    base_agent = AdvancedDQNAgent(
        input_shape=(5, 224, 224, 3),
        num_actions=5,
        use_double_dqn=True,
        use_dueling=True,
        use_prioritized_replay=True
    )
    
    # Create advanced agents
    print("\n🧠 Creating Advanced Agent Stack...")
    
    # Multi-class agent
    multiclass_agent = MultiClassRLAgent(
        input_shape=(5, 224, 224, 3),
        num_classes=5
    )
    
    # Robust agent with edge case handling
    robust_agent = RobustPotholeDetector(base_agent)
    
    # Weather-adaptive agent
    weather_agent = WeatherAdaptiveAgent(base_agent)
    
    # Evaluation suite
    evaluator = AdvancedEvaluationSuite()
    
    print("✅ Advanced agent stack created successfully!")
    
    # Comprehensive evaluation
    print("\n📊 Starting Comprehensive Evaluation...")
    results = evaluator.comprehensive_evaluation(base_agent, env, num_episodes=100)
    
    # Create evaluation dashboard
    dashboard_path = evaluator.create_evaluation_dashboard(results)
    
    print(f"\n🏆 Day 9 Integration Complete!")
    print(f"Overall Score: {results['overall_score']['overall_score']:.2f}")
    print(f"Grade: {results['overall_score']['grade']}")
    print(f"📊 Dashboard: {dashboard_path}")

if __name__ == "__main__":
    main()
