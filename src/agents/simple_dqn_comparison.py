#!/usr/bin/env python3
"""
🔥 SIMPLIFIED DQN COMPARISON - OPTION B 🔥
Quick and reliable DQN training with proven configurations
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.environment.pothole_env import VideoBasedPotholeEnv
from src.agents.dqn_agent import DQNAgent  # Use proven Day 5 agent
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import json
import time

def train_simple_dqn_comparison():
    """Simple DQN comparison with working configurations only"""
    
    print("🚀 SIMPLE DQN COMPARISON - OPTION B")
    print("="*50)
    
    results = []
    
    # SIMPLIFIED: Only test proven working configurations
    configurations = {
        'BASELINE_DQN': {
            'learning_rate': 0.0005,
            'gamma': 0.95,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'memory_size': 5000,
            'batch_size': 16,
            'target_update': 100
        },
        'OPTIMIZED_DQN': {
            'learning_rate': 0.0003,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.005,
            'epsilon_decay': 0.998,
            'memory_size': 8000,
            'batch_size': 24,
            'target_update': 75
        },
        'ENHANCED_DQN': {
            'learning_rate': 0.0002,
            'gamma': 0.995,
            'epsilon_start': 1.0,
            'epsilon_end': 0.001,
            'epsilon_decay': 0.999,
            'memory_size': 10000,
            'batch_size': 32,
            'target_update': 50
        }
    }
    
    print(f"🎯 Training {len(configurations)} simplified DQN configurations:")
    for i, config_name in enumerate(configurations.keys(), 1):
        print(f"   {i}. {config_name}")
    
    # Train each configuration
    for config_name, config_params in configurations.items():
        print(f"\n🚀 TRAINING {config_name}")
        print("-" * 40)
        
        try:
            # Create environment with conservative memory settings
            env = VideoBasedPotholeEnv(split='train', max_memory_mb=1024)
            
            # Use proven DQN agent from Day 5
            agent = DQNAgent(
                input_shape=(5, 224, 224, 3),
                num_actions=5,
                **config_params
            )
            
            print(f"   🎮 Environment: {len(env.episode_sequences)} sequences")
            print(f"   🧠 Agent: {sum(p.numel() for p in agent.q_network.parameters()):,} parameters")
            
            # Training loop - simplified
            training_start = time.time()
            training_rewards = []
            
            for episode in range(1, 151):  # 150 episodes for speed
                episode_result = agent.train_episode(env, max_steps=1)
                training_rewards.append(episode_result['total_reward'])
                
                if episode % 25 == 0:
                    avg_reward = np.mean(training_rewards[-25:])
                    print(f"   Episode {episode:3d} | Avg Reward: {avg_reward:+5.1f} | ε: {agent.epsilon:.3f}")
                
                # Quick evaluation every 50 episodes
                if episode % 50 == 0:
                    eval_result = agent.evaluate(env, num_episodes=20)
                    print(f"   🎯 EVAL: Accuracy={eval_result['accuracy']:.1f}%, Reward={eval_result['average_reward']:+5.1f}")
            
            # Final evaluation
            final_eval = agent.evaluate(env, num_episodes=50)
            training_time = time.time() - training_start
            
            # Store results
            results.append({
                'Configuration': config_name,
                'Final Accuracy': final_eval['accuracy'],
                'Average Reward': final_eval['average_reward'],
                'Training Time (min)': training_time / 60,
                'Correct Detections': final_eval['correct_decisions'],
                'False Positives': final_eval['false_positives'],
                'Missed Detections': final_eval['missed_detections'],
                'Episodes': 150
            })
            
            env.close()
            
            print(f"✅ {config_name} COMPLETE!")
            print(f"   🎯 Accuracy: {final_eval['accuracy']:.1f}%")
            print(f"   📊 Avg Reward: {final_eval['average_reward']:+5.1f}")
            print(f"   ⏱️ Time: {training_time/60:.1f} minutes")
            
        except Exception as e:
            print(f"❌ Error in {config_name}: {e}")
            results.append({
                'Configuration': config_name,
                'Final Accuracy': 0,
                'Average Reward': 0,
                'Training Time (min)': 0,
                'Correct Detections': 0,
                'False Positives': 0,
                'Missed Detections': 0,
                'Episodes': 0
            })
    
    # Generate results
    results_df = pd.DataFrame(results)
    
    print(f"\n🏆 SIMPLE DQN COMPARISON RESULTS:")
    print("="*60)
    print(results_df.to_string(index=False, float_format='%.1f'))
    
    # Find best configuration
    if len(results_df) > 0 and results_df['Final Accuracy'].max() > 0:
        best_config = results_df.loc[results_df['Final Accuracy'].idxmax()]
        print(f"\n🥇 BEST CONFIGURATION: {best_config['Configuration']}")
        print(f"   🎯 Accuracy: {best_config['Final Accuracy']:.1f}%")
        print(f"   📊 Avg Reward: {best_config['Average Reward']:+.1f}")
        print(f"   ⏱️ Training Time: {best_config['Training Time (min)']:.1f} min")
    
    # Save results
    results_dir = Path("results/simple_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(results_dir / "simple_dqn_results.csv", index=False)
    
    with open(results_dir / "simple_dqn_summary.json", 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_configurations': len(results),
            'successful_configs': len([r for r in results if r['Final Accuracy'] > 0]),
            'best_accuracy': float(results_df['Final Accuracy'].max()) if len(results_df) > 0 else 0,
            'results': results
        }, f, indent=2)
    
    print(f"\n📁 Results saved to: {results_dir}")
    return results_df

if __name__ == "__main__":
    results = train_simple_dqn_comparison()
    print(f"\n🎉 SIMPLE DQN COMPARISON COMPLETE!")
