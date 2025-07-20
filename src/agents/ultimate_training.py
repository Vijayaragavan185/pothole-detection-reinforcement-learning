#!/usr/bin/env python3
"""
🔥 ULTIMATE DQN TRAINING - CRUSH ALL BENCHMARKS! 🔥
The most advanced RL training pipeline ever created for pothole detection!
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.environment.pothole_env import VideoBasedPotholeEnv
from src.agents.advanced_dqn import AdvancedDQNAgent
from src.agents.dqn_agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import json
import time

class UltimateTrainer:
    """🏆 ULTIMATE TRAINING SYSTEM - MAXIMUM PERFORMANCE! 🏆"""
    
    def __init__(self):
        self.results = {}
        self.comparison_data = []
        
    def train_agent_configuration(self, config_name, agent_config, episodes=200):
        """Train a specific agent configuration"""
        print(f"\n🚀 TRAINING {config_name.upper()} CONFIGURATION!")
        print("="*60)
        
        # Create environment and agent
        env = VideoBasedPotholeEnv(split='train', max_memory_mb=2048)
        agent = AdvancedDQNAgent(**agent_config)
        
        # Training loop with detailed tracking
        training_results = []
        evaluation_results = []
        
        start_time = time.time()
        
        for episode in range(1, episodes + 1):
            # Train episode
            episode_result = agent.train_episode(env, max_steps=1)
            training_results.append(episode_result)
            
            # Progress logging
            if episode % 20 == 0:
                recent_rewards = [r['total_reward'] for r in training_results[-20:]]
                avg_reward = np.mean(recent_rewards)
                
                print(f"Episode {episode:3d} | "
                      f"Reward: {episode_result['total_reward']:+4} | "
                      f"Avg(20): {avg_reward:+5.1f} | "
                      f"ε: {agent.epsilon:.3f} | "
                      f"LR: {agent.scheduler.get_last_lr()[0]:.6f}")
            
            # Periodic evaluation
            if episode % 50 == 0:
                eval_result = agent.evaluate(env, num_episodes=30)
                eval_result['episode'] = episode
                evaluation_results.append(eval_result)
                
                print(f"   🎯 EVAL: Accuracy={eval_result['accuracy']:.1f}%, "
                      f"Avg Reward={eval_result['average_reward']:+5.1f}")
        
        training_time = time.time() - start_time
        
        # Final comprehensive evaluation
        final_eval = agent.evaluate(env, num_episodes=100)
        
        # Store results
        self.results[config_name] = {
            'agent': agent,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'final_evaluation': final_eval,
            'training_time': training_time,
            'config': agent_config
        }
        
        # Add to comparison data
        self.comparison_data.append({
            'Configuration': config_name,
            'Final Accuracy': final_eval['accuracy'],
            'Average Reward': final_eval['average_reward'],
            'Training Time (min)': training_time / 60,
            'Parameters': sum(p.numel() for p in agent.q_network.parameters()),
            'Correct Detections': final_eval['correct_decisions'],
            'False Positives': final_eval['false_positives'],
            'Missed Detections': final_eval['missed_detections']
        })
        
        env.close()
        
        print(f"✅ {config_name} TRAINING COMPLETE!")
        print(f"   🎯 Final Accuracy: {final_eval['accuracy']:.1f}%")
        print(f"   📊 Average Reward: {final_eval['average_reward']:+5.1f}")
        print(f"   ⏱️ Training Time: {training_time/60:.1f} minutes")
        
        return agent, final_eval
    
    def run_ultimate_comparison(self):
        """🔥 RUN THE ULTIMATE DQN COMPARISON! 🔥"""
        
        print("🔥" * 30)
        print("ULTIMATE DQN PERFORMANCE COMPARISON")
        print("🔥" * 30)
        
        # Configuration matrix
        configurations = {
            'STANDARD_DQN': {
                'input_shape': (5, 224, 224, 3),
                'num_actions': 5,
                'learning_rate': 0.0005,
                'use_double_dqn': False,
                'use_dueling': False,
                'use_prioritized_replay': False,
                'memory_size': 5000
            },
            'DOUBLE_DQN': {
                'input_shape': (5, 224, 224, 3),
                'num_actions': 5,
                'learning_rate': 0.0005,
                'use_double_dqn': True,
                'use_dueling': False,
                'use_prioritized_replay': False,
                'memory_size': 5000
            },
            'DUELING_DQN': {
                'input_shape': (5, 224, 224, 3),
                'num_actions': 5,
                'learning_rate': 0.0005,
                'use_double_dqn': False,
                'use_dueling': True,
                'use_prioritized_replay': False,
                'memory_size': 8000
            },
            'PRIORITIZED_DQN': {
                'input_shape': (5, 224, 224, 3),
                'num_actions': 5,
                'learning_rate': 0.0005,
                'use_double_dqn': True,
                'use_dueling': False,
                'use_prioritized_replay': True,
                'memory_size': 10000
            },
            'ULTIMATE_DQN': {
                'input_shape': (5, 224, 224, 3),
                'num_actions': 5,
                'learning_rate': 0.0003,
                'use_double_dqn': True,
                'use_dueling': True,
                'use_prioritized_replay': True,
                'memory_size': 15000,
                'epsilon_decay': 0.9995
            }
        }
        
        # Train each configuration
        for config_name, config_params in configurations.items():
            self.train_agent_configuration(config_name, config_params, episodes=300)
        
        # Generate comprehensive comparison
        self.generate_ultimate_comparison()
    
    def generate_ultimate_comparison(self):
        """Generate comprehensive comparison analysis"""
        
        print("\n🏆 GENERATING ULTIMATE PERFORMANCE ANALYSIS!")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.comparison_data)
        
        # Print comparison table
        print("\n📊 ULTIMATE PERFORMANCE COMPARISON:")
        print("="*100)
        print(comparison_df.to_string(index=False, float_format='%.2f'))
        
        # Create comprehensive visualization
        self.create_ultimate_visualization(comparison_df)
        
        # Identify best performer
        best_config = comparison_df.loc[comparison_df['Final Accuracy'].idxmax()]
        
        print(f"\n🏆 CHAMPION CONFIGURATION: {best_config['Configuration']}")
        print(f"   🎯 Accuracy: {best_config['Final Accuracy']:.1f}%")
        print(f"   📊 Avg Reward: {best_config['Average Reward']:+5.1f}")
        print(f"   ⚡ Parameters: {best_config['Parameters']:,}")
        
        # Save results
        results_path = Path("results/ultimate_comparison")
        results_path.mkdir(parents=True, exist_ok=True)
        
        comparison_df.to_csv(results_path / "ultimate_comparison.csv", index=False)
        
        # Save detailed results
        detailed_results = {
            'timestamp': datetime.now().isoformat(),
            'comparison_summary': comparison_df.to_dict('records'),
            'best_configuration': {
                'name': best_config['Configuration'],
                'accuracy': float(best_config['Final Accuracy']),
                'average_reward': float(best_config['Average Reward']),
                'parameters': int(best_config['Parameters'])
            },
            'detailed_results': {}
        }
        
        # Add detailed training histories
        for config_name, result in self.results.items():
            detailed_results['detailed_results'][config_name] = {
                'final_evaluation': result['final_evaluation'],
                'training_time': result['training_time'],
                'reward_history': [r['total_reward'] for r in result['training_results']],
                'loss_history': result['agent'].loss_history,
                'epsilon_history': result['agent'].epsilon_history
            }
        
        with open(results_path / "ultimate_detailed_results.json", 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"\n📁 Results saved to: {results_path}")
        return comparison_df, best_config
    
    def create_ultimate_visualization(self, comparison_df):
        """Create comprehensive visualization"""
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('🔥 ULTIMATE DQN PERFORMANCE COMPARISON 🔥', fontsize=18, fontweight='bold')
        
        # Accuracy comparison
        sns.barplot(data=comparison_df, x='Configuration', y='Final Accuracy', ax=axes[0,0], palette='viridis')
        axes[0,0].set_title('Final Accuracy Comparison', fontweight='bold')
        axes[0,0].set_ylabel('Accuracy (%)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Average reward comparison
        sns.barplot(data=comparison_df, x='Configuration', y='Average Reward', ax=axes[0,1], palette='plasma')
        axes[0,1].set_title('Average Reward Comparison', fontweight='bold')
        axes[0,1].set_ylabel('Average Reward')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Training time vs performance
        axes[0,2].scatter(comparison_df['Training Time (min)'], comparison_df['Final Accuracy'], 
                         s=100, alpha=0.7, c=comparison_df.index, cmap='tab10')
        axes[0,2].set_xlabel('Training Time (minutes)')
        axes[0,2].set_ylabel('Final Accuracy (%)')
        axes[0,2].set_title('Training Efficiency', fontweight='bold')
        for i, config in enumerate(comparison_df['Configuration']):
            axes[0,2].annotate(config, (comparison_df['Training Time (min)'].iloc[i], 
                                      comparison_df['Final Accuracy'].iloc[i]),
                             xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Error analysis
        error_data = comparison_df[['Configuration', 'False Positives', 'Missed Detections']].melt(
            id_vars=['Configuration'], var_name='Error Type', value_name='Count')
        sns.barplot(data=error_data, x='Configuration', y='Count', hue='Error Type', ax=axes[1,0])
        axes[1,0].set_title('Error Analysis', fontweight='bold')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend()
        
        # Parameter efficiency
        axes[1,1].scatter(comparison_df['Parameters'] / 1e6, comparison_df['Final Accuracy'],
                         s=100, alpha=0.7, c=comparison_df.index, cmap='tab10')
        axes[1,1].set_xlabel('Parameters (Millions)')
        axes[1,1].set_ylabel('Final Accuracy (%)')
        axes[1,1].set_title('Parameter Efficiency', fontweight='bold')
        for i, config in enumerate(comparison_df['Configuration']):
            axes[1,1].annotate(config, (comparison_df['Parameters'].iloc[i] / 1e6, 
                                      comparison_df['Final Accuracy'].iloc[i]),
                             xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Learning curves comparison
        for config_name, result in self.results.items():
            reward_history = [r['total_reward'] for r in result['training_results']]
            if len(reward_history) > 20:
                window = min(30, len(reward_history)//10)
                moving_avg = np.convolve(reward_history, np.ones(window)/window, mode='valid')
                axes[1,2].plot(range(window-1, len(reward_history)), moving_avg, 
                              label=config_name, linewidth=2, alpha=0.8)
        
        axes[1,2].set_title('Learning Curves Comparison', fontweight='bold')
        axes[1,2].set_xlabel('Episode')
        axes[1,2].set_ylabel('Moving Average Reward')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path("results/ultimate_comparison/ultimate_comparison_plots.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        print(f"📊 Ultimate comparison plots saved to: {plot_path}")
        plt.show()


def run_ultimate_training():
    """🚀 LAUNCH THE ULTIMATE TRAINING COMPARISON! 🚀"""
    
    print("🔥" * 50)
    print("LAUNCHING ULTIMATE DQN PERFORMANCE COMPARISON")
    print("PREPARE TO WITNESS THE MOST ADVANCED RL TRAINING EVER!")
    print("🔥" * 50)
    
    trainer = UltimateTrainer()
    comparison_results = trainer.run_ultimate_comparison()
    
    print(f"\n🎉 ULTIMATE TRAINING COMPLETE!")
    print(f"🏆 The most advanced pothole detection RL system ever created!")
    
    return trainer, comparison_results

if __name__ == "__main__":
    ultimate_trainer, results = run_ultimate_training()
