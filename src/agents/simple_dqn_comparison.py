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
            'learning_rate': 0.001,
            'gamma': 0.95,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'memory_size': 3000,
            'batch_size': 16,
            'target_update': 100
        },
        'OPTIMIZED_DQN': {
            'learning_rate': 0.0005,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.998,
            'memory_size': 5000,
            'batch_size': 32,
            'target_update': 75
        },
        'FAST_DQN': {
            'learning_rate': 0.002,
            'gamma': 0.95,
            'epsilon_start': 0.9,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.99,
            'memory_size': 2000,
            'batch_size': 24,
            'target_update': 50
        }
    }
    
    print(f"🎯 Training {len(configurations)} simplified DQN configurations:")
    for i, config_name in enumerate(configurations.keys(), 1):
        print(f"   {i}. {config_name}")
    
    # Train each configuration
    for config_name, config_params in configurations.items():
        print(f"\n🚀 TRAINING {config_name}!")
        print("-" * 40)
        
        try:
            # Create environment with reduced memory requirements
            env = VideoBasedPotholeEnv(split='train', max_memory_mb=512)
            
            # Use proven DQN agent from Day 5
            agent = DQNAgent(
                input_shape=(5, 224, 224, 3),
                num_actions=5,
                **config_params
            )
            
            print(f"   🎮 Environment: {len(env.episode_sequences)} sequences")
            print(f"   🧠 Agent: {sum(p.numel() for p in agent.q_network.parameters()):,} parameters")
            
            # Training loop - reduced episodes for quick results
            training_results = []
            best_accuracy = 0
            
            start_time = time.time()
            
            for episode in range(1, 101):  # Only 100 episodes for quick test
                # Train episode
                episode_result = agent.train_episode(env, max_steps=1)
                training_results.append(episode_result)
                
                # Progress logging
                if episode % 20 == 0:
                    recent_rewards = [r['total_reward'] for r in training_results[-20:]]
                    avg_reward = np.mean(recent_rewards)
                    
                    print(f"Ep {episode:3d} | Reward: {episode_result['total_reward']:+3} | "
                          f"Avg: {avg_reward:+5.1f} | ε: {agent.epsilon:.3f}")
                
                # Quick evaluation
                if episode % 25 == 0:
                    eval_result = agent.evaluate(env, num_episodes=20)
                    if eval_result['accuracy'] > best_accuracy:
                        best_accuracy = eval_result['accuracy']
                    
                    print(f"   🎯 Eval: Accuracy={eval_result['accuracy']:.1f}%, "
                          f"Best: {best_accuracy:.1f}%")
            
            training_time = time.time() - start_time
            
            # Final evaluation
            final_eval = agent.evaluate(env, num_episodes=50)
            
            # Store results
            result = {
                'Configuration': config_name,
                'Final Accuracy': final_eval['accuracy'],
                'Best Accuracy': best_accuracy,
                'Average Reward': final_eval['average_reward'],
                'Training Time (min)': training_time / 60,
                'Parameters': sum(p.numel() for p in agent.q_network.parameters()),
                'Correct Detections': final_eval['correct_decisions'],
                'False Positives': final_eval['false_positives'],
                'Missed Detections': final_eval['missed_detections'],
                'Episodes': 100,
                'Learning Rate': config_params['learning_rate'],
                'Memory Size': config_params['memory_size']
            }
            
            results.append(result)
            
            print(f"✅ {config_name} COMPLETE!")
            print(f"   🎯 Final Accuracy: {final_eval['accuracy']:.1f}%")
            print(f"   📊 Average Reward: {final_eval['average_reward']:+5.1f}")
            print(f"   ⏱️ Training Time: {training_time/60:.1f} minutes")
            
            env.close()
            
        except Exception as e:
            print(f"❌ Error training {config_name}: {e}")
            # Add failure entry
            results.append({
                'Configuration': config_name,
                'Final Accuracy': 0,
                'Best Accuracy': 0,
                'Average Reward': 0,
                'Training Time (min)': 0,
                'Parameters': 0,
                'Correct Detections': 0,
                'False Positives': 0,
                'Missed Detections': 0,
                'Episodes': 0,
                'Learning Rate': config_params['learning_rate'],
                'Memory Size': config_params['memory_size']
            })
    
    # Generate comparison report
    generate_simple_comparison_report(results)
    
    return results

def generate_simple_comparison_report(results):
    """Generate simple comparison report"""
    
    print("\n" + "="*60)
    print("SIMPLE DQN COMPARISON RESULTS")
    print("="*60)
    
    if not results:
        print("❌ No results to display")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Display results table
    display_columns = [
        'Configuration', 'Final Accuracy', 'Best Accuracy', 
        'Average Reward', 'Training Time (min)'
    ]
    
    print("\n📊 PERFORMANCE COMPARISON:")
    print("-" * 80)
    print(df[display_columns].to_string(index=False, float_format='%.2f'))
    
    # Display detailed metrics
    detail_columns = [
        'Configuration', 'Correct Detections', 'False Positives', 
        'Missed Detections', 'Parameters', 'Learning Rate'
    ]
    
    print(f"\n📈 DETAILED METRICS:")
    print("-" * 80)
    print(df[detail_columns].to_string(index=False, float_format='%.4f'))
    
    # Find best performer
    if df['Final Accuracy'].max() > 0:
        best_config = df.loc[df['Final Accuracy'].idxmax()]
        
        print(f"\n🏆 BEST PERFORMER: {best_config['Configuration']}")
        print(f"   🎯 Accuracy: {best_config['Final Accuracy']:.1f}%")
        print(f"   📊 Avg Reward: {best_config['Average Reward']:+5.1f}")
        print(f"   ⏱️ Training Time: {best_config['Training Time (min)']:.1f} min")
        print(f"   🧠 Parameters: {best_config['Parameters']:,}")
    
    # Create simple visualization
    create_simple_plots(df)
    
    # Save results
    save_simple_results(df)

def create_simple_plots(df):
    """Create simple comparison plots"""
    
    if len(df) == 0 or df['Final Accuracy'].max() == 0:
        print("⚠️ No data available for visualization")
        return
    
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('🚀 Simple DQN Comparison Results', fontsize=16, fontweight='bold')
        
        # Accuracy comparison
        axes[0].bar(df['Configuration'], df['Final Accuracy'], color='skyblue', alpha=0.7)
        axes[0].set_title('Final Accuracy Comparison')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(df['Final Accuracy']):
            axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Average reward comparison
        axes[1].bar(df['Configuration'], df['Average Reward'], color='lightgreen', alpha=0.7)
        axes[1].set_title('Average Reward Comparison')
        axes[1].set_ylabel('Average Reward')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Training time comparison
        axes[2].bar(df['Configuration'], df['Training Time (min)'], color='lightcoral', alpha=0.7)
        axes[2].set_title('Training Time Comparison')
        axes[2].set_ylabel('Training Time (minutes)')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        results_dir = Path("results/simple_comparison")
        results_dir.mkdir(parents=True, exist_ok=True)
        plot_path = results_dir / "simple_dqn_comparison.png"
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Comparison plots saved to: {plot_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Could not create plots: {e}")

def save_simple_results(df):
    """Save simple results to files"""
    
    # Create results directory
    results_dir = Path("results/simple_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_path = results_dir / "simple_dqn_results.csv"
    df.to_csv(csv_path, index=False)
    
    # Save JSON summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'training_approach': 'Simplified DQN Comparison (Option B)',
        'total_configurations': len(df),
        'results': df.to_dict('records'),
        'summary_statistics': {
            'best_accuracy': float(df['Final Accuracy'].max()),
            'average_accuracy': float(df['Final Accuracy'].mean()),
            'total_training_time': float(df['Training Time (min)'].sum()),
            'average_training_time': float(df['Training Time (min)'].mean())
        }
    }
    
    json_path = results_dir / "simple_dqn_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save text report
    report_text = f"""
🔥 SIMPLE DQN COMPARISON REPORT 🔥
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW:
- Training Approach: Simplified 3-configuration comparison
- Total Configurations: {len(df)}
- Training Episodes per Config: 100
- Total Training Time: {df['Training Time (min)'].sum():.1f} minutes

RESULTS SUMMARY:
"""
    
    for _, row in df.iterrows():
        report_text += f"""
{row['Configuration']}:
  Final Accuracy: {row['Final Accuracy']:.1f}%
  Average Reward: {row['Average Reward']:+.1f}
  Training Time: {row['Training Time (min)']:.1f} min
  Parameters: {row['Parameters']:,}
"""
    
    if df['Final Accuracy'].max() > 0:
        best_config = df.loc[df['Final Accuracy'].idxmax()]
        report_text += f"""
BEST PERFORMER: {best_config['Configuration']}
- Achieved {best_config['Final Accuracy']:.1f}% accuracy
- Average reward: {best_config['Average Reward']:+.1f}
- Training efficiency: {best_config['Final Accuracy'] / best_config['Training Time (min)']:.1f} accuracy/minute
"""
    
    report_path = results_dir / "simple_dqn_report.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"💾 Results saved to:")
    print(f"   📊 CSV: {csv_path}")
    print(f"   📝 JSON: {json_path}")
    print(f"   📋 Report: {report_path}")

if __name__ == "__main__":
    print("🚀 STARTING SIMPLE DQN COMPARISON (OPTION B)")
    print("=" * 60)
    print("This approach uses:")
    print("- ✅ Proven DQN agent from Day 5")
    print("- ✅ Working environment with fallback data")
    print("- ✅ 3 optimized configurations")
    print("- ✅ Quick training (100 episodes each)")
    print("- ✅ Guaranteed results in ~15-30 minutes")
    print()
    
    results = train_simple_dqn_comparison()
    
    print(f"\n🎉 SIMPLE DQN COMPARISON COMPLETE!")
    print(f"🏆 Successfully trained {len([r for r in results if r['Final Accuracy'] > 0])} configurations")
    print(f"📁 Results saved to: results/simple_comparison/")
