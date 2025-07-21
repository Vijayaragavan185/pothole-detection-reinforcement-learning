#!/usr/bin/env python3
"""
🚀 ULTIMATE DQN COMPARISON TRAINING! 🚀
Complete training pipeline using your comprehensive advanced_dqn.py implementation
with balanced data for comprehensive evaluation and comparison
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.environment.pothole_env import VideoBasedPotholeEnv
from src.agents.advanced_dqn import AdvancedDQNAgent, DuelingDQN, PotholeDetectionDQN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import json
import time
import warnings
from configs.config import DQN_CONFIGS, ENV_CONFIG, VIDEO_CONFIG

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

class UltimateTrainingPipeline:
    """
    🏆 ULTIMATE DQN COMPARISON TRAINER 🏆
    
    Comprehensive comparison of all DQN variants using your advanced implementation:
    - Standard DQN
    - Dueling DQN  
    - Ultimate DQN (Double + Dueling + Prioritized Replay)
    """
    
    def __init__(self, episodes_per_agent=300, balanced_training=True, target_sequences=5000):
        self.episodes_per_agent = episodes_per_agent
        self.balanced_training = balanced_training
        self.target_sequences = target_sequences
        
        # Results storage
        self.results = {}
        self.comparison_data = []
        self.training_histories = {}
        
        # Setup directories
        self.results_dir = Path("results/ultimate_comparison")
        self.models_dir = Path("results/ultimate_comparison/models")
        self.plots_dir = Path("results/ultimate_comparison/plots")
        
        for directory in [self.results_dir, self.models_dir, self.plots_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.start_time = datetime.now()
        
        print("🚀 ULTIMATE DQN COMPARISON TRAINING INITIALIZED!")
        print("="*70)
        print(f"   🎯 Episodes per agent: {episodes_per_agent}")
        print(f"   ⚖️ Balanced training: {balanced_training}")
        print(f"   📊 Target sequences: {target_sequences}")
        print(f"   📁 Results directory: {self.results_dir}")
        print(f"   ⏰ Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def create_environment(self):
        """Create training environment with your enhanced features"""
        return VideoBasedPotholeEnv(
            split='train',
            max_memory_mb=8192,
            target_sequences=self.target_sequences,
            balanced=self.balanced_training,
            verbose=True
        )
    
    def get_agent_configurations(self):
        """Define all DQN configurations to compare"""
        base_config = {
            'input_shape': (5, 224, 224, 3),
            'num_actions': 5,
            'learning_rate': 0.0005,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.9995,
            'memory_size': 10000,
            'batch_size': 32,
            'target_update': 100
        }
        
        configurations = {
            'Standard_DQN': {
                **base_config,
                'use_double_dqn': False,
                'use_dueling': False,
                'use_prioritized_replay': False,
                'description': 'Standard DQN with temporal CNN+LSTM'
            },
            'Dueling_DQN': {
                **base_config,
                'use_double_dqn': False,
                'use_dueling': True,
                'use_prioritized_replay': False,
                'memory_size': 12000,
                'description': 'DQN with Dueling Architecture'
            },
            'Ultimate_DQN': {
                **base_config,
                'use_double_dqn': True,
                'use_dueling': True,
                'use_prioritized_replay': True,
                'learning_rate': 0.0003,  # Lower for stability with all features
                'memory_size': 15000,
                'batch_size': 32,
                'target_update': 75,
                'description': 'Ultimate: Double + Dueling + Prioritized Replay'
            }
        }
        
        return configurations
    
    def train_single_agent(self, agent_name, agent_config):
        """Train a single agent configuration with comprehensive tracking"""
        print(f"\n🚀 TRAINING {agent_name.upper()}")
        print("="*60)
        
        # Create environment
        env = self.create_environment()
        
        if len(env.episode_sequences) == 0:
            print(f"❌ Error: No sequences loaded for {agent_name}")
            env.close()
            return None
        
        # Create agent using your advanced implementation
        try:
            agent = AdvancedDQNAgent(**agent_config)
        except Exception as e:
            print(f"❌ Error creating agent {agent_name}: {e}")
            env.close()
            return None
        
        # Training tracking
        training_start = time.time()
        episode_rewards = []
        episode_accuracies = []
        evaluation_history = []
        best_performance = {
            'accuracy': 0,
            'f1_score': 0,
            'episode': 0
        }
        
        print(f"   🎮 Environment: {len(env.episode_sequences)} sequences")
        print(f"   🧠 Agent: {sum(p.numel() for p in agent.q_network.parameters()):,} parameters")
        print(f"   📊 Architecture: {agent_config['description']}")
        
        # Initial validation
        try:
            initial_eval = agent.evaluate(env, num_episodes=10)
            print(f"   🧪 Initial performance: {initial_eval['overall_accuracy']:.1f}% accuracy")
        except:
            print("   ⚠️ Initial evaluation failed, continuing...")
        
        # Training loop
        for episode in range(1, self.episodes_per_agent + 1):
            try:
                # Train episode using your implementation
                episode_result = agent.train_episode(env, max_steps=200)
                episode_rewards.append(episode_result['total_reward'])
                
                # Progress reporting
                if episode % 25 == 0:
                    recent_rewards = episode_rewards[-25:]
                    avg_reward = np.mean(recent_rewards)
                    
                    print(f"   Episode {episode:3d} | "
                          f"Reward: {episode_result['total_reward']:+4} | "
                          f"Avg(25): {avg_reward:+5.1f} | "
                          f"ε: {agent.epsilon:.3f} | "
                          f"Loss: {episode_result['average_loss']:.4f} | "
                          f"Memory: {len(agent.memory):,}")
                
                # Comprehensive evaluation every 50 episodes
                if episode % 50 == 0:
                    print(f"\n   🧪 COMPREHENSIVE EVALUATION at Episode {episode}...")
                    
                    eval_results = agent.evaluate(env, num_episodes=50)
                    evaluation_history.append({
                        'episode': episode,
                        **eval_results
                    })
                    
                    episode_accuracies.append(eval_results['overall_accuracy'])
                    
                    # Display comprehensive metrics
                    print(f"      📊 Overall Accuracy: {eval_results['overall_accuracy']:.1f}%")
                    print(f"      🎯 Precision: {eval_results['precision']:.1f}%")
                    print(f"      📈 Recall: {eval_results['recall']:.1f}%")
                    print(f"      ⚖️ F1-Score: {eval_results['f1_score']:.1f}")
                    print(f"      🕳️ Pothole Accuracy: {eval_results['pothole_accuracy']:.1f}% ({eval_results['pothole_episodes']} episodes)")
                    print(f"      🛣️ Non-pothole Accuracy: {eval_results['non_pothole_accuracy']:.1f}% ({eval_results['non_pothole_episodes']} episodes)")
                    
                    # Confusion matrix
                    if 'confusion_matrix' in eval_results:
                        cm = eval_results['confusion_matrix']
                        print(f"      📊 Confusion Matrix:")
                        print(f"         TP: {cm['true_positives']}, TN: {cm['true_negatives']}")
                        print(f"         FP: {cm['false_positives']}, FN: {cm['false_negatives']}")
                    
                    # Threshold analysis
                    if 'threshold_analysis' in eval_results:
                        print(f"      🎯 Top Performing Thresholds:")
                        sorted_thresholds = sorted(
                            eval_results['threshold_analysis'].items(),
                            key=lambda x: x[1]['success_rate'],
                            reverse=True
                        )[:3]
                        
                        for action, stats in sorted_thresholds:
                            print(f"         Action {action} (th={stats['threshold']:.1f}): "
                                  f"{stats['success_rate']:.1f}% success ({stats['usage_count']} uses)")
                    
                    # Save best model
                    current_f1 = eval_results['f1_score']
                    if current_f1 > best_performance['f1_score']:
                        best_performance = {
                            'accuracy': eval_results['overall_accuracy'],
                            'f1_score': current_f1,
                            'episode': episode,
                            'full_results': eval_results
                        }
                        
                        model_path = self.models_dir / f"best_{agent_name.lower()}_model.pth"
                        agent.save_model(model_path)
                        print(f"      🏆 NEW BEST! F1-Score: {current_f1:.2f}, Saved to {model_path}")
                
            except Exception as e:
                print(f"   ❌ Error in episode {episode}: {e}")
                continue
        
        training_time = time.time() - training_start
        
        # Final comprehensive evaluation
        print(f"\n   🏁 FINAL EVALUATION for {agent_name}...")
        try:
            final_eval = agent.evaluate(env, num_episodes=100)
            
            # Get advanced statistics from your implementation
            advanced_stats = agent.get_advanced_stats()
            
            # Store comprehensive results
            agent_results = {
                'agent_name': agent_name,
                'config': agent_config,
                'training_time_minutes': training_time / 60,
                'episode_rewards': episode_rewards,
                'episode_accuracies': episode_accuracies,
                'evaluation_history': evaluation_history,
                'final_evaluation': final_eval,
                'best_performance': best_performance,
                'advanced_stats': advanced_stats,
                'total_episodes': len(episode_rewards),
                'convergence_episode': self._find_convergence_episode(episode_accuracies)
            }
            
            self.results[agent_name] = agent_results
            self.training_histories[agent_name] = {
                'rewards': episode_rewards,
                'accuracies': episode_accuracies,
                'losses': agent.loss_history,
                'epsilon': agent.epsilon_history,
                'td_errors': agent.td_error_history if hasattr(agent, 'td_error_history') else []
            }
            
            # Add to comparison data
            self.comparison_data.append({
                'Agent': agent_name.replace('_', ' '),
                'Architecture': agent_config['description'],
                'Final Accuracy (%)': final_eval['overall_accuracy'],
                'Best Accuracy (%)': best_performance['accuracy'],
                'Final F1-Score': final_eval['f1_score'],
                'Best F1-Score': best_performance['f1_score'],
                'Precision (%)': final_eval['precision'],
                'Recall (%)': final_eval['recall'],
                'Pothole Accuracy (%)': final_eval['pothole_accuracy'],
                'Non-pothole Accuracy (%)': final_eval['non_pothole_accuracy'],
                'Training Time (min)': training_time / 60,
                'Parameters': sum(p.numel() for p in agent.q_network.parameters()),
                'Convergence Episode': agent_results['convergence_episode'],
                'Memory Size': len(agent.memory),
                'Final Epsilon': agent.epsilon,
                'Double DQN': agent_config['use_double_dqn'],
                'Dueling': agent_config['use_dueling'],
                'Prioritized Replay': agent_config['use_prioritized_replay']
            })
            
            print(f"✅ {agent_name} TRAINING COMPLETED!")
            print(f"   🎯 Final Accuracy: {final_eval['overall_accuracy']:.1f}%")
            print(f"   🏆 Best Accuracy: {best_performance['accuracy']:.1f}% (Episode {best_performance['episode']})")
            print(f"   📊 Final F1-Score: {final_eval['f1_score']:.2f}")
            print(f"   ⏱️ Training Time: {training_time/60:.1f} minutes")
            
        except Exception as e:
            print(f"   ❌ Final evaluation failed: {e}")
            agent_results = None
        
        env.close()
        return agent_results
    
    def _find_convergence_episode(self, accuracies, window=10, threshold=2.0):
        """Find episode where accuracy converged (std dev < threshold for window)"""
        if len(accuracies) < window:
            return len(accuracies)
        
        for i in range(window, len(accuracies)):
            window_std = np.std(accuracies[i-window:i])
            if window_std < threshold:
                return i - window + 1
        
        return len(accuracies)
    
    def run_ultimate_comparison(self):
        """Run complete comparison of all DQN variants"""
        print("🔥" * 80)
        print("ULTIMATE DQN COMPARISON - COMPREHENSIVE EVALUATION")
        print(f"Training {self.episodes_per_agent} episodes per agent with balanced data")
        print("🔥" * 80)
        
        configurations = self.get_agent_configurations()
        
        print(f"\n🎯 Configurations to train:")
        for i, (name, config) in enumerate(configurations.items(), 1):
            print(f"   {i}. {name}: {config['description']}")
        
        # Train all configurations
        for agent_name, agent_config in configurations.items():
            try:
                self.train_single_agent(agent_name, agent_config)
            except Exception as e:
                print(f"❌ Failed to train {agent_name}: {e}")
                continue
        
        # Generate comprehensive analysis
        return self.generate_ultimate_analysis()
    
    def generate_ultimate_analysis(self):
        """Generate comprehensive performance analysis"""
        if not self.results:
            print("❌ No results to analyze!")
            return None
        
        print("\n📊 GENERATING ULTIMATE PERFORMANCE ANALYSIS...")
        print("="*70)
        
        # Create comparison DataFrame
        df_comparison = pd.DataFrame(self.comparison_data)
        
        # Display comparison table
        print("\n🏆 ULTIMATE DQN COMPARISON RESULTS")
        print("="*120)
        
        display_columns = [
            'Agent', 'Final Accuracy (%)', 'Best Accuracy (%)', 
            'Final F1-Score', 'Precision (%)', 'Recall (%)',
            'Training Time (min)', 'Convergence Episode'
        ]
        
        if not df_comparison.empty:
            print(df_comparison[display_columns].to_string(index=False, float_format='%.2f'))
        
        # Performance ranking
        print(f"\n🥇 PERFORMANCE RANKING:")
        if not df_comparison.empty:
            ranked = df_comparison.sort_values('Final F1-Score', ascending=False)
            for i, (_, row) in enumerate(ranked.iterrows(), 1):
                print(f"   {i}. {row['Agent']}: {row['Final F1-Score']:.3f} F1-Score, "
                      f"{row['Final Accuracy (%)']:.1f}% Accuracy")
        
        # Architecture analysis
        print(f"\n🏗️ ARCHITECTURE ANALYSIS:")
        if not df_comparison.empty:
            for _, row in df_comparison.iterrows():
                features = []
                if row['Double DQN']: features.append("Double DQN")
                if row['Dueling']: features.append("Dueling")
                if row['Prioritized Replay']: features.append("Prioritized Replay")
                
                feature_str = " + ".join(features) if features else "Standard"
                print(f"   {row['Agent']}: {feature_str}")
                print(f"      Performance: {row['Final Accuracy (%)']:.1f}% accuracy, "
                      f"{row['Final F1-Score']:.3f} F1-score")
                print(f"      Efficiency: {row['Convergence Episode']} episodes to converge, "
                      f"{row['Training Time (min)']:.1f}min training")
        
        # Generate visualizations
        self.create_ultimate_visualizations(df_comparison)
        
        # Save detailed results
        self.save_ultimate_results(df_comparison)
        
        return df_comparison
    
    def create_ultimate_visualizations(self, df_comparison):
        """Create comprehensive visualizations using your advanced plotting"""
        if df_comparison.empty:
            return
        
        print("\n📈 Generating comprehensive visualizations...")
        
        # 1. Performance Comparison Plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Ultimate DQN Performance Comparison', fontsize=16, fontweight='bold')
        
        agents = df_comparison['Agent']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#8E44AD', '#E74C3C'][:len(agents)]
        
        # Accuracy comparison
        x = np.arange(len(agents))
        width = 0.35
        
        ax1.bar(x - width/2, df_comparison['Final Accuracy (%)'], width, 
                label='Final Accuracy', color=colors[0], alpha=0.8)
        ax1.bar(x + width/2, df_comparison['Best Accuracy (%)'], width, 
                label='Best Accuracy', color=colors[1], alpha=0.8)
        
        ax1.set_xlabel('DQN Variants')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(agents, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # F1-Score comparison
        ax2.bar(agents, df_comparison['Final F1-Score'], color=colors[2], alpha=0.8)
        ax2.set_xlabel('DQN Variants')
        ax2.set_ylabel('F1-Score')
        ax2.set_title('F1-Score Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Training efficiency
        ax3.scatter(df_comparison['Training Time (min)'], df_comparison['Final Accuracy (%)'], 
                   s=df_comparison['Parameters']/10000, alpha=0.7, c=colors[:len(agents)])
        ax3.set_xlabel('Training Time (minutes)')
        ax3.set_ylabel('Final Accuracy (%)')
        ax3.set_title('Training Efficiency (bubble size = parameters)')
        ax3.grid(True, alpha=0.3)
        
        for i, agent in enumerate(agents):
            ax3.annotate(agent, 
                        (df_comparison.iloc[i]['Training Time (min)'], 
                         df_comparison.iloc[i]['Final Accuracy (%)']),
                        xytext=(5, 5), textcoords='offset points')
        
        # Convergence analysis
        ax4.bar(agents, df_comparison['Convergence Episode'], color=colors[3], alpha=0.8)
        ax4.set_xlabel('DQN Variants')
        ax4.set_ylabel('Episodes to Converge')
        ax4.set_title('Convergence Speed')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plots_dir / "ultimate_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   📊 Comparison plot saved: {plot_path}")
        plt.show()
        
        # 2. Training Progress Plots for each agent
        for agent_name, history in self.training_histories.items():
            if agent_name in self.results:
                agent = self.results[agent_name]
                
                # Use your advanced plotting if agent object is available
                try:
                    plot_path = self.plots_dir / f"{agent_name.lower()}_training_progress.png"
                    
                    # Create custom plot since we have the data
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                    fig.suptitle(f'{agent_name} Training Progress', fontsize=14, fontweight='bold')
                    
                    # Rewards
                    episodes = range(1, len(history['rewards']) + 1)
                    ax1.plot(episodes, history['rewards'], alpha=0.7)
                    if len(history['rewards']) > 50:
                        window = 50
                        moving_avg = np.convolve(history['rewards'], np.ones(window)/window, mode='valid')
                        ax1.plot(range(window, len(history['rewards']) + 1), moving_avg, 'r-', linewidth=2)
                    ax1.set_title('Episode Rewards')
                    ax1.set_xlabel('Episode')
                    ax1.set_ylabel('Reward')
                    ax1.grid(True)
                    
                    # Accuracy over time
                    if history['accuracies']:
                        eval_episodes = range(50, len(history['accuracies']) * 50 + 1, 50)
                        ax2.plot(eval_episodes, history['accuracies'], 'g-o', linewidth=2, markersize=4)
                        ax2.set_title('Accuracy Evolution')
                        ax2.set_xlabel('Episode')
                        ax2.set_ylabel('Accuracy (%)')
                        ax2.grid(True)
                    
                    # Loss
                    if history['losses']:
                        ax3.plot(history['losses'], alpha=0.7)
                        ax3.set_title('Training Loss')
                        ax3.set_xlabel('Training Step')
                        ax3.set_ylabel('Loss')
                        ax3.grid(True)
                    
                    # Epsilon decay
                    if history['epsilon']:
                        ax4.plot(history['epsilon'])
                        ax4.set_title('Epsilon Decay')
                        ax4.set_xlabel('Training Step')
                        ax4.set_ylabel('Epsilon')
                        ax4.grid(True)
                    
                    plt.tight_layout()
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"   📈 {agent_name} training progress saved: {plot_path}")
                    
                except Exception as e:
                    print(f"   ⚠️ Could not create plot for {agent_name}: {e}")
    
    def save_ultimate_results(self, df_comparison):
        """Save comprehensive results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comparison CSV
        comparison_path = self.results_dir / f"ultimate_comparison_{timestamp}.csv"
        df_comparison.to_csv(comparison_path, index=False)
        
        # Save detailed JSON results
        detailed_results = {
            'metadata': {
                'timestamp': timestamp,
                'training_session_start': self.start_time.isoformat(),
                'total_duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                'episodes_per_agent': self.episodes_per_agent,
                'balanced_training': self.balanced_training,
                'target_sequences': self.target_sequences
            },
            'comparison_summary': df_comparison.to_dict('records'),
            'detailed_results': {}
        }
        
        # Add detailed results for each agent
        for agent_name, results in self.results.items():
            detailed_results['detailed_results'][agent_name] = {
                'config': results['config'],
                'final_evaluation': results['final_evaluation'],
                'best_performance': results['best_performance'],
                'advanced_stats': results['advanced_stats'],
                'training_summary': {
                    'total_episodes': results['total_episodes'],
                    'training_time_minutes': results['training_time_minutes'],
                    'convergence_episode': results['convergence_episode']
                }
            }
        
        json_path = self.results_dir / f"ultimate_detailed_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"\n💾 ULTIMATE RESULTS SAVED:")
        print(f"   📊 Comparison CSV: {comparison_path}")
        print(f"   📝 Detailed JSON: {json_path}")
        print(f"   📈 Plots directory: {self.plots_dir}")


def main():
    """Main training function"""
    print("🚀" * 50)
    print("LAUNCHING ULTIMATE DQN COMPARISON TRAINING")
    print("Using your comprehensive Advanced DQN implementation")
    print("🚀" * 50)
    
    # Create ultimate trainer
    trainer = UltimateTrainingPipeline(
        episodes_per_agent=300,
        balanced_training=True,
        target_sequences=5000
    )
    
    # Run comprehensive comparison
    results = trainer.run_ultimate_comparison()
    
    total_time = (datetime.now() - trainer.start_time).total_seconds() / 3600
    
    print(f"\n🎉 ULTIMATE DQN COMPARISON COMPLETED!")
    print(f"⏱️ Total training time: {total_time:.2f} hours")
    print(f"📁 All results saved to: {trainer.results_dir}")
    
    if results is not None and not results.empty:
        best_agent = results.loc[results['Final F1-Score'].idxmax()]
        print(f"\n🏆 CHAMPION: {best_agent['Agent']}")
        print(f"   🎯 Architecture: {best_agent['Architecture']}")
        print(f"   📊 Final Accuracy: {best_agent['Final Accuracy (%)']:.1f}%")
        print(f"   🏅 F1-Score: {best_agent['Final F1-Score']:.3f}")
    
    print(f"🚀 Ready for production deployment analysis!")
    
    return trainer, results


if __name__ == "__main__":
    ultimate_trainer, comparison_results = main()
