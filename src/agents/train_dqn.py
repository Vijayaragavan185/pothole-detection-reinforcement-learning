#!/usr/bin/env python3
"""
🚀 DQN TRAINING PIPELINE FOR POTHOLE DETECTION! 🚀
Revolutionary RL training that learns optimal detection strategies!
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.environment.pothole_env import VideoBasedPotholeEnv
from src.agents.dqn_agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

def train_dqn_agent(episodes=500, save_interval=50, eval_interval=25):
    """🔥 TRAIN THE REVOLUTIONARY DQN AGENT! 🔥"""
    
    print("🚀 STARTING REVOLUTIONARY DQN TRAINING!")
    print("="*60)
    
    # Create environment and agent
    print("🎯 Initializing Environment and Agent...")
    env = VideoBasedPotholeEnv(
        split='train', 
        max_memory_mb=8192,  # Increased memory allowance
        target_sequences=5000,  # More sequences
        lazy=False,  # Full loading
        verbose=True
    )
    agent = DQNAgent(
        input_shape=(5, 224, 224, 3),
        num_actions=5,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.998,
        memory_size=10000,
        batch_size=32,
        target_update=100
    )
    
    # Training tracking
    training_results = []
    best_reward = float('-inf')
    
    print(f"✅ Training Setup Complete!")
    print(f"   🎮 Environment: {len(env.episode_sequences)} sequences loaded")
    print(f"   🧠 Agent: {sum(p.numel() for p in agent.q_network.parameters()):,} parameters")
    print(f"   🎯 Training Episodes: {episodes}")
    print(f"   💾 Memory Capacity: {len(agent.memory.buffer)}")
    
    # Training loop
    print(f"\n🔥 BEGINNING TRAINING LOOP...")
    for episode in range(1, episodes + 1):
        # Train one episode
        episode_result = agent.train_episode(env, max_steps=500)
        training_results.append(episode_result)
        
        # Print progress
        if episode % 10 == 0:
            recent_rewards = [r['total_reward'] for r in training_results[-10:]]
            avg_reward = np.mean(recent_rewards)
            
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_result['total_reward']:+4} | "
                  f"Avg(10): {avg_reward:+5.1f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Loss: {episode_result['average_loss']:.4f} | "
                  f"Action: {episode_result['action_taken']} | "
                  f"Memory: {len(agent.memory):4d}")
        
        # Evaluate agent performance
        if episode % eval_interval == 0:
            print(f"\n🧪 EVALUATION at Episode {episode}...")
            eval_results = agent.evaluate(env, num_episodes=20)
            
            print(f"   📊 Average Reward: {eval_results['average_reward']:+5.1f}")
            print(f"   🎯 Accuracy: {eval_results['accuracy']:.1f}%")
            print(f"   ✅ Correct: {eval_results['correct_decisions']}")
            print(f"   ❌ False Positives: {eval_results['false_positives']}")
            print(f"   💀 Missed Detections: {eval_results['missed_detections']}")
            
            # Save best model
            if eval_results['average_reward'] > best_reward:
                best_reward = eval_results['average_reward']
                model_path = Path("results/models/best_dqn_agent.pth")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                agent.save_model(model_path)
                print(f"   🏆 NEW BEST MODEL! Saved to {model_path}")
        
        # Save checkpoint
        if episode % save_interval == 0:
            checkpoint_path = Path(f"results/models/dqn_checkpoint_ep{episode}.pth")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            agent.save_model(checkpoint_path)
            print(f"   💾 Checkpoint saved: {checkpoint_path}")
    
    # Final evaluation
    print(f"\n🏁 FINAL EVALUATION...")
    final_eval = agent.evaluate(env, num_episodes=50)
    
    print(f"🎉 TRAINING COMPLETED!")
    print(f"   📊 Final Average Reward: {final_eval['average_reward']:+5.1f}")
    print(f"   🎯 Final Accuracy: {final_eval['accuracy']:.1f}%")
    print(f"   🧠 Total Training Steps: {agent.training_step:,}")
    print(f"   📈 Episodes Completed: {agent.episode_count:,}")
    
    # Save final model and results
    final_model_path = Path("results/models/final_dqn_agent.pth")
    agent.save_model(final_model_path)
    
    # Generate training plots
    plots_path = Path("results/plots/dqn_training_progress.png")
    plots_path.parent.mkdir(parents=True, exist_ok=True)
    agent.plot_training_progress(save_path=plots_path)
    
    # Save training results
    results_path = Path("results/training_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    training_summary = {
        "timestamp": datetime.now().isoformat(),
        "episodes": episodes,
        "final_evaluation": final_eval,
        "best_reward": best_reward,
        "total_training_steps": agent.training_step,
        "hyperparameters": {
            "learning_rate": agent.learning_rate,
            "gamma": agent.gamma,
            "epsilon_start": agent.epsilon_start,
            "epsilon_end": agent.epsilon_end,
            "epsilon_decay": agent.epsilon_decay,
            "batch_size": agent.batch_size,
            "target_update": agent.target_update
        },
        "environment_info": {
            "sequences_loaded": len(env.episode_sequences),
            "split": env.split,
            "action_thresholds": env.action_thresholds
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    print(f"\n📁 Results saved:")
    print(f"   🎯 Final Model: {final_model_path}")
    print(f"   📊 Training Plots: {plots_path}")
    print(f"   📋 Results Summary: {results_path}")
    
    env.close()
    return agent, training_results, final_eval

if __name__ == "__main__":
    # Start the revolutionary training!
    trained_agent, training_history, final_results = train_dqn_agent(
        episodes=300,  # Start with 300 episodes
        save_interval=50,
        eval_interval=25
    )
    
    print(f"\n🎉 DQN AGENT TRAINING BREAKTHROUGH COMPLETE!")
    print(f"🚀 Ready for real-world pothole detection optimization!")
