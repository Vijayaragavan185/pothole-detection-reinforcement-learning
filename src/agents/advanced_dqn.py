import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import json
import seaborn as sns
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import ENV_CONFIG, VIDEO_CONFIG
from src.agents.dqn_agent import PotholeDetectionDQN, ExperienceReplayBuffer

# Enhanced Experience for Prioritized Replay
PrioritizedExperience = namedtuple('PrioritizedExperience', 
    ['state', 'action', 'reward', 'next_state', 'done', 'priority'])

class DuelingDQN(nn.Module):
    """
    🏆 DUELING DQN ARCHITECTURE - ULTIMATE PERFORMANCE! 🏆
    
    Separates value and advantage functions for superior learning.
    Innovation: First Dueling DQN for video-based detection optimization!
    """
    
    def __init__(self, input_shape=(5, 224, 224, 3), num_actions=5):
        super(DuelingDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        print(f"🔥 Building DUELING DQN Architecture...")
        
        # Shared feature extraction (same as original DQN)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.frame_feature_size = 64 * 7 * 7
        self.temporal_fc = nn.Linear(self.frame_feature_size, 512)
        
        # Enhanced LSTM with more capacity
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=512,  # Increased capacity
            num_layers=3,     # Deeper network
            batch_first=True,
            dropout=0.2
        )
        
        # 🎯 DUELING ARCHITECTURE: Separate Value and Advantage streams
        
        # Value stream - estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single value output
        )
        
        # Advantage stream - estimates A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)  # Advantage for each action
        )
        
        self._initialize_weights()
        
        print(f"✅ DUELING DQN Complete!")
        print(f"   🧠 Total Parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"   🎯 Value Stream: {sum(p.numel() for p in self.value_stream.parameters()):,} params")
        print(f"   ⚡ Advantage Stream: {sum(p.numel() for p in self.advantage_stream.parameters()):,} params")
    
    def _initialize_weights(self):
        """Enhanced weight initialization for Dueling architecture"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass through Dueling DQN
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        """
        batch_size, seq_len, height, width, channels = x.shape
        
        # Shared feature extraction
        x = x.view(batch_size * seq_len, channels, height, width)
        conv_features = self.temporal_conv(x)
        conv_features = conv_features.view(batch_size * seq_len, -1)
        frame_features = F.relu(self.temporal_fc(conv_features))
        frame_features = frame_features.view(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(frame_features)
        final_features = lstm_out[:, -1, :]
        
        # Dueling streams
        value = self.value_stream(final_features)  # (batch_size, 1)
        advantage = self.advantage_stream(final_features)  # (batch_size, num_actions)
        
        # Combine value and advantage with mean subtraction
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)
        
        return q_values


class PrioritizedReplayBuffer:
    """
    🎮 PRIORITIZED EXPERIENCE REPLAY - INTELLIGENT LEARNING! 🎮
    
    Prioritizes important experiences for more efficient learning.
    """
    
    def __init__(self, capacity=10000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
        print(f"🧠 Prioritized Replay Buffer initialized:")
        print(f"   📊 Capacity: {capacity:,}")
        print(f"   ⚡ Alpha: {alpha}")
        print(f"   🎯 Beta start: {beta_start}")
    
    def beta(self):
        """Calculate current beta value (annealed)"""
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience with maximum priority"""
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample batch based on priorities"""
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        beta = self.beta()
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Convert to tensors
        states = torch.stack([torch.FloatTensor(s[0]) for s in samples])
        actions = torch.LongTensor([s[1] for s in samples])
        rewards = torch.FloatTensor([s[2] for s in samples])
        next_states = torch.stack([torch.FloatTensor(s[3]) for s in samples])
        dones = torch.BoolTensor([s[4] for s in samples])
        weights = torch.FloatTensor(weights)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
        
        self.frame += 1
    
    def __len__(self):
        return len(self.buffer)


class AdvancedDQNAgent:
    """
    🤖 ULTIMATE ADVANCED DQN AGENT - MAXIMUM PERFORMANCE! 🤖
    
    Combines Double DQN, Dueling DQN, and Prioritized Experience Replay
    for the most sophisticated pothole detection RL system ever created!
    """
    
    def __init__(self, 
                 input_shape=(5, 224, 224, 3), 
                 num_actions=5,
                 learning_rate=0.0003,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.9995,
                 memory_size=15000,
                 batch_size=32,
                 target_update=100,
                 use_double_dqn=True,
                 use_dueling=True,
                 use_prioritized_replay=True):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Initializing ADVANCED DQN Agent on device: {self.device}")
        
        # Architecture configuration
        self.use_double_dqn = use_double_dqn
        self.use_dueling = use_dueling
        self.use_prioritized_replay = use_prioritized_replay
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Networks - Choose architecture based on configuration
        if use_dueling:
            self.q_network = DuelingDQN(input_shape, num_actions).to(self.device)
            self.target_network = DuelingDQN(input_shape, num_actions).to(self.device)
            print("   🏆 Using DUELING DQN Architecture!")
        else:
            self.q_network = PotholeDetectionDQN(input_shape, num_actions).to(self.device)
            self.target_network = PotholeDetectionDQN(input_shape, num_actions).to(self.device)
            print("   🧠 Using Standard DQN Architecture!")
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer with advanced features
        self.optimizer = optim.AdamW(self.q_network.parameters(), 
                                   lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        
        # Experience replay
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(memory_size)
            print("   🎮 Using PRIORITIZED Experience Replay!")
        else:
            self.memory = ExperienceReplayBuffer(memory_size)
            print("   📝 Using Standard Experience Replay!")
        
        # Training tracking
        self.training_step = 0
        self.episode_count = 0
        self.loss_history = []
        self.reward_history = []
        self.epsilon_history = []
        self.lr_history = []
        self.td_error_history = []
        
        # Performance tracking
        self.action_distribution = np.zeros(num_actions)
        self.reward_distribution = {'correct': 0, 'false_positive': 0, 'missed': 0}
        
        config_str = f"Double DQN: {use_double_dqn}, Dueling: {use_dueling}, Prioritized: {use_prioritized_replay}"
        
        print(f"✅ ADVANCED DQN Agent Initialized!")
        print(f"   🔥 Configuration: {config_str}")
        print(f"   🧠 Network Parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
        print(f"   💾 Memory Capacity: {memory_size:,}")
        print(f"   🎯 Enhanced Features: Learning rate scheduling, Weight decay")
    
    def act(self, state, training=True):
        """Enhanced action selection with exploration tracking"""
        if training and random.random() < self.epsilon:
            # Exploration
            action = random.randrange(self.q_network.num_actions)
        else:
            # Exploitation
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
        
        # Track action distribution
        self.action_distribution[action] += 1
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
        
        # Track reward distribution
        if reward == 10:
            self.reward_distribution['correct'] += 1
        elif reward == -5:
            self.reward_distribution['false_positive'] += 1
        elif reward == -20:
            self.reward_distribution['missed'] += 1
    
    def replay(self):
        """Advanced training with Double DQN and Prioritized Replay"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        if self.use_prioritized_replay:
            states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
            weights = weights.to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values using Double DQN or standard DQN
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: Use main network to select actions, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(1)[0]
            
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Calculate TD errors
        td_errors = torch.abs(current_q_values.squeeze() - target_q_values).detach()
        
        # Weighted loss for prioritized replay
        loss = (weights * F.mse_loss(current_q_values.squeeze(), target_q_values, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Update priorities if using prioritized replay
        if self.use_prioritized_replay:
            new_priorities = td_errors.cpu().numpy() + 1e-6
            self.memory.update_priorities(indices, new_priorities)
        
        # Update tracking
        self.training_step += 1
        self.loss_history.append(loss.item())
        self.td_error_history.append(td_errors.mean().item())
        self.lr_history.append(self.scheduler.get_last_lr()[0])
        
        # Update target network
        if self.training_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"🎯 Target network updated at step {self.training_step}")
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
        
        return loss.item()
    
    def get_advanced_stats(self):
        """Get comprehensive agent statistics"""
        total_actions = self.action_distribution.sum()
        action_probs = self.action_distribution / max(total_actions, 1)
        
        total_rewards = sum(self.reward_distribution.values())
        reward_probs = {k: v/max(total_rewards, 1) for k, v in self.reward_distribution.items()}
        
        return {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'learning_rate': self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.learning_rate,
            'memory_size': len(self.memory),
            'action_distribution': action_probs.tolist(),
            'reward_distribution': reward_probs,
            'avg_td_error': np.mean(self.td_error_history[-100:]) if self.td_error_history else 0,
            'avg_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0
        }
    
    def plot_advanced_training_progress(self, save_path=None):
        """Enhanced training visualization"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Advanced DQN Training Progress - ULTIMATE PERFORMANCE ANALYSIS', fontsize=16, fontweight='bold')
        
        # Rewards
        if self.reward_history:
            axes[0,0].plot(self.reward_history, alpha=0.7, label='Episode Reward')
            if len(self.reward_history) > 20:
                window = min(50, len(self.reward_history)//10)
                moving_avg = np.convolve(self.reward_history, np.ones(window)/window, mode='valid')
                axes[0,0].plot(range(window-1, len(self.reward_history)), moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
            axes[0,0].set_title('Episode Rewards')
            axes[0,0].set_xlabel('Episode')
            axes[0,0].set_ylabel('Reward')
            axes[0,0].legend()
            axes[0,0].grid(True)
        
        # Loss
        if self.loss_history:
            axes[0,1].plot(self.loss_history, alpha=0.7)
            axes[0,1].set_title('Training Loss')
            axes[0,1].set_xlabel('Training Step')
            axes[0,1].set_ylabel('Loss')
            axes[0,1].grid(True)
        
        # Epsilon decay
        if self.epsilon_history:
            axes[0,2].plot(self.epsilon_history)
            axes[0,2].set_title('Epsilon Decay (Exploration)')
            axes[0,2].set_xlabel('Training Step')
            axes[0,2].set_ylabel('Epsilon')
            axes[0,2].grid(True)
        
        # Learning rate schedule
        if self.lr_history:
            axes[1,0].plot(self.lr_history)
            axes[1,0].set_title('Learning Rate Schedule')
            axes[1,0].set_xlabel('Training Step')
            axes[1,0].set_ylabel('Learning Rate')
            axes[1,0].grid(True)
        
        # TD Error
        if self.td_error_history:
            axes[1,1].plot(self.td_error_history)
            axes[1,1].set_title('TD Error Evolution')
            axes[1,1].set_xlabel('Training Step')
            axes[1,1].set_ylabel('Average TD Error')
            axes[1,1].grid(True)
        
        # Action distribution
        if self.action_distribution.sum() > 0:
            action_labels = [f'Threshold {i}\n({ENV_CONFIG["action_thresholds"][i]})' for i in range(len(self.action_distribution))]
            axes[1,2].bar(action_labels, self.action_distribution)
            axes[1,2].set_title('Action Distribution')
            axes[1,2].set_ylabel('Frequency')
            axes[1,2].tick_params(axis='x', rotation=45)
        
        # Reward distribution
        if sum(self.reward_distribution.values()) > 0:
            reward_labels = ['Correct\n(+10)', 'False Positive\n(-5)', 'Missed\n(-20)']
            reward_counts = [self.reward_distribution['correct'], 
                           self.reward_distribution['false_positive'], 
                           self.reward_distribution['missed']]
            colors = ['green', 'orange', 'red']
            axes[2,0].bar(reward_labels, reward_counts, color=colors, alpha=0.7)
            axes[2,0].set_title('Reward Distribution')
            axes[2,0].set_ylabel('Frequency')
        
        # Performance metrics over time
        if len(self.reward_history) > 10:
            window = min(25, len(self.reward_history)//5)
            accuracy_over_time = []
            for i in range(window, len(self.reward_history)+1):
                recent_rewards = self.reward_history[i-window:i]
                accuracy = sum(1 for r in recent_rewards if r == 10) / len(recent_rewards) * 100
                accuracy_over_time.append(accuracy)
            
            axes[2,1].plot(range(window, len(self.reward_history)+1), accuracy_over_time)
            axes[2,1].set_title(f'Accuracy Over Time (window={window})')
            axes[2,1].set_xlabel('Episode')
            axes[2,1].set_ylabel('Accuracy (%)')
            axes[2,1].grid(True)
        
        # Network architecture info
        axes[2,2].text(0.1, 0.9, 'ADVANCED DQN CONFIGURATION:', fontsize=12, fontweight='bold', transform=axes[2,2].transAxes)
        axes[2,2].text(0.1, 0.8, f'Double DQN: {self.use_double_dqn}', fontsize=10, transform=axes[2,2].transAxes)
        axes[2,2].text(0.1, 0.7, f'Dueling Architecture: {self.use_dueling}', fontsize=10, transform=axes[2,2].transAxes)
        axes[2,2].text(0.1, 0.6, f'Prioritized Replay: {self.use_prioritized_replay}', fontsize=10, transform=axes[2,2].transAxes)
        axes[2,2].text(0.1, 0.5, f'Parameters: {sum(p.numel() for p in self.q_network.parameters()):,}', fontsize=10, transform=axes[2,2].transAxes)
        axes[2,2].text(0.1, 0.4, f'Memory Size: {len(self.memory):,}', fontsize=10, transform=axes[2,2].transAxes)
        axes[2,2].text(0.1, 0.3, f'Training Steps: {self.training_step:,}', fontsize=10, transform=axes[2,2].transAxes)
        axes[2,2].text(0.1, 0.2, f'Episodes: {self.episode_count:,}', fontsize=10, transform=axes[2,2].transAxes)
        axes[2,2].set_xlim(0, 1)
        axes[2,2].set_ylim(0, 1)
        axes[2,2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 Advanced training plots saved to: {save_path}")
        
        plt.show()


# 🧪 TEST ADVANCED DQN
if __name__ == "__main__":
    print("🚀 TESTING ULTIMATE ADVANCED DQN AGENT!")
    print("="*70)
    
    # Test all configurations
    configurations = [
        {"use_double_dqn": True, "use_dueling": True, "use_prioritized_replay": True, "name": "ULTIMATE"},
        {"use_double_dqn": True, "use_dueling": False, "use_prioritized_replay": False, "name": "Double DQN"},
        {"use_double_dqn": False, "use_dueling": True, "use_prioritized_replay": False, "name": "Dueling DQN"},
    ]
    
    for config in configurations:
        print(f"\n🧪 Testing {config['name']} Configuration...")
        
        agent = AdvancedDQNAgent(
            input_shape=(5, 224, 224, 3),
            num_actions=5,
            learning_rate=0.0003,
            use_double_dqn=config["use_double_dqn"],
            use_dueling=config["use_dueling"],
            use_prioritized_replay=config["use_prioritized_replay"]
        )
        
        # Test forward pass
        dummy_state = np.random.rand(5, 224, 224, 3).astype(np.float32)
        action = agent.act(dummy_state, training=True)
        
        print(f"   ✅ {config['name']}: Action {action} selected")
        print(f"   📊 Parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    
    print(f"\n🎉 ALL ADVANCED DQN CONFIGURATIONS TESTED SUCCESSFULLY!")
    print(f"🚀 Ready to OBLITERATE all performance benchmarks!")
