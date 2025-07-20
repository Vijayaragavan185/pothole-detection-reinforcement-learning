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
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import ENV_CONFIG, VIDEO_CONFIG

# Define Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class PotholeDetectionDQN(nn.Module):
    """
    🧠 REVOLUTIONARY DQN NETWORK FOR POTHOLE DETECTION! 🧠
    
    This neural network learns optimal confidence threshold selection
    for pothole detection from video sequences.
    
    Innovation: First DQN to optimize detection thresholds using temporal video data!
    """
    
    def __init__(self, input_shape=(5, 224, 224, 3), num_actions=5):
        super(PotholeDetectionDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # Calculate input size for first linear layer
        # After conv layers: 5 frames -> feature maps
        self.conv_output_size = self._get_conv_output_size()
        
        print(f"🧠 Building DQN Network Architecture...")
        print(f"   📊 Input Shape: {input_shape}")
        print(f"   🎯 Output Actions: {num_actions}")
        
        # 🎬 TEMPORAL CNN FEATURE EXTRACTOR
        self.temporal_conv = nn.Sequential(
            # Process each frame through CNN layers
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),  # 224->54
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 54->27
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # 27->27
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))  # Ensure consistent output size
        )
        
        # 🔥 TEMPORAL PROCESSING LAYERS
        self.frame_feature_size = 64 * 7 * 7  # 3136
        self.temporal_fc = nn.Linear(self.frame_feature_size, 512)
        
        # 🧠 LSTM for Temporal Sequence Learning
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 🎯 Q-VALUE PREDICTION HEAD
        self.q_network = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_actions)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"✅ DQN Network Architecture Complete!")
        print(f"   🔥 Total Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _get_conv_output_size(self):
        """Calculate the output size after conv layers"""
        # This is a helper to understand the architecture
        return 64 * 7 * 7
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x):
        """
        Forward pass through the network
        Input: x shape = (batch_size, sequence_length, height, width, channels)
        Output: Q-values for each action
        """
        batch_size, seq_len, height, width, channels = x.shape
        
        # Reshape for CNN processing: (batch_size * seq_len, channels, height, width)
        x = x.view(batch_size * seq_len, channels, height, width)
        
        # Extract features from each frame
        conv_features = self.temporal_conv(x)  # (batch_size * seq_len, 64, 7, 7)
        conv_features = conv_features.view(batch_size * seq_len, -1)  # Flatten
        
        # Process through temporal FC
        frame_features = F.relu(self.temporal_fc(conv_features))  # (batch_size * seq_len, 512)
        
        # Reshape for LSTM: (batch_size, seq_len, feature_size)
        frame_features = frame_features.view(batch_size, seq_len, -1)
        
        # Process temporal sequence through LSTM
        lstm_out, _ = self.lstm(frame_features)  # (batch_size, seq_len, 256)
        
        # Use the last timestep for Q-value prediction
        final_features = lstm_out[:, -1, :]  # (batch_size, 256)
        
        # Predict Q-values for each action
        q_values = self.q_network(final_features)  # (batch_size, num_actions)
        
        return q_values


class ExperienceReplayBuffer:
    """
    🎮 EXPERIENCE REPLAY BUFFER FOR STABLE LEARNING 🎮
    
    Stores and samples past experiences for efficient DQN training.
    """
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample random batch of experiences"""
        experiences = random.sample(self.buffer, batch_size)
        
        # Convert to tensors
        states = torch.stack([torch.FloatTensor(e.state) for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.stack([torch.FloatTensor(e.next_state) for e in experiences])
        dones = torch.BoolTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    🤖 REVOLUTIONARY DQN AGENT FOR POTHOLE DETECTION! 🤖
    
    This agent learns optimal confidence threshold selection strategies
    through interaction with the video-based RL environment.
    """
    
    def __init__(self, 
                 input_shape=(5, 224, 224, 3), 
                 num_actions=5,
                 learning_rate=0.001,
                 gamma=0.95,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 memory_size=10000,
                 batch_size=32,
                 target_update=100):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Initializing DQN Agent on device: {self.device}")
        
        # Network architecture
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Networks
        self.q_network = PotholeDetectionDQN(input_shape, num_actions).to(self.device)
        self.target_network = PotholeDetectionDQN(input_shape, num_actions).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Experience replay
        self.memory = ExperienceReplayBuffer(memory_size)
        
        # Training tracking
        self.training_step = 0
        self.episode_count = 0
        self.loss_history = []
        self.reward_history = []
        self.epsilon_history = []
        
        print(f"✅ DQN Agent Initialized!")
        print(f"   🧠 Network Parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
        print(f"   💾 Memory Capacity: {memory_size:,}")
        print(f"   🎯 Action Space: {num_actions}")
        print(f"   ⚡ Learning Rate: {learning_rate}")
    
    def act(self, state, training=True):
        """
        Choose action using epsilon-greedy policy
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randrange(self.num_actions)
        
        # Exploitation: best action according to Q-network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        state      = self._to_canonical(state)
        next_state = self._to_canonical(next_state)
        self.memory.push(state, action, reward, next_state, done)

    def _to_canonical(self, array):
        """Guarantee (5,224,224,3) float32 layout."""
        if array.ndim == 3:                       # (5,224,224)  → add channel dim
            array = np.expand_dims(array, -1)     # (5,224,224,1)
            array = np.repeat(array, 3, axis=-1)  # grayscale → 3-channel
        if array.shape != (5,224,224,3):
            raise ValueError(f"Bad state shape {array.shape}")
        return array.astype(np.float32)
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch of experiences
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update training tracking
        self.training_step += 1
        self.loss_history.append(loss.item())
        
        # Update target network
        if self.training_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"🎯 Target network updated at step {self.training_step}")
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
        
        return loss.item()
    
    def train_episode(self, env, max_steps=1000):
        """Train for one episode"""
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        episode_losses = []
        
        while steps < max_steps:
            # Choose action
            action = self.act(state, training=True)
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store experience
            self.remember(state, action, reward, next_state, done)
            
            # Train network
            loss = self.replay()
            if loss is not None:
                episode_losses.append(loss)
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        self.episode_count += 1
        self.reward_history.append(total_reward)
        
        return {
            'episode': self.episode_count,
            'total_reward': total_reward,
            'steps': steps,
            'average_loss': np.mean(episode_losses) if episode_losses else 0,
            'epsilon': self.epsilon,
            'action_taken': action,
            'final_info': info
        }
    
    def evaluate(self, env, num_episodes=10):
        """FIXED: Complete episode evaluation"""
        self.q_network.eval()
        total_rewards = []
        correct_decisions = 0
        false_positives = 0
        missed_detections = 0
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            steps = 0
            max_eval_steps = 100  # Allow more steps for evaluation
            
            # ✅ COMPLETE EPISODE LOOP
            while steps < max_eval_steps:
                action = self.act(state, training=False)
                next_state, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                state = next_state
                steps += 1
                
                # Track performance
                if reward == 10:
                    correct_decisions += 1
                elif reward == -5:
                    false_positives += 1
                elif reward == -20:
                    missed_detections += 1
                
                if done or truncated:
                    break
            
            total_rewards.append(episode_reward)
        
        self.q_network.train()
        
        total_decisions = correct_decisions + false_positives + missed_detections
        accuracy = (correct_decisions / max(total_decisions, 1)) * 100
        
        return {
            'average_reward': np.mean(total_rewards),
            'accuracy': accuracy,
            'correct_decisions': correct_decisions,
            'false_positives': false_positives,
            'missed_detections': missed_detections,
            'total_episodes': num_episodes,
            'total_decisions': total_decisions
        }

    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update': self.target_update
            },
            'training_history': {
                'loss_history': self.loss_history,
                'reward_history': self.reward_history,
                'epsilon_history': self.epsilon_history,
                'training_step': self.training_step,
                'episode_count': self.episode_count
            }
        }, filepath)
        print(f"💾 Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training state
        self.loss_history = checkpoint['training_history']['loss_history']
        self.reward_history = checkpoint['training_history']['reward_history']
        self.epsilon_history = checkpoint['training_history']['epsilon_history']
        self.training_step = checkpoint['training_history']['training_step']
        self.episode_count = checkpoint['training_history']['episode_count']
        
        print(f"📥 Model loaded from: {filepath}")
    
    def plot_training_progress(self, save_path=None):
        """Plot training progress"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Rewards
        ax1.plot(self.reward_history)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # Loss
        if self.loss_history:
            ax2.plot(self.loss_history)
            ax2.set_title('Training Loss')
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Loss')
            ax2.grid(True)
        
        # Epsilon decay
        ax3.plot(self.epsilon_history)
        ax3.set_title('Epsilon Decay')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Epsilon')
        ax3.grid(True)
        
        # Moving average of rewards
        if len(self.reward_history) > 10:
            window_size = min(50, len(self.reward_history) // 10)
            moving_avg = np.convolve(self.reward_history, 
                                   np.ones(window_size)/window_size, mode='valid')
            ax4.plot(moving_avg)
            ax4.set_title(f'Moving Average Rewards (window={window_size})')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Average Reward')
            ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training plots saved to: {save_path}")
        
        plt.show()


# 🧪 TEST DQN AGENT
if __name__ == "__main__":
    print("🚀 TESTING REVOLUTIONARY DQN AGENT!")
    print("="*60)
    
    # Create DQN agent
    agent = DQNAgent(
        input_shape=(5, 224, 224, 3),
        num_actions=5,
        learning_rate=0.001,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    
    print("\n🧪 Testing DQN Agent Components...")
    
    # Test neural network forward pass
    dummy_state = np.random.rand(5, 224, 224, 3).astype(np.float32)
    dummy_state_tensor = torch.FloatTensor(dummy_state).unsqueeze(0)
    
    with torch.no_grad():
        q_values = agent.q_network(dummy_state_tensor)
    
    print(f"✅ Neural Network Test:")
    print(f"   Input shape: {dummy_state.shape}")
    print(f"   Q-values shape: {q_values.shape}")
    print(f"   Q-values: {q_values.squeeze().tolist()}")
    
    # Test action selection
    action = agent.act(dummy_state, training=True)
    print(f"✅ Action Selection Test:")
    print(f"   Selected action: {action}")
    print(f"   Current epsilon: {agent.epsilon:.3f}")
    
    # Test experience storage
    agent.remember(dummy_state, action, 10, dummy_state, False)
    print(f"✅ Experience Replay Test:")
    print(f"   Memory size: {len(agent.memory)}")
    
    print(f"\n🎉 DQN AGENT TEST COMPLETED SUCCESSFULLY!")
    print(f"🚀 Ready for integration with RL Environment!")
