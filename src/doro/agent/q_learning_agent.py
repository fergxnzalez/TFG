import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import List, Tuple, Optional
import cv2
from torchvision import models, transforms

# Experience tuple for replay buffer
Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


class DQN(nn.Module):
    """Deep Q-Network for the robot navigation agent"""

    def __init__(
        self, input_size: int = 2049, hidden_size: int = 512, num_actions: int = 3
    ):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, num_actions)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class LSTMFeatureExtractor(nn.Module):
    """LSTM model to process sequence of frames and generate embeddings"""

    def __init__(
        self, input_size: int = 2048, hidden_size: int = 512, num_layers: int = 2
    ):
        super(LSTMFeatureExtractor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm1 = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.1
        )
        self.lstm2 = nn.LSTM(
            hidden_size, hidden_size, num_layers, batch_first=True, dropout=0.1
        )
        self.output_layer = nn.Linear(hidden_size, 2048)  # Output embedding size

    def forward(self, x, hidden_states=None):
        # x shape: (batch_size, sequence_length, input_size)
        if hidden_states is None:
            hidden1 = None
            hidden2 = None
        else:
            hidden1, hidden2 = hidden_states

        lstm1_out, hidden1 = self.lstm1(x, hidden1)
        lstm2_out, hidden2 = self.lstm2(lstm1_out, hidden2)
        output = self.output_layer(lstm2_out[:, -1, :])
        return output, (hidden1, hidden2)


class AnomalyDetector:
    """Handles anomaly detection using embedding comparison with ResNet50"""

    def __init__(
        self,
        threshold: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.resnet50 = models.resnet50(pretrained=True)
        # Remove the final classification layer to get embeddings
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.resnet50.eval()
        self.resnet50.to(device)

        # Image preprocessing for ResNet50
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.anomaly_threshold = threshold

    def get_resnet_embedding(self, image: np.ndarray) -> torch.Tensor:
        """Generate ResNet50 embedding for an image"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB if needed
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.resnet50(image_tensor)
            embedding = embedding.view(embedding.size(0), -1)  # Flatten
            # Pad or truncate to 2048 dimensions
            if embedding.size(1) != 2048:
                if embedding.size(1) < 2048:
                    padding = torch.zeros(1, 2048 - embedding.size(1)).to(self.device)
                    embedding = torch.cat([embedding, padding], dim=1)
                else:
                    embedding = embedding[:, :2048]

        return embedding

    def detect_anomaly(
        self, lstm_embedding: torch.Tensor, image: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Compare LSTM embedding with ResNet50 embedding to detect anomalies
        Returns: (is_anomaly, distance)
        """
        resnet_embedding = self.get_resnet_embedding(image)

        # Calculate cosine distance
        cosine_sim = F.cosine_similarity(lstm_embedding, resnet_embedding, dim=1)
        distance = 1 - cosine_sim.item()

        is_anomaly = distance > self.anomaly_threshold
        return is_anomaly, distance


class QLearningAgent:
    """Q-Learning agent for robot navigation and anomaly investigation"""

    # Action mapping
    ACTIONS = {
        0: "forward",
        1: "turn_left",
        2: "turn_right",
    }

    def __init__(
        self,
        lstm_model_path: Optional[str] = None,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.num_actions = len(self.ACTIONS)

        # Q-Networks
        self.q_network = DQN(num_actions=self.num_actions).to(device)
        self.target_network = DQN(num_actions=self.num_actions).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # LSTM Feature Extractor
        self.lstm_model = LSTMFeatureExtractor().to(device)

        if lstm_model_path:
            self.lstm_model.load_state_dict(
                torch.load(lstm_model_path, map_location=device)
            )

        self.lstm_model.eval()

        # Anomaly Detector
        self.anomaly_detector = AnomalyDetector(device=device)

        # Experience replay
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

        # Training parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.step_count = 0

        # Frame buffer for LSTM (last 5 frames)
        self.frame_buffer = deque(maxlen=5)
        self.lstm_hidden = None

        # Copy weights to target network
        self.update_target_network()

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame to embedding using ResNet50 as base feature"""
        return self.anomaly_detector.get_resnet_embedding(frame)

    def get_lstm_embedding(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Process sequence of frames through LSTM to get embedding"""
        if len(frames) < 5:
            # Pad with the first frame if we don't have enough frames
            while len(frames) < 5:
                frames = [frames[0]] + frames

        # Convert frames to embeddings
        embeddings = []
        for frame in frames[-5:]:  # Take last 5 frames
            embed = self.preprocess_frame(frame)
            embeddings.append(embed)

        # Stack embeddings for LSTM
        sequence = torch.stack(embeddings, dim=1)  # (batch_size, seq_len, embed_size)

        with torch.no_grad():
            lstm_embedding, self.lstm_hidden = self.lstm_model(
                sequence, self.lstm_hidden
            )

        return lstm_embedding

    def get_state(
        self,
        frames: List[np.ndarray],
        current_frame: np.ndarray,
        ultrasonic_distance: float = 0.0,
    ) -> Tuple[torch.Tensor, bool, float]:
        """
        Get state representation and anomaly information
        Returns: (state_with_ultrasonic, is_anomaly, anomaly_distance)
        """
        # Get LSTM embedding from frame sequence
        lstm_embedding = self.get_lstm_embedding(frames)

        # Detect anomaly
        is_anomaly, distance = self.anomaly_detector.detect_anomaly(
            lstm_embedding, current_frame
        )

        # Normalize ultrasonic distance (assuming max range of 5 meters)
        normalized_ultrasonic = min(ultrasonic_distance / 5.0, 1.0)

        # Concatenate LSTM embedding with ultrasonic sensor reading
        ultrasonic_tensor = torch.tensor(
            [[normalized_ultrasonic]], dtype=torch.float32, device=self.device
        )
        state_with_ultrasonic = torch.cat([lstm_embedding, ultrasonic_tensor], dim=1)

        return state_with_ultrasonic, is_anomaly, distance

    def select_action(self, state: torch.Tensor, is_anomaly: bool = False) -> int:
        """Select action using epsilon-greedy policy with anomaly bias"""
        if random.random() < self.epsilon:
            if is_anomaly:
                # Bias towards forward action when anomaly detected
                return (
                    0
                    if random.random() < 0.7
                    else random.randint(0, self.num_actions - 1)
                )
            else:
                return random.randint(0, self.num_actions - 1)

        with torch.no_grad():
            q_values = self.q_network(state)
            if is_anomaly:
                # Boost forward action Q-value when anomaly detected
                q_values[0, 0] += 1.0
            return q_values.argmax().item()

    def store_experience(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ):
        """Store experience in replay buffer"""
        self.memory.append(
            Experience(state.cpu(), action, reward, next_state.cpu(), done)
        )

    def calculate_reward(
        self,
        is_anomaly: bool,
        distance: float,
        action: int,
        investigation_progress: float = 0.0,
        ultrasonic_distance: float = 0.0,
    ) -> float:
        """
        Calculate reward based on anomaly detection and action taken
        """
        reward = 0.0

        # Collision avoidance using ultrasonic sensor
        collision_risk = False
        if ultrasonic_distance < 0.3:  # Less than 30cm - high collision risk
            collision_risk = True
            if action == 0:  # Forward action when obstacle is close
                reward -= 2.0  # Strong penalty for potential collision
            else:  # Turn left or right to avoid collision
                reward += 0.5  # Reward for smart avoidance
        elif ultrasonic_distance < 0.5:  # Less than 50cm - moderate risk
            if action == 0:  # Forward action
                reward -= 0.5  # Moderate penalty
            else:  # Turn left or right for safety
                reward += 0.2  # Small reward for cautious behavior

        # Anomaly investigation rewards
        if is_anomaly and not collision_risk:  # Only investigate if safe
            if action == 0:  # Forward action to investigate
                reward += 2.0 + investigation_progress  # Reward for investigating
            else:
                reward -= 0.3  # Small penalty for not investigating anomaly when safe
        elif is_anomaly and collision_risk:
            # Anomaly detected but collision risk - encourage turning to navigate around
            if action in [1, 2]:  # Turn left or right
                reward += 1.0  # Reward for smart avoidance while still being curious
        else:
            # Normal exploration reward
            if action == 0:  # Forward - exploration
                reward += 0.1
            else:  # Turn actions - useful for exploration but less rewarded than forward
                reward += 0.05

        # Distance-based reward (closer investigation gets higher reward, but only if safe)
        if is_anomaly and action == 0 and not collision_risk:
            reward += max(0, (1.0 - distance))

        # Bonus for maintaining safe distance
        if ultrasonic_distance > 0.5:
            reward += 0.05

        return reward

    def learn(self):
        """Perform one step of learning from experience replay"""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([e.state.squeeze(0) for e in batch]).to(self.device)
        actions = torch.tensor([e.action for e in batch], dtype=torch.long).to(
            self.device
        )
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float).to(
            self.device
        )
        next_states = torch.stack([e.next_state.squeeze(0) for e in batch]).to(self.device)
        dones = torch.tensor([e.done for e in batch], dtype=torch.bool).to(self.device)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_models(self, path_prefix: str = "models/"):
        """Save trained models"""
        import os

        os.makedirs(path_prefix, exist_ok=True)
        torch.save(self.q_network.state_dict(), f"{path_prefix}q_network.pth")
        torch.save(self.target_network.state_dict(), f"{path_prefix}target_network.pth")
        torch.save(self.lstm_model.state_dict(), f"{path_prefix}lstm_model.pth")

    def load_models(self, path_prefix: str = "models/"):
        """Load trained models"""
        self.q_network.load_state_dict(
            torch.load(f"{path_prefix}q_network.pth", map_location=self.device)
        )
        self.target_network.load_state_dict(
            torch.load(f"{path_prefix}target_network.pth", map_location=self.device)
        )
        self.lstm_model.load_state_dict(
            torch.load(f"{path_prefix}lstm_model.pth", map_location=self.device)
        )
