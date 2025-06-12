import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import time
import argparse

from doro.agent.q_learning_agent import QLearningAgent
from doro.environment.simulated import SimulatedRobotEnvironment
from doro.environment.base import RobotEnvironment


class TrainingLogger:
    """Logger to track training progress"""

    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.anomalies_detected = []
        self.anomalies_investigated = []

    def log_episode(
        self,
        episode_reward: float,
        episode_length: int,
        anomalies_detected: int,
        anomalies_investigated: int,
    ):
        """Log episode statistics"""
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.anomalies_detected.append(anomalies_detected)
        self.anomalies_investigated.append(anomalies_investigated)

    def log_loss(self, loss: float):
        """Log training loss"""
        if loss is not None:
            self.losses.append(loss)

    def plot_training_progress(self, save_path: str = "training_progress.png"):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Total Reward")
        axes[0, 0].grid(True)

        # Episode lengths
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_title("Episode Lengths")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Steps")
        axes[0, 1].grid(True)

        # Training loss
        if self.losses:
            axes[1, 0].plot(self.losses)
            axes[1, 0].set_title("Training Loss")
            axes[1, 0].set_xlabel("Training Step")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].grid(True)

        # Anomaly detection performance
        if self.anomalies_detected:
            axes[1, 1].plot(self.anomalies_detected, label="Detected")
            axes[1, 1].plot(self.anomalies_investigated, label="Investigated")
            axes[1, 1].set_title("Anomaly Detection Performance")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

    def print_statistics(self, window_size: int = 100):
        """Print recent training statistics"""
        if len(self.episode_rewards) >= window_size:
            recent_rewards = self.episode_rewards[-window_size:]
            recent_lengths = self.episode_lengths[-window_size:]

            print(f"Last {window_size} episodes:")
            print(
                f"  Average reward: {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}"
            )
            print(
                f"  Average length: {np.mean(recent_lengths):.1f} ± {np.std(recent_lengths):.1f}"
            )

            if len(self.anomalies_detected) >= window_size:
                recent_detected = self.anomalies_detected[-window_size:]
                recent_investigated = self.anomalies_investigated[-window_size:]
                print(f"  Anomalies detected: {np.mean(recent_detected):.2f}")
                print(f"  Anomalies investigated: {np.mean(recent_investigated):.2f}")
                if np.sum(recent_detected) > 0:
                    investigation_rate = np.sum(recent_investigated) / np.sum(
                        recent_detected
                    )
                    print(f"  Investigation rate: {investigation_rate:.2%}")


def train_agent(
    agent: QLearningAgent,
    environment: RobotEnvironment,
    num_episodes: int = 100,
    render: bool = False,
    save_frequency: int = 10,
) -> TrainingLogger:
    """Train the Q-learning agent"""

    logger = TrainingLogger()
    frame_buffer = deque(maxlen=5)

    print(f"Starting training for {num_episodes} episodes...")
    print(f"Device: {agent.device}")

    for episode in range(num_episodes):
        # Reset environment and get initial observation
        observation = environment.reset()
        frame_buffer.clear()
        frame_buffer.append(observation)

        episode_reward = 0.0
        episode_length = 0
        anomalies_detected = 0
        anomalies_investigated = 0

        done = False

        while not done:
            # Get current state from frame buffer
            frames = list(frame_buffer)
            if len(frames) < 5:
                # Pad with current frame if we don't have enough
                frames = [observation] * (5 - len(frames)) + frames

            # Get ultrasonic sensor reading
            ultrasonic_distance = environment.get_ultrasonic_distance()

            state, is_anomaly, distance = agent.get_state(
                frames, observation, ultrasonic_distance
            )

            if is_anomaly:
                print("ANOMALIA")
                anomalies_detected += 1

            # Select action
            action = agent.select_action(state, is_anomaly)

            # Track investigation
            if is_anomaly and action == 0:  # Forward action to investigate
                anomalies_investigated += 1

            # Execute action
            next_observation, env_reward, done, info = environment.step(action)

            # Calculate custom reward
            investigation_progress = 0.0
            if info.get("anomalies_nearby"):
                # Calculate investigation progress based on proximity to anomalies
                nearest_anomaly = min(
                    info["anomalies_nearby"], key=lambda x: x["distance"]
                )
                investigation_progress = max(
                    0, 1.0 - nearest_anomaly["distance"] / 100.0
                )

            reward = agent.calculate_reward(
                is_anomaly,
                distance,
                action,
                investigation_progress,
                ultrasonic_distance,
            )

            # Add frame to buffer
            frame_buffer.append(next_observation)

            # Get next state
            next_frames = list(frame_buffer)
            if len(next_frames) < 5:
                next_frames = [next_observation] * (5 - len(next_frames)) + next_frames

            next_ultrasonic = environment.get_ultrasonic_distance()
            next_state, _, _ = agent.get_state(
                next_frames, next_observation, next_ultrasonic
            )

            # Store experience
            agent.store_experience(state, action, reward, next_state, done)

            # Learn from experience
            loss = agent.learn()
            logger.log_loss(loss)

            # Update for next iteration
            observation = next_observation
            episode_reward += reward
            episode_length += 1

            # Render if requested
            if render:
                environment.render()
                time.sleep(0.01)  # Small delay for visualization

        # Log episode statistics
        logger.log_episode(
            episode_reward, episode_length, anomalies_detected, anomalies_investigated
        )

        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Length: {episode_length}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(
                f"  Anomalies: {anomalies_detected} detected, {anomalies_investigated} investigated"
            )

            if len(agent.memory) > 0:
                print(f"  Memory size: {len(agent.memory)}")

            if len(logger.losses) > 0:
                recent_loss = (
                    np.mean(logger.losses[-10:])
                    if len(logger.losses) >= 10
                    else logger.losses[-1]
                )
                print(f"  Avg loss: {recent_loss:.4f}")
            print()

        # Save models periodically
        if (episode + 1) % save_frequency == 0:
            agent.save_models(f"models/checkpoint_{episode + 1}/")
            print(f"Models saved at episode {episode + 1}")

        # Print statistics every 100 episodes
        if (episode + 1) % 100 == 0:
            logger.print_statistics()
            print("-" * 50)

    return logger


def evaluate_agent(
    agent: QLearningAgent, environment, num_episodes: int = 10, render: bool = True
) -> dict:
    """Evaluate the trained agent"""

    # Set agent to evaluation mode (no exploration)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    results = {
        "episode_rewards": [],
        "episode_lengths": [],
        "anomalies_detected": [],
        "anomalies_investigated": [],
        "investigation_rate": [],
    }

    frame_buffer = deque(maxlen=5)

    print(f"Evaluating agent for {num_episodes} episodes...")

    for episode in range(num_episodes):
        observation = environment.reset()
        frame_buffer.clear()
        frame_buffer.append(observation)

        episode_reward = 0.0
        episode_length = 0
        anomalies_detected = 0
        anomalies_investigated = 0

        done = False

        while not done:
            # Get current state
            frames = list(frame_buffer)
            if len(frames) < 5:
                frames = [observation] * (5 - len(frames)) + frames

            ultrasonic_distance = environment.get_ultrasonic_distance()
            state, is_anomaly, distance = agent.get_state(
                frames, observation, ultrasonic_distance
            )

            if is_anomaly:
                anomalies_detected += 1

            # Select action (no exploration)
            action = agent.select_action(state, is_anomaly)

            if is_anomaly and action == 0:
                anomalies_investigated += 1

            # Execute action
            next_observation, env_reward, done, info = environment.step(action)

            # Calculate reward
            investigation_progress = 0.0
            if info.get("anomalies_nearby"):
                nearest_anomaly = min(
                    info["anomalies_nearby"], key=lambda x: x["distance"]
                )
                investigation_progress = max(
                    0, 1.0 - nearest_anomaly["distance"] / 100.0
                )

            reward = agent.calculate_reward(
                is_anomaly,
                distance,
                action,
                investigation_progress,
                ultrasonic_distance,
            )

            frame_buffer.append(next_observation)
            observation = next_observation
            episode_reward += reward
            episode_length += 1

            if render:
                environment.render()
                time.sleep(0.05)

        # Store results
        results["episode_rewards"].append(episode_reward)
        results["episode_lengths"].append(episode_length)
        results["anomalies_detected"].append(anomalies_detected)
        results["anomalies_investigated"].append(anomalies_investigated)

        investigation_rate = (
            anomalies_investigated / anomalies_detected
            if anomalies_detected > 0
            else 0.0
        )
        results["investigation_rate"].append(investigation_rate)

        print(
            f"Eval Episode {episode + 1}: "
            f"Reward={episode_reward:.2f}, "
            f"Length={episode_length}, "
            f"Anomalies={anomalies_detected}/{anomalies_investigated}, "
            f"Rate={investigation_rate:.2%}"
        )

    # Restore original epsilon
    agent.epsilon = original_epsilon

    # Print summary
    print("\nEvaluation Summary:")
    print(
        f"Average reward: {np.mean(results['episode_rewards']):.2f} ± {np.std(results['episode_rewards']):.2f}"
    )
    print(
        f"Average length: {np.mean(results['episode_lengths']):.1f} ± {np.std(results['episode_lengths']):.1f}"
    )
    print(f"Average anomalies detected: {np.mean(results['anomalies_detected']):.1f}")
    print(f"Average investigation rate: {np.mean(results['investigation_rate']):.2%}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train Q-Learning Agent for Anomaly Investigation"
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of training episodes"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render training environment"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate trained agent"
    )
    parser.add_argument("--load_model", type=str, help="Path to load pretrained models")
    parser.add_argument("--lstm_model", type=str, help="Path to pretrained LSTM model")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Create environment
    env = SimulatedRobotEnvironment(width=640, height=480, max_steps=500)

    # Create agent
    agent = QLearningAgent(
        lstm_model_path=args.lstm_model,
        device=device,
        learning_rate=1e-4,
        epsilon=1.0 if not args.load_model else 0.1,
    )

    # Load pretrained models if specified
    if args.load_model:
        try:
            agent.load_models(args.load_model)
            print(f"Loaded models from {args.load_model}")
        except Exception as e:
            print(f"Failed to load models: {e}")

    try:
        if args.evaluate:
            # Evaluate agent
            results = evaluate_agent(agent, env, num_episodes=20, render=args.render)
        else:
            # Train agent
            logger = train_agent(
                agent,
                env,
                num_episodes=args.episodes,
                render=args.render,
                save_frequency=100,
            )

            # Save final models
            agent.save_models("models/final/")
            print("Final models saved to models/final/")

            # Plot training progress
            logger.plot_training_progress("training_progress.png")
            print("Training progress plot saved to training_progress.png")

            # Final evaluation
            print("\nRunning final evaluation...")
            results = evaluate_agent(agent, env, num_episodes=10, render=False)

    finally:
        env.close()


if __name__ == "__main__":
    main()
