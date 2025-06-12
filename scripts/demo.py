import numpy as np
import torch
import cv2
from collections import deque
import time

from doro.agent.q_learning_agent import QLearningAgent
from doro.environment.simulated import SimulatedRobotEnvironment


def run_demo(
    model_path: str = None,
    render: bool = True,
    num_episodes: int = 5,
    device: str = "auto",
):
    """
    Run a demonstration of the trained Q-learning agent
    """

    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Q-Learning Agent Demo")
    print("===================")
    print(f"Device: {device}")
    print(f"Model path: {model_path}")
    print()

    # Create environment
    env = SimulatedRobotEnvironment(width=640, height=480, max_steps=300)

    # Create agent
    agent = QLearningAgent(
        device=device,
        epsilon=0.0,  # No exploration in demo
    )

    # Load trained models if available
    if model_path:
        try:
            agent.load_models(model_path)
            print(f"✓ Loaded trained models from {model_path}")
        except Exception as e:
            print(f"✗ Failed to load models: {e}")
            print("  Using untrained agent for demonstration")
    else:
        print("Using untrained agent for demonstration")

    print()

    # Run demonstration episodes
    frame_buffer = deque(maxlen=5)

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        print("-" * 30)

        # Reset environment
        observation = env.reset()
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

            ultrasonic_distance = env.get_ultrasonic_distance()
            state, is_anomaly, distance = agent.get_state(
                frames, observation, ultrasonic_distance
            )

            # Track anomaly detection
            if is_anomaly:
                anomalies_detected += 1
                print(f"  🚨 ANOMALY DETECTED! Distance: {distance:.3f}")

            # Display ultrasonic sensor reading
            print(f"  📡 Ultrasonic: {ultrasonic_distance:.2f}m")
            if ultrasonic_distance < 0.3:
                print(f"  ⚠️  COLLISION RISK!")

            # Select action
            action = agent.select_action(state, is_anomaly)
            action_name = agent.ACTIONS[action]

            # Track investigation
            if is_anomaly and action == 0:
                anomalies_investigated += 1
                print(f"  🔍 Investigating anomaly (moving forward)")

            print(f"  Step {episode_length + 1}: Action = {action_name}")

            # Execute action
            next_observation, env_reward, done, info = env.step(action)

            # Calculate reward
            investigation_progress = 0.0
            if info.get("anomalies_nearby"):
                nearest_anomaly = min(
                    info["anomalies_nearby"], key=lambda x: x["distance"]
                )
                investigation_progress = max(
                    0, 1.0 - nearest_anomaly["distance"] / 100.0
                )
                print(
                    f"    Nearby anomaly at distance: {nearest_anomaly['distance']:.1f}"
                )

            reward = agent.calculate_reward(
                is_anomaly,
                distance,
                action,
                investigation_progress,
                ultrasonic_distance,
            )

            # Update for next iteration
            frame_buffer.append(next_observation)
            observation = next_observation
            episode_reward += reward
            episode_length += 1

            # Render if requested
            if render:
                env.render()
                time.sleep(0.1)  # Slower for demo

            # Early termination for demo
            if episode_length >= 100:
                break

        # Episode summary
        investigation_rate = (
            anomalies_investigated / anomalies_detected
            if anomalies_detected > 0
            else 0.0
        )

        print(f"Episode Summary:")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Episode length: {episode_length}")
        print(f"  Anomalies detected: {anomalies_detected}")
        print(f"  Anomalies investigated: {anomalies_investigated}")
        print(f"  Investigation rate: {investigation_rate:.2%}")
        print()

        # Wait between episodes
        if render and episode < num_episodes - 1:
            print("Press any key to continue to next episode...")
            cv2.waitKey(0)

    # Close environment
    env.close()
    print("Demo completed!")


def interactive_demo(model_path: str = None, device: str = "auto"):
    """
    Interactive demo where user can control the agent manually
    """

    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Interactive Q-Learning Agent Demo")
    print("=================================")
    print("Controls:")
    print("  w - forward")
    print("  a - turn left")
    print("  d - turn right")
    print("  r - reset environment")
    print("  x - exit")
    print("  SPACE - let agent decide")
    print()

    # Create environment
    env = SimulatedRobotEnvironment(width=640, height=480, max_steps=1000)

    # Create agent
    agent = QLearningAgent(device=device, epsilon=0.0)

    # Load trained models if available
    if model_path:
        try:
            agent.load_models(model_path)
            print(f"✓ Loaded trained models from {model_path}")
        except Exception as e:
            print(f"✗ Failed to load models: {e}")

    # Manual control mapping
    key_to_action = {
        ord("w"): 0,  # forward
        ord("a"): 1,  # turn left
        ord("d"): 2,  # turn right
    }

    frame_buffer = deque(maxlen=5)
    observation = env.reset()
    frame_buffer.append(observation)

    step_count = 0

    while True:
        # Get current state
        frames = list(frame_buffer)
        if len(frames) < 5:
            frames = [observation] * (5 - len(frames)) + frames

        ultrasonic_distance = env.get_ultrasonic_distance()
        state, is_anomaly, distance = agent.get_state(
            frames, observation, ultrasonic_distance
        )

        # Display information
        print(f"\nStep {step_count + 1}")
        print(f"📡 Ultrasonic: {ultrasonic_distance:.2f}m")
        if ultrasonic_distance < 0.3:
            print("⚠️  COLLISION RISK!")
        if is_anomaly:
            print(f"🚨 ANOMALY DETECTED! Distance: {distance:.3f}")
        else:
            print("No anomaly detected")

        # Render environment
        env.render()

        # Get user input
        print("Enter command (or 'h' for help): ", end="")
        key = cv2.waitKey(0) & 0xFF

        if key == ord("x"):
            break
        elif key == ord("r"):
            observation = env.reset()
            frame_buffer.clear()
            frame_buffer.append(observation)
            step_count = 0
            print("Environment reset!")
            continue
        elif key == ord("h"):
            print("\nControls:")
            print("  w - forward")
            print("  a - turn left")
            print("  d - turn right")
            print("  r - reset environment")
            print("  x - exit")
            print("  SPACE - let agent decide")
            continue
        elif key == ord(" "):
            # Let agent decide
            action = agent.select_action(state, is_anomaly)
            print(f"Agent selected: {agent.ACTIONS[action]}")
        elif key in key_to_action:
            action = key_to_action[key]
            print(f"Manual action: {agent.ACTIONS[action]}")
        else:
            print("Invalid key. Press 'h' for help.")
            continue

        # Execute action
        next_observation, env_reward, done, info = env.step(action)

        # Calculate reward
        investigation_progress = 0.0
        if info.get("anomalies_nearby"):
            nearest_anomaly = min(info["anomalies_nearby"], key=lambda x: x["distance"])
            investigation_progress = max(0, 1.0 - nearest_anomaly["distance"] / 100.0)
            print(f"Nearby anomaly at distance: {nearest_anomaly['distance']:.1f}")

        reward = agent.calculate_reward(
            is_anomaly, distance, action, investigation_progress, ultrasonic_distance
        )
        print(f"Reward: {reward:.3f}")

        # Update for next iteration
        frame_buffer.append(next_observation)
        observation = next_observation
        step_count += 1

        if done:
            print("Episode completed!")
            observation = env.reset()
            frame_buffer.clear()
            frame_buffer.append(observation)
            step_count = 0

    env.close()
    print("Interactive demo ended!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Q-Learning Agent Demo")
    parser.add_argument("--model", type=str, help="Path to trained models")
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of demo episodes"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run interactive demo"
    )
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_demo(args.model, args.device)
    else:
        run_demo(args.model, not args.no_render, args.episodes, args.device)


if __name__ == "__main__":
    main()
