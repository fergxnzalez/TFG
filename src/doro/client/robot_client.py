#!/usr/bin/env python3
"""
Q-Learning Client - Runs on computer
Handles Q-learning agent processing and decision making
Communicates with robot server over TCP
"""

import asyncio
import time
from typing import Optional, Dict, Any
from collections import deque

import cv2
import numpy as np
import torch
import logging
import argparse

from doro.agent.q_learning_agent import QLearningAgent
from doro.environment.real import RealRobotEnvironment

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RobotClient:
    """Main Q-learning client class"""

    def __init__(
        self,
        robot_host: str = "192.168.1.100",
        robot_port: int = 8888,
        model_path: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Initialize Q-learning client
        Args:
            robot_host: Robot server IP address
            robot_port: Robot server port
            model_path: Path to pretrained models
            device: Computing device (auto, cpu, cuda)
        """
        self.robot_host = robot_host
        self.robot_port = robot_port

        # Set device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize environment
        self.env = RealRobotEnvironment(robot_host, robot_port)

        # Initialize agent
        self.agent = QLearningAgent(
            device=device,
            epsilon=0.1,  # Small exploration for real robot
        )

        # Load models if provided
        if model_path:
            try:
                self.agent.load_models(model_path)
                logger.info(f"Loaded models from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load models: {e}")

        # State tracking
        self.frame_buffer = deque(maxlen=5)
        self.episode_count = 0
        self.step_count = 0
        self.running = False

        logger.info(f"Q-learning client initialized (device: {device})")

    async def connect_to_robot(self) -> bool:
        """Connect to robot server"""
        return await self.env.connect()

    async def disconnect_from_robot(self):
        """Disconnect from robot server"""
        await self.env.disconnect()

    async def run_episode(
        self, max_steps: int = 1000, render: bool = True
    ) -> Dict[str, Any]:
        """Run one episode of Q-learning"""
        if not self.env.connected:
            logger.error("Not connected to robot")
            return {"error": "Not connected"}

        # Reset environment
        observation = await self.env.async_reset()

        # Initialize episode
        self.frame_buffer.clear()

        # Get initial state by sending a neutral action (0 - forward)
        observation, _, _, _ = await self.env.step(0)
        if observation is None:
            logger.error("Failed to get initial observation")
            return {"error": "Failed to reset"}

        self.frame_buffer.append(observation)

        episode_reward = 0.0
        anomalies_detected = 0
        anomalies_investigated = 0

        logger.info(f"Starting episode {self.episode_count + 1}")

        for step in range(max_steps):
            if not self.env.connected:
                logger.warning("Lost connection to robot")
                break

            # Get current state
            frames = list(self.frame_buffer)
            if len(frames) < 5:
                frames = [observation] * (5 - len(frames)) + frames

            ultrasonic_distance = self.env.get_ultrasonic_distance()
            state, is_anomaly, distance = self.agent.get_state(
                frames, observation, ultrasonic_distance
            )

            # Track anomaly detection
            if is_anomaly:
                anomalies_detected += 1
                logger.info(
                    f"Step {step + 1}: Anomaly detected! Distance: {distance:.3f}"
                )

            # Log sensor data
            logger.info(f"Step {step + 1}: Ultrasonic: {ultrasonic_distance:.2f}m")
            if ultrasonic_distance < 0.3:
                logger.warning("Step {}: COLLISION RISK!".format(step + 1))

            # Select action
            action = self.agent.select_action(state, is_anomaly)
            action_name = self.agent.ACTIONS[action]

            # Track investigation
            if is_anomaly and action == 0:
                anomalies_investigated += 1
                logger.info(f"Step {step + 1}: Investigating anomaly (moving forward)")

            logger.info(f"Step {step + 1}: Selected action: {action_name}")

            # Execute action
            next_observation, env_reward, done, info = await self.env.step(action)

            if next_observation is None:
                logger.error("Failed to get next observation")
                break

            # Calculate reward
            investigation_progress = 0.0
            # Note: We don't have anomalies_nearby info in real environment
            # This would need to be computed differently for real robots

            reward = self.agent.calculate_reward(
                is_anomaly,
                distance,
                action,
                investigation_progress,
                ultrasonic_distance,
            )

            episode_reward += reward

            # Log step results
            latency = info.get("network_latency", 0.0)
            logger.info(
                f"Step {step + 1}: Reward: {reward:.3f}, Latency: {latency:.3f}s"
            )

            # Update frame buffer
            self.frame_buffer.append(next_observation)
            observation = next_observation

            # Render if requested
            if render:
                self.render_state(
                    observation, is_anomaly, ultrasonic_distance, action_name, reward
                )

            # Small delay to prevent overwhelming the robot
            time.sleep(0.1)

            if done:
                break

        # Episode summary
        investigation_rate = (
            anomalies_investigated / anomalies_detected
            if anomalies_detected > 0
            else 0.0
        )

        episode_stats = {
            "episode": self.episode_count + 1,
            "steps": step + 1,
            "total_reward": episode_reward,
            "anomalies_detected": anomalies_detected,
            "anomalies_investigated": anomalies_investigated,
            "investigation_rate": investigation_rate,
        }

        logger.info(f"Episode {self.episode_count + 1} completed:")
        logger.info(f"  Steps: {step + 1}")
        logger.info(f"  Total reward: {episode_reward:.2f}")
        logger.info(
            f"  Anomalies: {anomalies_detected} detected, {anomalies_investigated} investigated"
        )
        logger.info(f"  Investigation rate: {investigation_rate:.2%}")

        self.episode_count += 1
        return episode_stats

    def render_state(
        self,
        frame: np.ndarray,
        is_anomaly: bool,
        ultrasonic_distance: float,
        action: str,
        reward: float,
    ):
        """Render current state for visualization"""
        display_frame = frame.copy()

        # Add overlay information
        height, width = display_frame.shape[:2]
        overlay_y = 30

        # Anomaly status
        if is_anomaly:
            cv2.putText(
                display_frame,
                "ANOMALY DETECTED!",
                (10, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        else:
            cv2.putText(
                display_frame,
                "Normal",
                (10, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        overlay_y += 30

        # Ultrasonic distance
        color = (
            (0, 255, 0)
            if ultrasonic_distance > 0.5
            else (0, 255, 255)
            if ultrasonic_distance > 0.3
            else (0, 0, 255)
        )
        cv2.putText(
            display_frame,
            f"Distance: {ultrasonic_distance:.2f}m",
            (10, overlay_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        overlay_y += 25

        # Current action
        cv2.putText(
            display_frame,
            f"Action: {action}",
            (10, overlay_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        overlay_y += 25

        # Reward
        cv2.putText(
            display_frame,
            f"Reward: {reward:.2f}",
            (10, overlay_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Show frame
        cv2.imshow("Q-Learning Client - Robot View", display_frame)
        cv2.waitKey(1)

    async def run_continuous(self, render: bool = True):
        """Run continuous Q-learning"""
        logger.info("Starting continuous Q-learning mode")
        self.running = True

        try:
            while self.running:
                stats = await self.run_episode(max_steps=500, render=render)

                if "error" in stats:
                    logger.error(f"Episode failed: {stats['error']}")
                    if not self.env.connected:
                        logger.info("Attempting to reconnect...")
                        if not await self.connect_to_robot():
                            logger.error("Failed to reconnect, stopping")
                            break
                        await asyncio.sleep(1)
                        continue

                # Brief pause between episodes
                await asyncio.sleep(2)

        except KeyboardInterrupt:
            logger.info("Stopping continuous mode")
        finally:
            self.running = False
            cv2.destroyAllWindows()

    def stop(self):
        """Stop the client"""
        self.running = False


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Q-Learning WebSocket Client for Robot Control"
    )
    parser.add_argument(
        "--robot-host",
        type=str,
        default="192.168.1.100",
        help="Robot server IP address",
    )
    parser.add_argument(
        "--robot-port", type=int, default=8888, help="Robot server port"
    )
    parser.add_argument("--model-path", type=str, help="Path to pretrained models")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computing device",
    )
    parser.add_argument(
        "--episodes", type=int, default=0, help="Number of episodes (0 for continuous)"
    )
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")

    args = parser.parse_args()

    try:
        # Create client
        client = RobotClient(
            robot_host=args.robot_host,
            robot_port=args.robot_port,
            model_path=args.model_path,
            device=args.device,
        )

        # Connect to robot
        logger.info(f"Connecting to robot at {args.robot_host}:{args.robot_port}")
        if not await client.connect_to_robot():
            logger.error("Failed to connect to robot server")
            return

        # Handle Ctrl+C gracefully
        import signal

        def signal_handler(sig, frame):
            logger.info("Received interrupt signal")
            client.stop()

        signal.signal(signal.SIGINT, signal_handler)

        # Run episodes
        if args.episodes > 0:
            # Run specific number of episodes
            for i in range(args.episodes):
                stats = await client.run_episode(render=not args.no_render)
                if "error" in stats:
                    logger.error(f"Episode {i + 1} failed: {stats['error']}")
                    break
        else:
            # Run continuously
            await client.run_continuous(render=not args.no_render)

    except KeyboardInterrupt:
        logger.info("Client stopped by user")
    except Exception as e:
        logger.error(f"Client error: {e}")
    finally:
        try:
            await client.disconnect_from_robot()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(main())
