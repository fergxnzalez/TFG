import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any
import math
import random

from doro.environment.base import RobotEnvironment


class SimulatedRobotEnvironment(RobotEnvironment):
    """Simulated robot environment for testing and training"""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        max_steps: int = 1000,
        anomaly_probability: float = 0.1,
    ):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.anomaly_probability = anomaly_probability

        # Robot state
        self.robot_x = width // 2
        self.robot_y = height // 2
        self.robot_angle = 0.0  # In radians
        self.step_count = 0

        # Environment objects and anomalies
        self.objects = []
        self.anomalies = []
        self.investigation_targets = []

        # Movement parameters
        self.move_distance = 10
        self.turn_angle = math.pi / 6  # 30 degrees

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the environment"""
        self.robot_x = self.width // 2
        self.robot_y = self.height // 2
        self.robot_angle = 0.0
        self.step_count = 0

        # Generate random objects and anomalies
        self._generate_environment()

        return self.get_current_frame()

    def _generate_environment(self):
        """Generate random environment with objects and anomalies"""
        self.objects = []
        self.anomalies = []

        # Generate normal objects
        num_objects = random.randint(5, 15)
        for _ in range(num_objects):
            x = random.randint(50, self.width - 50)
            y = random.randint(50, self.height - 50)
            size = random.randint(20, 40)
            color = (
                random.randint(100, 200),
                random.randint(100, 200),
                random.randint(100, 200),
            )
            self.objects.append(
                {"x": x, "y": y, "size": size, "color": color, "type": "normal"}
            )

        # Generate anomalies
        num_anomalies = random.randint(1, 3)
        for _ in range(num_anomalies):
            x = random.randint(50, self.width - 50)
            y = random.randint(50, self.height - 50)
            size = random.randint(30, 60)
            # Anomalies have bright/unusual colors
            color = (
                random.randint(200, 255),
                random.randint(0, 100),
                random.randint(200, 255),
            )
            self.anomalies.append(
                {"x": x, "y": y, "size": size, "color": color, "type": "anomaly"}
            )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute action and return results"""
        self.step_count += 1

        # Execute action
        self._execute_action(action)

        # Get new observation
        observation = self.get_current_frame()

        # Calculate reward (will be overridden by agent's reward calculation)
        reward = self._calculate_base_reward(action)

        # Check if episode is done
        done = self.step_count >= self.max_steps

        # Additional info
        info = {
            "step_count": self.step_count,
            "robot_position": (self.robot_x, self.robot_y),
            "robot_angle": self.robot_angle,
            "anomalies_nearby": self._get_nearby_anomalies(),
            "ultrasonic_distance": self.get_ultrasonic_distance(),
        }

        return observation, reward, done, info

    def _execute_action(self, action: int):
        """Execute the given action"""
        if action == 0:  # forward
            self.robot_x += int(self.move_distance * math.cos(self.robot_angle))
            self.robot_y += int(self.move_distance * math.sin(self.robot_angle))
        elif action == 1:  # turn left
            self.robot_angle += self.turn_angle
        elif action == 2:  # turn right
            self.robot_angle -= self.turn_angle

        # Keep robot within bounds
        self.robot_x = max(20, min(self.width - 20, self.robot_x))
        self.robot_y = max(20, min(self.height - 20, self.robot_y))

        # Normalize angle
        self.robot_angle = self.robot_angle % (2 * math.pi)

    def _calculate_base_reward(self, action: int) -> float:
        """Calculate base reward for the action"""
        reward = 0.0

        # Small penalty for each step to encourage efficiency
        reward -= 0.01

        # Penalty for going out of bounds
        if (
            self.robot_x <= 25
            or self.robot_x >= self.width - 25
            or self.robot_y <= 25
            or self.robot_y >= self.height - 25
        ):
            reward -= 0.5

        return reward

    def _get_nearby_anomalies(self, radius: int = 100) -> List[Dict[str, Any]]:
        """Get anomalies within a certain radius of the robot"""
        nearby = []
        for anomaly in self.anomalies:
            distance = math.sqrt(
                (anomaly["x"] - self.robot_x) ** 2 + (anomaly["y"] - self.robot_y) ** 2
            )
            if distance <= radius:
                nearby.append({**anomaly, "distance": distance})
        return nearby

    def get_ultrasonic_distance(self) -> float:
        """Simulate ultrasonic sensor reading by finding closest object in front"""
        # Get robot direction vector
        direction_x = math.cos(self.robot_angle)
        direction_y = math.sin(self.robot_angle)

        min_distance = 5.0  # Maximum sensor range (5 meters)

        # Check distance to all objects and walls
        all_objects = self.objects + self.anomalies

        for obj in all_objects:
            # Vector from robot to object
            obj_vector_x = obj["x"] - self.robot_x
            obj_vector_y = obj["y"] - self.robot_y

            # Check if object is in front of robot (dot product > 0)
            dot_product = obj_vector_x * direction_x + obj_vector_y * direction_y

            if dot_product > 0:  # Object is in front
                # Calculate perpendicular distance to robot's path
                # Project object position onto robot's direction vector
                projection_length = dot_product

                # Find closest point on robot's path to the object
                closest_x = self.robot_x + direction_x * projection_length
                closest_y = self.robot_y + direction_y * projection_length

                # Distance from object center to robot's path
                path_distance = math.sqrt(
                    (obj["x"] - closest_x) ** 2 + (obj["y"] - closest_y) ** 2
                )

                # If object intersects with robot's path (considering object size)
                if path_distance <= obj["size"]:
                    # Distance from robot to intersection point
                    intersection_distance = projection_length - math.sqrt(
                        max(0, obj["size"] ** 2 - path_distance**2)
                    )
                    if intersection_distance > 0:
                        min_distance = min(
                            min_distance, intersection_distance / 100.0
                        )  # Convert pixels to meters

        # Check distance to walls
        if direction_x > 0:  # Moving right
            wall_distance = (self.width - self.robot_x) / 100.0
            min_distance = min(min_distance, wall_distance)
        elif direction_x < 0:  # Moving left
            wall_distance = self.robot_x / 100.0
            min_distance = min(min_distance, wall_distance)

        if direction_y > 0:  # Moving down
            wall_distance = (self.height - self.robot_y) / 100.0
            min_distance = min(min_distance, wall_distance)
        elif direction_y < 0:  # Moving up
            wall_distance = self.robot_y / 100.0
            min_distance = min(min_distance, wall_distance)

        # Add some noise to simulate real sensor
        noise = random.uniform(-0.05, 0.05)  # ±5cm noise
        return max(0.1, min_distance + noise)  # Minimum 10cm reading

    def get_current_frame(self) -> np.ndarray:
        """Generate current camera frame"""
        # Create base image
        frame = (
            np.ones((self.height, self.width, 3), dtype=np.uint8) * 50
        )  # Dark background

        # Draw objects
        for obj in self.objects:
            cv2.circle(frame, (obj["x"], obj["y"]), obj["size"], obj["color"], -1)

        # Draw anomalies (they should look different/unusual)
        for anomaly in self.anomalies:
            # Draw anomaly with special pattern
            cv2.circle(
                frame,
                (anomaly["x"], anomaly["y"]),
                anomaly["size"],
                anomaly["color"],
                -1,
            )
            # Add some noise or pattern to make it more anomalous
            cv2.circle(
                frame,
                (anomaly["x"], anomaly["y"]),
                anomaly["size"] // 2,
                (255, 255, 255),
                2,
            )

        # Draw robot position (optional, for visualization)
        robot_color = (0, 255, 0)  # Green
        cv2.circle(frame, (self.robot_x, self.robot_y), 8, robot_color, -1)

        # Draw robot direction
        end_x = int(self.robot_x + 20 * math.cos(self.robot_angle))
        end_y = int(self.robot_y + 20 * math.sin(self.robot_angle))
        cv2.line(frame, (self.robot_x, self.robot_y), (end_x, end_y), robot_color, 2)

        return frame

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment"""
        frame = self.get_current_frame()

        if mode == "human":
            cv2.imshow("Robot Environment", frame)
            cv2.waitKey(1)
        elif mode == "rgb_array":
            return frame

        return None

    def close(self):
        """Close the environment"""
        cv2.destroyAllWindows()
