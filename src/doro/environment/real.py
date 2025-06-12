"""
Real Robot Environment - Runs on computer
Handles real robot environment and communication with robot server over WebSockets
"""

import websockets
import time
import json
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging

from doro.environment.base import RobotEnvironment
from doro.server.robot_server import RobotState

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RealRobotEnvironment(RobotEnvironment):
    """Robot environment that communicates with robot server over WebSocket"""

    def __init__(self, host: str = "localhost", port: int = 8888):
        """
        Initialize network robot environment
        Args:
            host: Robot server IP address
            port: Robot server port
        """
        self.host = host
        self.port = port
        self.uri = f"ws://{host}:{port}"
        self.websocket = None
        self.connected = False
        self.current_state = None

        logger.info(f"Network robot environment initialized for {self.uri}")

    async def connect(self) -> bool:
        """Connect to robot server"""
        try:
            self.websocket = await websockets.connect(self.uri)
            self.connected = True
            logger.info(f"Connected to robot WebSocket server at {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to robot server: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """Disconnect from robot server"""
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
            self.websocket = None
        self.connected = False
        logger.info("Disconnected from robot server")

    async def receive_state(self) -> Optional[RobotState]:
        """Receive robot state from server"""
        if not self.connected or not self.websocket:
            return None

        try:
            # Receive metadata (text message)
            metadata_message = await self.websocket.recv()
            metadata = json.loads(metadata_message)
            # Receive frame data (binary message)
            frame_data = await self.websocket.recv()

            # Decode frame from bytes
            # TODO: Revisit this
            frame_array = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            if frame is None:
                logger.warning("Failed to decode frame")
                return None

            return RobotState(
                ultrasonic_distance=metadata["ultrasonic_distance"],
                frame=frame,
                timestamp=metadata["timestamp"],
                robot_status=metadata["robot_status"],
            )
        except websockets.ConnectionClosed:
            logger.warning("WebSocket connection closed by server")
            self.connected = False
            return None
        except Exception as e:
            logger.error(f"Error receiving state: {e}")
            self.connected = False
            return None

    async def send_action(self, action: int) -> bool:
        """Send action to robot server"""
        if not self.connected or not self.websocket:
            return False

        try:
            payload = json.dumps({"action": action})
            await self.websocket.send(payload)
            return True
        except websockets.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"Error sending action: {e}")
            self.connected = False
            return False

    def get_ultrasonic_distance(self) -> float:
        if self.current_state:
            return self.current_state.ultrasonic_distance
        return 5.0  # Default safe distance

    def get_current_frame(self) -> Optional[np.ndarray]:
        if self.current_state:
            return self.current_state.frame
        return None

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment (required by RobotEnvironment interface)"""
        if mode == "human" and self.current_state is not None:
            cv2.imshow("Network Robot Environment", self.current_state.frame)
            cv2.waitKey(1)
        elif mode == "rgb_array" and self.current_state is not None:
            return self.current_state.frame
        return None

    def reset(self) -> Optional[np.ndarray]:
        """Reset environment (not applicable for real robot)"""
        # For network environment, we can't really reset the robot
        # Just clear any stored state
        self.current_state = None
        return None

    async def async_reset(self) -> Optional[np.ndarray]:
        """Async version of reset for WebSocket communication"""
        return self.reset()

    async def step(
        self, action: int
    ) -> Tuple[Optional[np.ndarray], float, bool, Dict[str, Any]]:
        """Execute action and get next state"""
        if not self.connected:
            return None, 0.0, True, {"error": "Not connected"}

        # Send action to robot
        action_sent = await self.send_action(action)

        if not action_sent:
            return None, 0.0, True, {"error": "Failed to send action"}

        # Receive new state
        state = await self.receive_state()

        if state is None:
            return None, 0.0, True, {"error": "Failed to receive state"}

        # Update current state
        self.current_state = state

        # Create info dict
        info = {
            "robot_status": state.robot_status,
            "timestamp": state.timestamp,
            "ultrasonic_distance": state.ultrasonic_distance,
            "network_latency": time.time() - state.timestamp,
        }

        # Reward will be calculated by the agent
        reward = 0.0
        done = False

        return state.frame, reward, done, info
