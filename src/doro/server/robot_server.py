#!/usr/bin/env python3
"""
Robot Server - Runs on Raspberry Pi
Handles sensor data collection and action execution
Communicates with Q-learning client over TCP
"""

import asyncio
import websockets
import time
import json
import numpy as np
from typing import Optional
import logging
from dataclasses import dataclass

# Import existing sensor classes
#from camera import Camera
#from ultrasonic import Ultrasonic
#from control import Control

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@dataclass
class RobotState:
    """Robot state information"""

    ultrasonic_distance: float
    frame: bytes
    timestamp: float
    robot_status: str = "active"


class RobotServer:
    """Main robot server class using WebSocket"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8888):
        """
        Initialize robot server
        Args:
            host: Server host address
            port: Server port
        """
        self.host = host
        self.port = port
        self.running = False
        self.clients = set()
        
        #Variable
        self.move_point = [325, 635]
        self.action_flag = 1
        self.gait_flag = 1

        # Initialize robot components
        self.ultrasonic = Ultrasonic()
        self.camera = Camera()
        self.control = Control()
        self.control.condition_thread.start()

        logger.info(f"Robot WebSocket server initialized on {host}:{port}")

    async def start_server(self):
        """Start the WebSocket server"""
        try:
            logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
            self.running = True

            async with websockets.serve(self.handle_client, self.host, self.port):
                logger.info(f"WebSocket server listening on {self.host}:{self.port}")
                await asyncio.Future()  # Run forever

        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.cleanup()

    def map(self, value, fromLow, fromHigh, toLow, toHigh):
        return (toHigh - toLow) * (value - fromLow) / (fromHigh - fromLow) + toLow

    def execute_action(self, action: int):
        """Execute robot action using Control class"""
        # 0 --> forward
        # 1 --> turn left
        # 2 --> turn right

        try:
            if action == 0:
                command = ['CMD_MOVE', '1', '0', '35', '10', '0']
            elif action == 1:
                command = ['CMD_MOVE', '2', '-35', '0', '10', '10']
            elif action == 2:
                command = ['CMD_MOVE', '2', '35', '0', '10', '10']
            else:
                command = None

            if command is not None:
                self.control.run_gait(command)
        except Exception as e:
            print(e)

    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection"""
        client_address = websocket.remote_address
        logger.info(f"Client connected from {client_address}")

        self.clients.add(websocket)

        try:
            async for message in websocket:
                # Parse action from client
                try:
                    data = json.loads(message)
                    action = data.get("action")

                    if isinstance(action, int) and 0 <= action <= 2:
                        self.execute_action(action)
                        robot_state = self.get_robot_state()
                        if robot_state:
                            await self.send_state(websocket, robot_state)
                    else:
                        logger.warning(f"Invalid action received: {action}")

                except json.JSONDecodeError:
                    logger.warning("Invalid JSON received from client")
                except Exception as e:
                    logger.error(f"Error processing client message: {e}")

        except websockets.ConnectionClosed:
            logger.info(f"Client {client_address} disconnected")
        except Exception as e:
            logger.error(f"Client handling error: {e}")
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client {client_address} removed from active connections")

    def get_robot_state(self) -> Optional[RobotState]:
        """Get current robot state"""
        try:
            ultrasonic_distance = self.ultrasonic.get_distance()
            frame_bytes = self.camera.get_frame()

            if frame_bytes is None:
                return None

            return RobotState(
                ultrasonic_distance=ultrasonic_distance,
                frame=frame_bytes,
                timestamp=time.time(),
                robot_status="active",
            )

        except Exception as e:
            logger.error(f"Error getting robot state: {e}")
            return None

    async def send_state(self, websocket, state: RobotState):
        """Send robot state to client via WebSocket"""
        try:
            metadata = {
                "ultrasonic_distance": state.ultrasonic_distance,
                "timestamp": state.timestamp,
                "robot_status": state.robot_status,
                "frame_size": len(state.frame),
            }
            await websocket.send(json.dumps(metadata))
            await websocket.send(state.frame)
        except Exception as e:
            logger.error(f"Error sending state: {e}")
            self.clients.discard(websocket)

    def stop_server(self):
        logger.info("Stopping server...")
        self.running = False

    def cleanup(self):
        logger.info("Cleaning up resources...")
        # Close all client connections
        for client in self.clients.copy():
            try:
                asyncio.create_task(client.close())
            except:
                pass
        self.clients.clear()
        # Cleanup camera and sensors
        self.camera.close()
        self.ultrasonic.close()


async def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Robot WebSocket Server for Q-Learning Agent"
    )
    parser.add_argument("--host", type=str, default="192.168.1.100", help="Server host")
    parser.add_argument("--port", type=int, default=8888, help="Server port")

    args = parser.parse_args()

    try:
        # Create server
        server = RobotServer(args.host, args.port)

        # Handle Ctrl+C gracefully
        import signal

        def signal_handler(sig, frame):
            logger.info("Received interrupt signal")
            server.stop_server()

        signal.signal(signal.SIGINT, signal_handler)

        # Start server
        await server.start_server()

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
