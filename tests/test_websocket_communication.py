#!/usr/bin/env python3
"""
Test script for WebSocket communication between robot server and client
This script can be used to test the communication protocol without the full Q-learning system
"""

import asyncio
import websockets
import json
import cv2
import numpy as np
import time
import argparse


class MockRobotServer:
    """Mock robot server for testing"""

    def __init__(self, host="localhost", port=8888):
        self.host = host
        self.port = port

    async def handle_client(self, websocket, path):
        """Handle client connection"""
        print(f"Client connected: {websocket.remote_address}")

        try:
            async for message in websocket:
                # Parse action
                try:
                    data = json.loads(message)
                    action = data.get("action", 0)
                    print(f"Received action: {action}")

                    # Create mock frame
                    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        frame,
                        f"Action: {action}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

                    # Encode frame
                    _, encoded_frame = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                    )
                    frame_bytes = encoded_frame.tobytes()

                    # Create metadata
                    metadata = {
                        "ultrasonic_distance": np.random.uniform(0.1, 3.0),
                        "timestamp": time.time(),
                        "robot_status": "active",
                        "frame_size": len(frame_bytes),
                    }

                    # Send metadata and frame
                    await websocket.send(json.dumps(metadata))
                    await websocket.send(frame_bytes)

                except json.JSONDecodeError:
                    print("Invalid JSON received")

        except websockets.ConnectionClosed:
            print("Client disconnected")

    async def start_server(self):
        """Start the mock server"""
        print(f"Starting mock robot server on {self.host}:{self.port}")
        async with websockets.serve(self.handle_client, self.host, self.port):
            print(f"Server listening on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever


class MockClient:
    """Mock client for testing"""

    def __init__(self, host="localhost", port=8888):
        self.uri = f"ws://{host}:{port}"

    async def run_test(self, num_actions=10):
        """Run test communication"""
        print(f"Connecting to {self.uri}")

        try:
            async with websockets.connect(self.uri) as websocket:
                print("Connected to server")

                for i in range(num_actions):
                    # Send action
                    action = i % 3  # Cycle through actions 0-2
                    message = {"action": action}
                    await websocket.send(json.dumps(message))
                    print(f"Sent action: {action}")

                    # Receive metadata
                    metadata_msg = await websocket.recv()
                    metadata = json.loads(metadata_msg)
                    print(f"Received metadata: {metadata}")

                    # Receive frame
                    frame_data = await websocket.recv()
                    frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                    if frame is not None:
                        print(f"Received frame: {frame.shape}")
                        cv2.imshow("Test Frame", frame)
                        cv2.waitKey(100)
                    else:
                        print("Failed to decode frame")

                    await asyncio.sleep(0.5)

                print("Test completed successfully!")
                cv2.destroyAllWindows()

        except Exception as e:
            print(f"Test failed: {e}")


async def run_server():
    """Run mock server"""
    server = MockRobotServer()
    await server.start_server()


async def run_client(host, port, num_actions):
    """Run mock client"""
    client = MockClient(host, port)
    await client.run_test(num_actions)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test WebSocket Communication")
    parser.add_argument(
        "--mode",
        choices=["server", "client"],
        required=True,
        help="Run as server or client",
    )
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8888, help="Server port")
    parser.add_argument(
        "--actions",
        type=int,
        default=10,
        help="Number of test actions (client mode only)",
    )

    args = parser.parse_args()

    try:
        if args.mode == "server":
            print("=== WebSocket Server Test ===")
            asyncio.run(run_server())
        else:
            print("=== WebSocket Client Test ===")
            asyncio.run(run_client(args.host, args.port, args.actions))

    except KeyboardInterrupt:
        print("\nTest stopped by user")
    except Exception as e:
        print(f"Test error: {e}")


if __name__ == "__main__":
    main()
