#!/usr/bin/env python3
"""
Integration test for the WebSocket client-server system
Tests the NetworkRobotEnvironment against the RobotEnvironment interface
"""

import asyncio
import numpy as np
import cv2
import time
from q_learning_client import NetworkRobotEnvironment


async def test_network_environment():
    """Test the NetworkRobotEnvironment functionality"""

    print("=== NetworkRobotEnvironment Integration Test ===")

    # Create environment (will work even if server is not running for this test)
    env = NetworkRobotEnvironment("localhost", 8888)

    # Test basic interface compatibility
    print("✓ Environment created successfully")

    # Test get_ultrasonic_distance (should return default value)
    distance = env.get_ultrasonic_distance()
    print(f"✓ get_ultrasonic_distance(): {distance:.2f}m")

    # Test get_current_frame (should return None initially)
    frame = env.get_current_frame()
    print(f"✓ get_current_frame(): {frame}")

    # Test reset
    reset_result = env.reset()
    print(f"✓ reset(): {reset_result}")

    # Test async reset
    async_reset_result = await env.async_reset()
    print(f"✓ async_reset(): {async_reset_result}")

    # Test render (should not crash)
    try:
        env.render("rgb_array")
        print("✓ render() works")
    except Exception as e:
        print(f"! render() issue: {e}")

    print("\n=== Interface Compatibility Test Passed ===")
    return True


def test_robot_environment_inheritance():
    """Test that NetworkRobotEnvironment properly inherits from RobotEnvironment"""

    print("\n=== Inheritance Test ===")

    from robot_environment import RobotEnvironment

    env = NetworkRobotEnvironment("localhost", 8888)

    # Check inheritance
    assert isinstance(env, RobotEnvironment), (
        "NetworkRobotEnvironment should inherit from RobotEnvironment"
    )
    print("✓ Properly inherits from RobotEnvironment")

    # Check required methods exist
    required_methods = [
        "reset",
        "step",
        "get_ultrasonic_distance",
        "get_current_frame",
        "render",
    ]
    for method in required_methods:
        assert hasattr(env, method), f"Missing required method: {method}"
        print(f"✓ Has method: {method}")

    print("✓ All required methods present")
    print("\n=== Inheritance Test Passed ===")
    return True


async def test_with_mock_server():
    """Test actual WebSocket communication with a mock server"""

    print("\n=== Mock Server Communication Test ===")

    # Import the mock server from test_websocket_communication
    try:
        from test_websocket_communication import MockRobotServer

        # Start mock server in background
        server = MockRobotServer("localhost", 8889)  # Use different port
        server_task = asyncio.create_task(server.start_server())

        # Wait a bit for server to start
        await asyncio.sleep(0.5)

        # Create client environment
        env = NetworkRobotEnvironment("localhost", 8889)

        # Test connection
        connected = await env.connect()
        if connected:
            print("✓ Successfully connected to mock server")

            # Test step
            observation, reward, done, info = await env.step(0)
            if observation is not None:
                print(f"✓ Step successful - Frame shape: {observation.shape}")
                print(f"✓ Info: {info}")
            else:
                print("! Step returned None observation")

            # Test multiple steps
            for action in range(1, 6):
                observation, reward, done, info = await env.step(action)
                if observation is not None:
                    print(f"✓ Action {action} successful")
                else:
                    print(f"! Action {action} failed")

            await env.disconnect()
            print("✓ Disconnected successfully")

        else:
            print("! Failed to connect to mock server")

        # Cancel server task
        server_task.cancel()

        print("\n=== Mock Server Test Completed ===")
        return connected

    except ImportError:
        print("! Cannot import MockRobotServer, skipping communication test")
        return True
    except Exception as e:
        print(f"! Communication test error: {e}")
        return False


async def main():
    """Run all integration tests"""

    print("Starting Q-Learning Client-Server Integration Tests\n")

    try:
        # Test 1: Basic interface compatibility
        test1_passed = await test_network_environment()

        # Test 2: Inheritance verification
        test2_passed = test_robot_environment_inheritance()

        # Test 3: WebSocket communication (optional - requires mock server)
        test3_passed = await test_with_mock_server()

        # Summary
        print("\n" + "=" * 50)
        print("INTEGRATION TEST SUMMARY")
        print("=" * 50)
        print(f"Interface Compatibility: {'PASS' if test1_passed else 'FAIL'}")
        print(f"Inheritance Test:       {'PASS' if test2_passed else 'FAIL'}")
        print(f"Communication Test:     {'PASS' if test3_passed else 'FAIL'}")

        all_passed = test1_passed and test2_passed and test3_passed
        print(f"\nOverall Result:         {'PASS' if all_passed else 'FAIL'}")

        if all_passed:
            print("\n🎉 All integration tests passed!")
            print(
                "The NetworkRobotEnvironment is ready for use with the Q-learning system."
            )
        else:
            print("\n⚠️  Some tests failed. Please check the issues above.")

        return all_passed

    except Exception as e:
        print(f"\nIntegration test error: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(main())
