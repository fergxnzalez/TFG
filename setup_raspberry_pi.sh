#!/bin/bash
# Setup script for Raspberry Pi
# Installs dependencies and configures the robot server

set -e

echo "=== Q-Learning Robot Server Setup for Raspberry Pi ==="

# Update system
echo "Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install Python and pip
echo "Installing Python dependencies..."
sudo apt install -y python3 python3-pip python3-dev python3-venv

# Install system dependencies for OpenCV and GPIO
echo "Installing system dependencies..."
sudo apt install -y \
    libopencv-dev python3-opencv \
    libhdf5-dev libhdf5-serial-dev \
    libatlas-base-dev libjasper-dev \
    libqtgui4 libqt4-test \
    git cmake build-essential pkg-config \
    libjpeg-dev libtiff5-dev libjasper-dev libpng-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev \
    libfontconfig1-dev libcairo2-dev \
    libgdk-pixbuf2.0-dev libpango1.0-dev \
    libgtk2.0-dev libgtk-3-dev

# Install GPIO library
echo "Installing RPi.GPIO..."
sudo apt install -y python3-rpi.gpio

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv robot_env
source robot_env/bin/activate

# Install Python packages
echo "Installing Python packages..."
pip install --upgrade pip
pip install opencv-python numpy websockets

# Enable camera and I2C interfaces
echo "Enabling camera and I2C interfaces..."
sudo raspi-config nonint do_camera 0
sudo raspi-config nonint do_i2c 0

# Create robot server directory
echo "Setting up robot server..."
mkdir -p ~/robot_server
cd ~/robot_server

# Copy robot server script (assuming it's in the current directory)
if [ -f ../robot_server.py ]; then
    cp ../robot_server.py .
    echo "Robot server script copied."
else
    echo "Warning: robot_server.py not found. Please copy it manually."
fi

# Create systemd service for auto-start
echo "Creating systemd service..."
cat > robot_server.service << EOF
[Unit]
Description=Q-Learning Robot Server
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/robot_server
Environment=PATH=/home/pi/robot_server/robot_env/bin
ExecStart=/home/pi/robot_server/robot_env/bin/python robot_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Install service
sudo cp robot_server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable robot_server.service

# Create configuration file
echo "Creating configuration file..."
cat > config.json << EOF
{
    "server": {
        "host": "0.0.0.0",
        "port": 8888
    },
    "camera": {
        "index": 0,
        "width": 640,
        "height": 480,
        "fps": 30
    },
    "ultrasonic": {
        "trigger_pin": 18,
        "echo_pin": 24
    },
    "motors": {
        "type": "gpio",
        "left_motor": {
            "forward_pin": 2,
            "backward_pin": 3
        },
        "right_motor": {
            "forward_pin": 4,
            "backward_pin": 14
        }
    }
}
EOF

# Create test script
echo "Creating test script..."
cat > test_sensors.py << 'EOF'
#!/usr/bin/env python3
"""Test script for robot sensors and actuators"""

import sys
import time
import cv2

# Test camera
def test_camera():
    print("Testing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera failed to initialize")
        return False
    
    ret, frame = cap.read()
    if ret:
        height, width = frame.shape[:2]
        print(f"✅ Camera working - Resolution: {width}x{height}")
    else:
        print("❌ Failed to capture frame")
        cap.release()
        return False
    
    cap.release()
    return True

# Test GPIO (ultrasonic sensor)
def test_ultrasonic():
    print("Testing ultrasonic sensor...")
    try:
        import RPi.GPIO as GPIO
        
        # Setup pins
        trigger_pin = 18
        echo_pin = 24
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(trigger_pin, GPIO.OUT)
        GPIO.setup(echo_pin, GPIO.IN)
        GPIO.output(trigger_pin, False)
        
        time.sleep(0.1)
        
        # Send pulse
        GPIO.output(trigger_pin, True)
        time.sleep(0.00001)
        GPIO.output(trigger_pin, False)
        
        # Measure echo
        start_time = time.time()
        timeout = start_time + 0.1
        
        while GPIO.input(echo_pin) == 0 and time.time() < timeout:
            start_time = time.time()
        
        while GPIO.input(echo_pin) == 1 and time.time() < timeout:
            end_time = time.time()
        
        if time.time() >= timeout:
            print("❌ Ultrasonic sensor timeout")
            GPIO.cleanup()
            return False
        
        duration = end_time - start_time
        distance = (duration * 34300) / 2  # cm
        print(f"✅ Ultrasonic sensor working - Distance: {distance:.1f} cm")
        
        GPIO.cleanup()
        return True
        
    except ImportError:
        print("❌ RPi.GPIO not available")
        return False
    except Exception as e:
        print(f"❌ Ultrasonic sensor error: {e}")
        try:
            GPIO.cleanup()
        except:
            pass
        return False

def main():
    print("=== Robot Sensor Test ===")
    
    camera_ok = test_camera()
    ultrasonic_ok = test_ultrasonic()
    
    print("\n=== Test Results ===")
    print(f"Camera: {'✅ OK' if camera_ok else '❌ FAIL'}")
    print(f"Ultrasonic: {'✅ OK' if ultrasonic_ok else '❌ FAIL'}")
    
    if camera_ok and ultrasonic_ok:
        print("\n🎉 All systems ready!")
        return 0
    else:
        print("\n⚠️  Some systems failed. Check connections and configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x test_sensors.py

# Create startup script
echo "Creating startup script..."
cat > start_robot.sh << EOF
#!/bin/bash
# Script to start the robot server

cd /home/pi/robot_server
source robot_env/bin/activate

echo "Starting Q-Learning Robot WebSocket Server..."
echo "Press Ctrl+C to stop"

python robot_server.py --host 0.0.0.0 --port 8888
EOF

chmod +x start_robot.sh

# Print network information
echo ""
echo "=== Network Information ==="
ip_address=$(hostname -I | awk '{print $1}')
echo "Raspberry Pi IP Address: $ip_address"
echo "Server will be available at: $ip_address:8888"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To test the setup:"
echo "  cd ~/robot_server"
echo "  source robot_env/bin/activate"
echo "  python test_sensors.py"
echo ""
echo "To start the robot server manually:"
echo "  ./start_robot.sh"
echo ""
echo "To start the robot server automatically on boot:"
echo "  sudo systemctl start robot_server"
echo ""
echo "To check server status:"
echo "  sudo systemctl status robot_server"
echo ""
echo "Configure your client to connect to: $ip_address:8888"

deactivate 