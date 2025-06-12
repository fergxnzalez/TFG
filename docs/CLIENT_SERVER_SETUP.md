# Q-Learning Robot Client-Server Architecture

This document describes the setup and usage of the distributed Q-learning system where the robot (Raspberry Pi) acts as a server providing sensor data, and a computer acts as a client running the Q-learning agent.

## Architecture Overview

```
┌─────────────────────┐         Network          ┌──────────────────────┐
│   Raspberry Pi      │◄──────────────────────────►│      Computer        │
│   (Robot Server)    │       TCP Socket           │    (Q-Client)        │
│                     │                            │                      │
│  ┌─────────────────┐│                            │ ┌──────────────────┐ │
│  │ Camera          ││                            │ │ Q-Learning Agent │ │
│  │ Ultrasonic      ││                            │ │ LSTM Model       │ │
│  │ Motors          ││                            │ │ ResNet50         │ │
│  │ Robot Control   ││                            │ │ Anomaly Detection│ │
│  └─────────────────┘│                            │ └──────────────────┘ │
└─────────────────────┘                            └──────────────────────┘
```

## Files

- **`src/doro/server/robot_server.py`**: Server script that runs on Raspberry Pi
- **`src/doro/client/robot_client.py`**: Client script that runs on computer
- **`setup_raspberry_pi.sh`**: Automated setup script for Raspberry Pi

## Raspberry Pi Setup (Robot Server)

### 1. Hardware Requirements

- Raspberry Pi 4B (recommended) or Raspberry Pi 3B+
- Camera module or USB camera
- HC-SR04 ultrasonic sensor
- Motor driver (L298N or similar)
- Motors/wheels for robot movement
- MicroSD card (32GB recommended)

### 2. Wiring Diagram

```
Raspberry Pi GPIO Pinout:
┌─────────┬─────────┬─────────────────────┐
│   Pin   │  GPIO   │      Function       │
├─────────┼─────────┼─────────────────────┤
│   12    │   18    │ Ultrasonic Trigger  │
│   18    │   24    │ Ultrasonic Echo     │
│    3    │    2    │ Left Motor Forward  │
│    5    │    3    │ Left Motor Backward │
│    7    │    4    │ Right Motor Forward │
│    8    │   14    │ Right Motor Backward│
└─────────┴─────────┴─────────────────────┘
```

### 3. Software Installation

```bash
# Download the setup script
wget https://raw.githubusercontent.com/your-repo/setup_raspberry_pi.sh

# Make it executable
chmod +x setup_raspberry_pi.sh

# Run the setup (will take 30-60 minutes)
./setup_raspberry_pi.sh
```

Or manually:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3 python3-pip python3-venv python3-opencv python3-rpi.gpio

# Create virtual environment
python3 -m venv robot_env
source robot_env/bin/activate

# Install Python packages
pip install opencv-python numpy

# Enable camera
sudo raspi-config nonint do_camera 0
```

### 4. Configuration

Edit the configuration file:
```bash
cd ~/robot_server
nano config.json
```

```json
{
    "server": {
        "host": "0.0.0.0",
        "port": 8888
    },
    "camera": {
        "index": 0,
        "width": 640,
        "height": 480
    },
    "ultrasonic": {
        "trigger_pin": 18,
        "echo_pin": 24
    }
}
```

### 5. Testing

```bash
cd ~/tfg-fer
source robot_env/bin/activate
python -m doro.server.robot_server --test
```

Expected output:
```
=== Robot Sensor Test ===
Testing camera...
✅ Camera working - Resolution: 640x480
Testing ultrasonic sensor...
✅ Ultrasonic sensor working - Distance: 123.4 cm

=== Test Results ===
Camera: ✅ OK
Ultrasonic: ✅ OK

🎉 All systems ready!
```

### 6. Running the Server

Manual start:
```bash
cd ~/tfg-fer
python -m doro.server.robot_server
```

Auto-start on boot:
```bash
sudo systemctl enable robot_server
sudo systemctl start robot_server
```

Check status:
```bash
sudo systemctl status robot_server
```

## Computer Setup (Q-Learning Client)

### 1. System Requirements

- Python 3.8+
- CUDA-capable GPU (optional but recommended)
- 8GB+ RAM
- Network connectivity to Raspberry Pi

### 2. Installation

```bash
# Clone the repository
git clone <your-repo>
cd tfg-fer

# Install the package in development mode
pip install -e .
```

### 3. Usage

#### Basic Usage

```bash
# Connect to robot and run one episode
python -m doro.client.robot_client --robot-host 192.168.1.100 --episodes 1

# Run continuous learning
python -m doro.client.robot_client --robot-host 192.168.1.100

# Load pretrained models
python -m doro.client.robot_client --robot-host 192.168.1.100 --model-path models/

# Use CPU only
python -m doro.client.robot_client --robot-host 192.168.1.100 --device cpu
```

#### Command Line Options

```
--robot-host      Robot server IP address (default: 192.168.1.100)
--robot-port      Robot server port (default: 8888)
--model-path      Path to pretrained models
--device          Computing device: auto, cpu, cuda
--episodes        Number of episodes (0 for continuous)
--no-render       Disable video display
```

## Network Configuration

### 1. Find Raspberry Pi IP Address

On Raspberry Pi:
```bash
hostname -I
```

### 2. Network Setup Options

#### Option A: Same WiFi Network
- Connect both devices to the same WiFi network
- Use the Pi's WiFi IP address

#### Option B: Ethernet Connection
- Connect Pi to router via Ethernet
- Use Ethernet IP address

#### Option C: Direct Connection
- Connect devices directly via Ethernet cable
- Configure static IP addresses:

On Raspberry Pi:
```bash
sudo nano /etc/dhcpcd.conf
# Add:
interface eth0
static ip_address=192.168.1.100/24
```

On Computer:
```bash
# Set static IP: 192.168.1.101/24
```

### 3. Firewall Configuration

On Raspberry Pi:
```bash
# Allow incoming connections on port 8888
sudo ufw allow 8888
```

## Communication Protocol

### 1. Message Format

#### Robot State (Server → Client)
```json
{
    "ultrasonic_distance": 1.23,
    "frame_data": "base64_encoded_jpeg",
    "timestamp": 1634567890.123,
    "robot_status": "active"
}
```

#### Action Command (Client → Server)
```json
{
    "action": 0
}
```

### 2. Action Mapping
```
0: forward
1: backward
2: left (strafe)
3: right (strafe)
4: turn_right
5: turn_left
```

### 3. Protocol Flow

1. Client connects to server
2. Server sends robot state
3. Client processes state with Q-learning agent
4. Client sends action command
5. Server executes action
6. Repeat from step 2

## Troubleshooting

### Common Issues

#### Connection Issues
```bash
# Check if server is running
sudo systemctl status robot_server

# Check network connectivity
ping 192.168.1.100

# Check if port is open
nmap -p 8888 192.168.1.100
```

#### Camera Issues
```bash
# Test camera directly
raspistill -v -o test.jpg

# Check camera module connection
vcgencmd get_camera
```

#### GPIO Issues
```bash
# Check GPIO permissions
groups $USER | grep gpio

# Add user to gpio group if needed
sudo usermod -a -G gpio $USER
```

### Debug Mode

Enable debug logging:

On Raspberry Pi:
```bash
export LOG_LEVEL=DEBUG
python robot_server.py
```

On Computer:
```bash
export LOG_LEVEL=DEBUG
python q_learning_client.py --robot-host 192.168.1.100
```

### Performance Optimization

#### Network Optimization
- Use wired connection for better stability
- Adjust image compression quality in server
- Monitor network latency

#### Robot Performance
- Ensure adequate power supply for Pi and motors
- Use heat sinks for continuous operation
- Monitor CPU/memory usage

#### Q-Learning Performance
- Use GPU acceleration on client
- Load pretrained models for faster convergence
- Adjust exploration rate for real robot

## Safety Considerations

1. **Emergency Stop**: Implement hardware emergency stop button
2. **Collision Avoidance**: Ultrasonic sensor provides collision detection
3. **Power Management**: Monitor battery levels
4. **Network Security**: Use VPN or secure network
5. **Robot Boundaries**: Implement software boundaries to prevent robot from leaving safe area

## Example Usage Scenarios

### Scenario 1: Remote Monitoring
```bash
# On Raspberry Pi
./start_robot.sh

# On Computer (view only)
python q_learning_client.py --robot-host 192.168.1.100 --episodes 1 --no-render
```

### Scenario 2: Training Session
```bash
# On Computer (continuous learning)
python q_learning_client.py --robot-host 192.168.1.100 --device cuda
```

### Scenario 3: Evaluation with Pretrained Model
```bash
# On Computer
python q_learning_client.py --robot-host 192.168.1.100 --model-path models/ --episodes 5
```

## Integration with Existing Codebase

The client-server architecture is designed to work with the existing Q-learning system:

- **`q_learning_agent.py`**: Used by client for decision making
- **`train_agent.py`**: Can be adapted for distributed training
- **`demo.py`**: Local demonstration mode still available
- **`robot_environment.py`**: Simulated environment for testing

## Future Enhancements

1. **Multi-Robot Support**: Server can handle multiple robot connections
2. **WebSocket Protocol**: For real-time bidirectional communication
3. **Cloud Integration**: Deploy client on cloud platforms
4. **Mobile App**: Control robot via smartphone app
5. **ROS Integration**: Compatibility with Robot Operating System 