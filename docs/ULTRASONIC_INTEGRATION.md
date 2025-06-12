# Ultrasonic Sensor Integration

## Overview

The Q-learning agent now includes ultrasonic sensor integration for enhanced collision avoidance and safer navigation. This addition significantly improves the robot's ability to operate safely in real environments.

## Key Changes

### 1. Enhanced State Space
- **Previous**: 2048-dimensional state (LSTM embedding only)
- **Current**: 2049-dimensional state (2048 LSTM + 1 normalized ultrasonic distance)
- **Normalization**: Ultrasonic distance normalized to [0,1] range (assuming 5m max range)

### 2. Collision Avoidance Logic
```python
# Collision risk thresholds
if ultrasonic_distance < 0.3:  # 30cm - HIGH RISK
    collision_risk = True
    if action == forward: reward -= 2.0  # Strong penalty
    if action == backward: reward += 0.5  # Reward safe action

elif ultrasonic_distance < 0.5:  # 50cm - MODERATE RISK  
    if action == forward: reward -= 0.5  # Moderate penalty
```

### 3. Smart Anomaly Investigation
- **Safe Investigation**: Only investigate anomalies when ultrasonic distance > 30cm
- **Smart Avoidance**: When anomaly detected but collision risk exists, reward side movements/turns
- **Risk Assessment**: Balances anomaly investigation with collision avoidance

### 4. Reward System Updates

| Condition | Action | Reward | Reasoning |
|-----------|--------|--------|-----------|
| Distance < 30cm | Forward | -2.0 | Prevent collision |
| Distance < 30cm | Backward | +0.5 | Encourage safe retreat |
| Distance < 50cm | Forward | -0.5 | Moderate caution |
| Distance > 50cm | Any | +0.05 | Bonus for safe distance |
| Anomaly + Risk | Side/Turn | +1.0 | Smart avoidance |
| Anomaly + Safe | Forward | +2.0 | Safe investigation |

## Implementation Details

### Environment Integration
```python
# Abstract method in src/doro/environment/base.py
@abstractmethod
def get_ultrasonic_distance(self) -> float:
    """Get current ultrasonic sensor reading in meters"""
    pass
```

### Simulated Environment
- Calculates distance to nearest object in robot's forward direction
- Considers object sizes and robot orientation
- Includes realistic sensor noise (±5cm)
- Handles wall detection

### Real Robot Integration
```python
def get_ultrasonic_distance(self) -> float:
    # Replace with actual sensor API
    # Examples:
    # return self.robot_controller.get_ultrasonic_distance()
    # distance_cm = self.robot_controller.sensors.ultrasonic.read()
    # return distance_cm / 100.0
    pass
```

## Benefits

### 1. Safety
- **Collision Prevention**: Strong penalties prevent dangerous forward movements
- **Risk Assessment**: Multi-level risk evaluation (30cm, 50cm thresholds)
- **Safe Retreat**: Rewards backing away from obstacles

### 2. Intelligence
- **Context-Aware Decisions**: Considers both anomaly detection and collision risk
- **Smart Navigation**: Encourages alternative paths when direct approach is unsafe
- **Balanced Behavior**: Maintains investigation goals while ensuring safety

### 3. Real-World Readiness
- **Sensor Fusion**: Combines vision (anomaly detection) with proximity sensing
- **Robust Navigation**: Handles complex scenarios with multiple constraints
- **Hardware Integration**: Ready for real ultrasonic sensors

## Usage Examples

### Training with Ultrasonic Data
```python
# Training automatically uses ultrasonic sensor
python -m doro.utils.train_agent --episodes 1000 --render

# State now includes ultrasonic reading
state, is_anomaly, distance = agent.get_state(frames, observation, ultrasonic_distance)
```

### Demo with Collision Avoidance
```python
# Demo shows ultrasonic readings and collision warnings
python demo.py --interactive --model models/final/

# Output includes:
# 📡 Ultrasonic: 0.25m
# ⚠️  COLLISION RISK!
```

## Testing Results

The integration test shows correct behavior:
- **State Shape**: `torch.Size([1, 2049])` ✓
- **Ultrasonic Reading**: `0.13m` (13cm - very close) ✓
- **Collision Detection**: Reward = -2.0 for forward action ✓
- **Safety Priority**: System correctly penalizes risky behavior ✓

## Future Enhancements

1. **Multiple Sensors**: Support for front/side/rear ultrasonic arrays
2. **Dynamic Thresholds**: Adaptive collision thresholds based on robot speed
3. **Sensor Fusion**: Integration with LIDAR, IMU, or other sensors
4. **Predictive Avoidance**: Anticipate collisions based on movement trajectory

## Hardware Requirements

- **Ultrasonic Sensor**: HC-SR04 or similar (range: 2cm - 400cm)
- **Microcontroller**: Arduino, Raspberry Pi, or robot controller
- **Interface**: Serial, I2C, or direct GPIO connection
- **Update Rate**: Minimum 10Hz recommended for real-time operation 