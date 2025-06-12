import numpy as np
from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod


class RobotEnvironment(ABC):
    """Abstract base class for robot environment"""

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute action and return next observation, reward, done flag, and info
        Returns: (observation, reward, done, info)
        """
        pass

    @abstractmethod
    def get_ultrasonic_distance(self) -> float:
        """Get current ultrasonic sensor reading in meters"""
        pass

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation"""
        pass

    @abstractmethod
    def get_current_frame(self) -> np.ndarray:
        """Get current camera frame"""
        pass
