"""
Frame preprocessing utilities for Atari games
"""
import cv2
import numpy as np
from collections import deque


class FramePreprocessor:
    """Preprocess Atari frames for Pong"""
    def __init__(self, frame_size=(84, 84)):
        self.frame_size = frame_size
        self.prev_frame = None
    
    def preprocess(self, frame, use_diff=False):
        """Convert to grayscale, downsample, and optionally compute frame difference"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize to target size
        resized = cv2.resize(gray, self.frame_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        normalized = resized / 255.0
        
        # Frame difference (optional)
        if use_diff and self.prev_frame is not None:
            diff = normalized - self.prev_frame
            self.prev_frame = normalized
            return diff
        else:
            self.prev_frame = normalized
            return normalized
    
    def reset(self):
        self.prev_frame = None


class FrameStack:
    """Stack multiple frames to capture motion"""
    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
    
    def reset(self, frame):
        for _ in range(self.num_frames):
            self.frames.append(frame)
        return self.get_state()
    
    def push(self, frame):
        self.frames.append(frame)
        return self.get_state()
    
    def get_state(self):
        return np.stack(self.frames, axis=0)
