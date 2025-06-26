from typing import TypedDict, List, Dict, Literal, Any
from time import time

# ---------- Type Definitions ---------- #

GazeDirection = Literal["left", "right", "center", "unknown"]
TiltDirection = Literal["left", "right", "up", "down", "neutral"]

class EyeData(TypedDict):
    gaze_direction_log: List[GazeDirection]
    blink_count: int
    total_samples: int

class HeadData(TypedDict):
    tilt_counts: Dict[TiltDirection, int]
    total_samples: int

class AudioData(TypedDict):
    anomaly_detected: bool
    noise_levels: List[float]
    total_samples: int

class AgentState(TypedDict):
    eye_data: EyeData
    head_data: HeadData
    audio_data: AudioData
    current_frame_timestamp: float
    last_action_timestamp: float
    action_log: List[str]
    frame: Any  # Usually an np.ndarray from OpenCV


# ---------- Initial State Generator ---------- #

def get_initial_state() -> AgentState:
    return {
        "eye_data": {
            "gaze_direction_log": [],
            "blink_count": 0,
            "total_samples": 0
        },
        "head_data": {
            "tilt_counts": {
                "left": 0,
                "right": 0,
                "up": 0,
                "down": 0,
                "neutral": 0
            },
            "total_samples": 0
        },
        "audio_data": {
            "anomaly_detected": False,
            "noise_levels": [],
            "total_samples": 0
        },
        "current_frame_timestamp": time(),
        "last_action_timestamp": time(),
        "action_log": [],
        "frame": None
    }
