import numpy as np
import sounddevice as sd
from graph.state import AgentState
import threading

audio_lock = threading.Lock()

def record_audio(duration=0.5, samplerate=16000):
    try:
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()
        return audio.flatten()
    except Exception as e:
        print(f"[VoiceNode] Audio capture failed: {e}")
        return np.array([])

def classify_audio(audio: np.ndarray, threshold=0.02) -> str:
    if len(audio) == 0:
        return "noise"
    rms = np.sqrt(np.mean(audio ** 2))
    return "voice" if rms > threshold else "noise"

def voice_node(state: AgentState) -> AgentState:
    audio = record_audio()
    label = classify_audio(audio)

    with audio_lock:
        if len(audio) > 0:
            rms = float(np.sqrt(np.mean(audio ** 2)))
            state["audio_data"]["noise_levels"].append(rms)
        else:
            state["audio_data"]["noise_levels"].append(0.0)

        state["audio_data"]["total_samples"] += 1
        if label == "voice":
            state["audio_data"]["anomaly_detected"] = True  # Optional

    print(f"[VoiceNode] Audio classified as: {label}")
    return state
