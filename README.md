Live test/interview tracker

A modular pipeline to track face orientation (head pose), eye movement (gaze direction), and voice activity in real-time using OpenCV and audio processing.

Workflow:
[Capture Frame] → [eye_node] → [head_pose_node] → [voice_node] → [store_node]
eye_node: detects gaze direction

head_pose_node: detects head tilt

voice_node: detects voice/noise presence

store_node: updates shared state

All modules update a central AgentState object.

Python 3.11

How to run:
git clone https://github.com/Kavyanegi0007/live-face-voice-tracker.git
cd live-face-voice-tracker

pip install -r requirements.txt
python main.py
