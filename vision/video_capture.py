import cv2

def get_video_frame(cap):
    success, frame = cap.read()
    if not success:
        raise RuntimeError("Failed to capture frame from camera.")
    return frame
