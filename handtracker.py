import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Tells OpenCV which points to connect to create the hand skeleton
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# Set up the hand landmarker with the specified options
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

def main():
    timestamp = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp += 1
        results = detector.detect_for_video(mp_image, timestamp)

        if results.hand_landmarks:
            for hand in results.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

                for a, b in CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 0, 255), 2)

                for cx, cy in pts:
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        cv2.imshow("Hand Tracker", frame)
        if cv2.waitKey(30) & 0xFF in [ord('q'), ord('Q')]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()