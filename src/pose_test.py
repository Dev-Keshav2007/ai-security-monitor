import cv2
from ultralytics import YOLO


def start_pose_camera():
    print("Loading YOLO pose model...")

    # YOLO pose model detects people and body keypoints
    model = YOLO("yolo11n-pose.pt")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not access camera.")
        return

    print("Camera started.")
    print("Pose detection active.")
    print("Press Q to stop.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read camera frame.")
            break

        # Run pose estimation
        results = model(
            frame,
            conf=0.50,
            verbose=False
        )

        # Draw detected people and body keypoints
        annotated_frame = results[0].plot()

        cv2.imshow(
            "AI Security Monitor - Pose Test",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("Camera stopped.")


if __name__ == "__main__":
    start_pose_camera()