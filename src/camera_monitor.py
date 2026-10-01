import cv2
from ultralytics import YOLO


def start_camera():
    print("Loading AI detection model...")

    model = YOLO("yolo11n.pt")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not access camera.")
        return

    print("Camera started.")
    print("AI person tracking active.")
    print("Press Q to stop.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read camera frame.")
            break

        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=0.50,
            verbose=False
        )

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                if box.id is None:
                    continue

                track_id = int(box.id[0])
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = f"Person #{track_id} {confidence:.0%}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        cv2.imshow(
            "AI Security Monitor - Person Tracking",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("Camera stopped.")


if __name__ == "__main__":
    start_camera()