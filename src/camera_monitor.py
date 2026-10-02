import cv2
from ultralytics import YOLO


def point_inside_zone(point, zone):
    """Check whether a point is inside a rectangular zone."""
    x, y = point
    x1, y1, x2, y2 = zone

    return x1 <= x <= x2 and y1 <= y <= y2


def start_camera():
    print("Loading AI pose model...")

    # Pose model gives us body keypoints such as wrists
    model = YOLO("yolo11n-pose.pt")

    # Shelf zone: left, top, right, bottom
    shelf_zone = (50, 150, 300, 450)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not access camera.")
        return

    print("Camera started.")
    print("AI pose detection active.")
    print("Wrist-to-shelf interaction detection active.")
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

        shelf_x1, shelf_y1, shelf_x2, shelf_y2 = shelf_zone

        # Normal shelf appearance
        shelf_color = (255, 0, 0)
        shelf_label = "SHELF ZONE"

        for result in results:
            if result.keypoints is None:
                continue

            # Go through each detected person
            for person_index, keypoints in enumerate(
                result.keypoints.xy
            ):
                # COCO pose keypoints:
                # 9  = left wrist
                # 10 = right wrist
                if len(keypoints) <= 10:
                    continue

                left_wrist = keypoints[9]
                right_wrist = keypoints[10]

                wrists = [
                    ("LEFT", left_wrist),
                    ("RIGHT", right_wrist)
                ]

                for wrist_name, wrist in wrists:
                    wrist_x = int(wrist[0])
                    wrist_y = int(wrist[1])

                    # A missing/undetected keypoint can appear as (0, 0)
                    if wrist_x <= 0 or wrist_y <= 0:
                        continue

                    wrist_point = (wrist_x, wrist_y)

                    # Check if wrist is inside shelf zone
                    inside_shelf = point_inside_zone(
                        wrist_point,
                        shelf_zone
                    )

                    if inside_shelf:
                        wrist_color = (0, 0, 255)

                        shelf_color = (0, 0, 255)
                        shelf_label = (
                            f"HAND-SHELF INTERACTION "
                            f"({wrist_name} WRIST)"
                        )
                    else:
                        wrist_color = (0, 255, 255)

                    # Draw wrist marker
                    cv2.circle(
                        frame,
                        wrist_point,
                        10,
                        wrist_color,
                        -1
                    )

                    # Label wrist
                    cv2.putText(
                        frame,
                        f"{wrist_name} WRIST",
                        (wrist_x + 10, wrist_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        wrist_color,
                        2
                    )

        # Draw shelf zone
        cv2.rectangle(
            frame,
            (shelf_x1, shelf_y1),
            (shelf_x2, shelf_y2),
            shelf_color,
            2
        )

        cv2.putText(
            frame,
            shelf_label,
            (shelf_x1, max(shelf_y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            shelf_color,
            2
        )

        # Draw the normal YOLO pose skeleton
        if results:
            pose_frame = results[0].plot(
                img=frame
            )
        else:
            pose_frame = frame

        cv2.imshow(
            "AI Security Monitor - Hand Shelf Interaction",
            pose_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("Camera stopped.")


if __name__ == "__main__":
    start_camera()