import cv2
from ultralytics import YOLO


def point_inside_zone(point, zone):
    """Return True if a point is inside the rectangular zone."""
    x, y = point
    x1, y1, x2, y2 = zone

    return x1 <= x <= x2 and y1 <= y <= y2


def start_object_camera():
    print("Loading YOLO object detection model...")

    model = YOLO("yolo11n.pt")

    # Same shelf zone used in our previous tests
    shelf_zone = (50, 150, 300, 450)

    # Remember the bottle's previous state
    bottle_was_in_shelf = False

    # Keep the removal message visible briefly
    removal_message_frames = 0

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not access camera.")
        return

    print("Camera started.")
    print("Bottle shelf tracking active.")
    print("Place the bottle inside the shelf zone.")
    print("Press Q to stop.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read camera frame.")
            break

        results = model(
            frame,
            conf=0.35,
            verbose=False
        )

        shelf_x1, shelf_y1, shelf_x2, shelf_y2 = shelf_zone

        shelf_color = (255, 0, 0)
        shelf_label = "SHELF ZONE"

        bottle_detected = False
        bottle_currently_in_shelf = False

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                object_name = model.names[class_id]

                if object_name != "bottle":
                    continue

                bottle_detected = True

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # Use the center of the bottle to determine its zone
                bottle_center_x = (x1 + x2) // 2
                bottle_center_y = (y1 + y2) // 2

                bottle_center = (
                    bottle_center_x,
                    bottle_center_y
                )

                bottle_currently_in_shelf = point_inside_zone(
                    bottle_center,
                    shelf_zone
                )

                if bottle_currently_in_shelf:
                    bottle_color = (255, 0, 0)
                    bottle_label = (
                        f"BOTTLE IN SHELF {confidence:.0%}"
                    )

                    shelf_color = (0, 255, 0)
                    shelf_label = "PRODUCT ON SHELF"

                else:
                    bottle_color = (0, 255, 0)
                    bottle_label = (
                        f"BOTTLE {confidence:.0%}"
                    )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    bottle_color,
                    2
                )

                cv2.circle(
                    frame,
                    bottle_center,
                    6,
                    bottle_color,
                    -1
                )

                cv2.putText(
                    frame,
                    bottle_label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    bottle_color,
                    2
                )

        # Detect the transition:
        # bottle was previously in shelf, but is now detected outside
        if (
            bottle_was_in_shelf
            and bottle_detected
            and not bottle_currently_in_shelf
        ):
            print("EVENT: Product removed from shelf.")

            removal_message_frames = 90
            bottle_was_in_shelf = False

        # Remember that we have seen the bottle inside the shelf
        if bottle_currently_in_shelf:
            bottle_was_in_shelf = True

        # Show removal warning for several frames
        if removal_message_frames > 0:
            cv2.putText(
                frame,
                "PRODUCT REMOVED FROM SHELF",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )

            removal_message_frames -= 1

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
            0.65,
            shelf_color,
            2
        )

        cv2.imshow(
            "AI Security Monitor - Product Tracking",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("Camera stopped.")


if __name__ == "__main__":
    start_object_camera()