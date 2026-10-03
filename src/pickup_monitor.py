import cv2
import time
from ultralytics import YOLO


def point_inside_zone(point, zone):
    """Check whether a point is inside a rectangular zone."""
    x, y = point
    x1, y1, x2, y2 = zone

    return x1 <= x <= x2 and y1 <= y <= y2


def start_pickup_monitor():
    print("Loading AI models...")

    # Object detection model
    object_model = YOLO("yolo11n.pt")

    # Human pose model
    pose_model = YOLO("yolo11n-pose.pt")

    # Shelf zone: left, top, right, bottom
    shelf_zone = (50, 150, 300, 450)

    # --------------------------------
    # Product state machine
    # --------------------------------
    #
    # WAITING:
    # No confirmed product on shelf yet.
    #
    # ON_SHELF:
    # Product has remained on shelf long enough
    # to be considered ready for pickup.
    #
    # INTERACTION:
    # A wrist entered the shelf while product
    # was confirmed on shelf.
    #
    # PICKED_UP:
    # Product left shelf after interaction.
    # No more pickup events are allowed until
    # product is returned and stabilized.
    #
    product_state = "WAITING"

    # Timing variables
    shelf_entry_time = None
    interaction_time = None

    # Product must stay on shelf this long
    # before system arms itself.
    shelf_confirmation_time = 1.0

    # Product must leave within this many seconds
    # after hand interaction.
    pickup_window = 3.0

    # Screen alert duration
    pickup_alert_frames = 0

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not access camera.")
        return

    print("Camera started.")
    print("Pickup state machine active.")
    print("")
    print("Test sequence:")
    print("1. Put bottle in shelf zone.")
    print("2. Wait for PRODUCT READY.")
    print("3. Reach into shelf zone.")
    print("4. Remove bottle.")
    print("5. Return bottle to shelf.")
    print("6. Wait for PRODUCT READY again.")
    print("")
    print("Press Q to stop.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read camera frame.")
            break

        current_time = time.time()

        # --------------------------------
        # Run AI models
        # --------------------------------

        object_results = object_model(
            frame,
            conf=0.35,
            verbose=False
        )

        pose_results = pose_model(
            frame,
            conf=0.50,
            verbose=False
        )

        shelf_x1, shelf_y1, shelf_x2, shelf_y2 = shelf_zone

        bottle_detected = False
        bottle_in_shelf = False
        hand_in_shelf = False

        shelf_color = (255, 0, 0)
        shelf_label = "SHELF ZONE"

        # --------------------------------
        # Bottle detection
        # --------------------------------

        for result in object_results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                object_name = object_model.names[class_id]

                if object_name != "bottle":
                    continue

                bottle_detected = True

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                bottle_center = (
                    (x1 + x2) // 2,
                    (y1 + y2) // 2
                )

                current_in_shelf = point_inside_zone(
                    bottle_center,
                    shelf_zone
                )

                if current_in_shelf:
                    bottle_in_shelf = True
                    bottle_color = (0, 255, 0)

                    cv2.putText(
                        frame,
                        f"BOTTLE {confidence:.0%}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        bottle_color,
                        2
                    )

                else:
                    bottle_color = (0, 255, 255)

                    cv2.putText(
                        frame,
                        f"BOTTLE {confidence:.0%}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        bottle_color,
                        2
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

        # --------------------------------
        # Wrist detection
        # --------------------------------

        for result in pose_results:
            if result.keypoints is None:
                continue

            for keypoints in result.keypoints.xy:

                if len(keypoints) <= 10:
                    continue

                wrists = [
                    ("LEFT", keypoints[9]),
                    ("RIGHT", keypoints[10])
                ]

                for wrist_name, wrist in wrists:
                    wrist_x = int(wrist[0])
                    wrist_y = int(wrist[1])

                    if wrist_x <= 0 or wrist_y <= 0:
                        continue

                    wrist_point = (
                        wrist_x,
                        wrist_y
                    )

                    wrist_inside = point_inside_zone(
                        wrist_point,
                        shelf_zone
                    )

                    if wrist_inside:
                        hand_in_shelf = True
                        wrist_color = (0, 0, 255)
                    else:
                        wrist_color = (0, 255, 255)

                    cv2.circle(
                        frame,
                        wrist_point,
                        8,
                        wrist_color,
                        -1
                    )

                    cv2.putText(
                        frame,
                        f"{wrist_name} WRIST",
                        (wrist_x + 10, wrist_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        wrist_color,
                        1
                    )

        # ========================================
        # STATE MACHINE
        # ========================================

        # --------------------------------
        # WAITING
        # --------------------------------

        if product_state == "WAITING":

            shelf_label = "WAITING FOR PRODUCT"

            if bottle_in_shelf:

                if shelf_entry_time is None:
                    shelf_entry_time = current_time

                time_on_shelf = (
                    current_time - shelf_entry_time
                )

                if time_on_shelf >= shelf_confirmation_time:
                    product_state = "ON_SHELF"

                    print(
                        "STATE: Product confirmed on shelf."
                    )

            else:
                shelf_entry_time = None

        # --------------------------------
        # ON SHELF
        # --------------------------------

        elif product_state == "ON_SHELF":

            shelf_color = (0, 255, 0)
            shelf_label = "PRODUCT READY"

            if hand_in_shelf:
                product_state = "INTERACTION"
                interaction_time = current_time

                print(
                    "STATE: Hand-product interaction."
                )

        # --------------------------------
        # INTERACTION
        # --------------------------------

        elif product_state == "INTERACTION":

            shelf_color = (0, 165, 255)
            shelf_label = "HAND-PRODUCT INTERACTION"

            interaction_age = (
                current_time - interaction_time
            )

            # Product is visible outside shelf
            if (
                bottle_detected
                and not bottle_in_shelf
                and interaction_age <= pickup_window
            ):
                product_state = "PICKED_UP"

                pickup_alert_frames = 120

                print(
                    "EVENT: Product pickup detected."
                )

            # Interaction expired without pickup
            elif interaction_age > pickup_window:
                product_state = "ON_SHELF"
                interaction_time = None

                print(
                    "STATE: Interaction expired."
                )

        # --------------------------------
        # PICKED UP
        # --------------------------------

        elif product_state == "PICKED_UP":

            shelf_color = (0, 0, 255)
            shelf_label = "PRODUCT PICKED UP"

            # Product must be returned to shelf
            # before another pickup can occur.
            if bottle_in_shelf:

                if shelf_entry_time is None:
                    shelf_entry_time = current_time

                returned_time = (
                    current_time - shelf_entry_time
                )

                if returned_time >= shelf_confirmation_time:
                    product_state = "ON_SHELF"

                    shelf_entry_time = None
                    interaction_time = None

                    print(
                        "STATE: Product returned to shelf."
                    )

            else:
                # Reset timer whenever product
                # leaves shelf again.
                shelf_entry_time = None

        # --------------------------------
        # Pickup alert
        # --------------------------------

        if pickup_alert_frames > 0:
            cv2.putText(
                frame,
                "PRODUCT PICKUP DETECTED",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )

            pickup_alert_frames -= 1

        # --------------------------------
        # Display current state
        # --------------------------------

        cv2.putText(
            frame,
            f"STATE: {product_state}",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # --------------------------------
        # Draw shelf
        # --------------------------------

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

        cv2.imshow(
            "AI Security Monitor - Pickup State Machine",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("Camera stopped.")


if __name__ == "__main__":
    start_pickup_monitor()