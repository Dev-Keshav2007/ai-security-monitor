import cv2
import time
import math
from ultralytics import YOLO


def point_inside_zone(point, zone):
    x, y = point
    x1, y1, x2, y2 = zone
    return x1 <= x <= x2 and y1 <= y <= y2


def distance(point1, point2):
    return math.sqrt(
        (point1[0] - point2[0]) ** 2
        + (point1[1] - point2[1]) ** 2
    )


def start_monitor():
    print("Loading AI models...")

    object_model = YOLO("yolo11n.pt")
    pose_model = YOLO("yolo11n-pose.pt")

    # Shelf zone
    shelf_zone = (50, 150, 300, 450)

    # State machine
    product_state = "WAITING"

    shelf_entry_time = None
    interaction_time = None

    shelf_confirmation_time = 1.0
    pickup_window = 3.0

    # Post-pickup variables
    last_product_seen_time = None
    product_near_body = False

    # Don't treat a short YOLO miss as concealment.
    disappearance_threshold = 2.5

    pickup_alert_frames = 0
    review_alert_frames = 0

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not access camera.")
        return

    print("Camera started.")
    print("Post-pickup monitoring active.")
    print("")
    print("Test:")
    print("1. Put bottle in shelf zone.")
    print("2. Wait for PRODUCT READY.")
    print("3. Reach into shelf.")
    print("4. Remove bottle.")
    print("5. Move bottle near torso.")
    print("6. Hide bottle from camera for 3+ seconds.")
    print("")
    print("Press Q to stop.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read camera frame.")
            break

        current_time = time.time()

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

        bottle_center = None
        torso_center = None

        shelf_color = (255, 0, 0)
        shelf_label = "SHELF ZONE"

        # ==================================
        # OBJECT DETECTION
        # ==================================

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

                bottle_in_shelf = point_inside_zone(
                    bottle_center,
                    shelf_zone
                )

                if bottle_in_shelf:
                    bottle_color = (0, 255, 0)
                else:
                    bottle_color = (0, 255, 255)

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
                    f"BOTTLE {confidence:.0%}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    bottle_color,
                    2
                )

        # ==================================
        # POSE / BODY DETECTION
        # ==================================

        for result in pose_results:
            if result.keypoints is None:
                continue

            for keypoints in result.keypoints.xy:

                if len(keypoints) <= 12:
                    continue

                # COCO keypoints
                left_shoulder = keypoints[5]
                right_shoulder = keypoints[6]

                left_hip = keypoints[11]
                right_hip = keypoints[12]

                left_wrist = keypoints[9]
                right_wrist = keypoints[10]

                # ----------------------------------
                # Calculate torso center
                # ----------------------------------

                body_points = [
                    left_shoulder,
                    right_shoulder,
                    left_hip,
                    right_hip
                ]

                valid_body_points = []

                for point in body_points:
                    px = int(point[0])
                    py = int(point[1])

                    if px > 0 and py > 0:
                        valid_body_points.append(
                            (px, py)
                        )

                if len(valid_body_points) >= 2:
                    torso_center = (
                        sum(
                            p[0]
                            for p in valid_body_points
                        ) // len(valid_body_points),
                        sum(
                            p[1]
                            for p in valid_body_points
                        ) // len(valid_body_points)
                    )

                    cv2.circle(
                        frame,
                        torso_center,
                        10,
                        (255, 0, 255),
                        -1
                    )

                    cv2.putText(
                        frame,
                        "TORSO",
                        (
                            torso_center[0] + 10,
                            torso_center[1]
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 255),
                        2
                    )

                # ----------------------------------
                # Wrist interaction
                # ----------------------------------

                wrists = [
                    ("LEFT", left_wrist),
                    ("RIGHT", right_wrist)
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

        # ==================================
        # BOTTLE / BODY RELATIONSHIP
        # ==================================

        if (
            bottle_center is not None
            and torso_center is not None
        ):
            body_distance = distance(
                bottle_center,
                torso_center
            )

            # Initial prototype threshold.
            # Later this should scale with person size.
            if body_distance < 180:
                product_near_body = True

                cv2.line(
                    frame,
                    bottle_center,
                    torso_center,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "PRODUCT NEAR BODY",
                    (
                        bottle_center[0],
                        bottle_center[1] + 30
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2
                )

        # ==================================
        # STATE MACHINE
        # ==================================

        if product_state == "WAITING":

            shelf_label = "WAITING FOR PRODUCT"

            if bottle_in_shelf:

                if shelf_entry_time is None:
                    shelf_entry_time = current_time

                if (
                    current_time - shelf_entry_time
                    >= shelf_confirmation_time
                ):
                    product_state = "ON_SHELF"

                    print(
                        "STATE: Product confirmed on shelf."
                    )

            else:
                shelf_entry_time = None

        elif product_state == "ON_SHELF":

            shelf_color = (0, 255, 0)
            shelf_label = "PRODUCT READY"

            if hand_in_shelf:
                product_state = "INTERACTION"
                interaction_time = current_time

                print(
                    "STATE: Hand-product interaction."
                )

        elif product_state == "INTERACTION":

            shelf_color = (0, 165, 255)
            shelf_label = "HAND-PRODUCT INTERACTION"

            interaction_age = (
                current_time - interaction_time
            )

            if (
                bottle_detected
                and not bottle_in_shelf
                and interaction_age <= pickup_window
            ):
                product_state = "PICKED_UP"

                last_product_seen_time = current_time
                product_near_body = False

                pickup_alert_frames = 100

                print(
                    "EVENT: Product pickup detected."
                )

            elif interaction_age > pickup_window:
                product_state = "ON_SHELF"
                interaction_time = None

                print(
                    "STATE: Interaction expired."
                )

        elif product_state == "PICKED_UP":

            shelf_color = (0, 0, 255)
            shelf_label = "MONITORING PRODUCT"

            # Bottle is still visible
            if bottle_detected:

                last_product_seen_time = current_time

                # Product returned
                if bottle_in_shelf:

                    if shelf_entry_time is None:
                        shelf_entry_time = current_time

                    if (
                        current_time - shelf_entry_time
                        >= shelf_confirmation_time
                    ):
                        product_state = "ON_SHELF"

                        shelf_entry_time = None
                        interaction_time = None
                        product_near_body = False

                        print(
                            "STATE: Product returned to shelf."
                        )

                else:
                    shelf_entry_time = None

            # Bottle disappeared
            else:

                if last_product_seen_time is not None:

                    missing_time = (
                        current_time
                        - last_product_seen_time
                    )

                    if (
                        missing_time
                        >= disappearance_threshold
                        and product_near_body
                    ):
                        product_state = "REVIEW"

                        review_alert_frames = 180

                        print(
                            "ALERT: Possible concealment - review footage."
                        )

        elif product_state == "REVIEW":

            shelf_color = (0, 0, 255)
            shelf_label = "REVIEW EVENT"

            # If bottle becomes visible again,
            # don't automatically claim anything happened.
            if bottle_detected:
                last_product_seen_time = current_time

        # ==================================
        # ALERT DISPLAY
        # ==================================

        if pickup_alert_frames > 0:

            cv2.putText(
                frame,
                "PRODUCT PICKUP DETECTED",
                (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                3
            )

            pickup_alert_frames -= 1

        if review_alert_frames > 0:

            cv2.putText(
                frame,
                "POSSIBLE CONCEALMENT - REVIEW",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                3
            )

            review_alert_frames -= 1

        # ==================================
        # STATE DISPLAY
        # ==================================

        cv2.putText(
            frame,
            f"STATE: {product_state}",
            (30, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # ==================================
        # SHELF DISPLAY
        # ==================================

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
            (
                shelf_x1,
                max(shelf_y1 - 10, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            shelf_color,
            2
        )

        cv2.imshow(
            "AI Security Monitor - Post Pickup Tracking",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("Camera stopped.")


if __name__ == "__main__":
    start_monitor()