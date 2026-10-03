import cv2
import time
import math
from ultralytics import YOLO


def point_inside_zone(point, zone):
    x, y = point
    x1, y1, x2, y2 = zone
    return x1 <= x <= x2 and y1 <= y <= y2


def calculate_distance(point1, point2):
    return math.sqrt(
        (point1[0] - point2[0]) ** 2
        + (point1[1] - point2[1]) ** 2
    )


def start_monitor():
    print("Loading AI models...")

    object_model = YOLO("yolo11n.pt")
    pose_model = YOLO("yolo11n-pose.pt")

    # ---------------------------------------
    # Shelf configuration
    # ---------------------------------------

    shelf_zone = (50, 150, 300, 450)

    # ---------------------------------------
    # State machine
    # ---------------------------------------

    product_state = "WAITING"

    shelf_entry_time = None
    interaction_time = None
    pickup_time = None

    last_product_seen_time = None
    last_product_position = None

    # ---------------------------------------
    # Near-body evidence
    # ---------------------------------------

    near_body_start_time = None
    last_near_body_time = None
    product_was_near_body = False

    # ---------------------------------------
    # Timing settings
    # ---------------------------------------

    shelf_confirmation_time = 1.0
    pickup_window = 3.0
    carry_confirmation_time = 0.8

    # Bottle must remain near the body this
    # long before it becomes valid evidence.
    near_body_confirmation_time = 0.7

    # Ignore short object-detector failures.
    disappearance_threshold = 3.0

    # How recently valid near-body evidence
    # must have occurred before disappearance.
    near_body_memory = 2.0

    pickup_alert_frames = 0
    review_alert_frames = 0

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not access camera.")
        return

    print("Camera started.")
    print("Concealment monitoring active.")
    print("")
    print("Important:")
    print("Near-body evidence is only valid AFTER pickup.")
    print("")
    print("Press Q to stop.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read camera frame.")
            break

        current_time = time.time()

        # =======================================
        # RUN AI MODELS
        # =======================================

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

        # =======================================
        # BOTTLE DETECTION
        # =======================================

        best_bottle = None
        best_confidence = 0.0

        for result in object_results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                object_name = object_model.names[class_id]

                if object_name != "bottle":
                    continue

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_bottle = box

        if best_bottle is not None:
            bottle_detected = True

            x1, y1, x2, y2 = map(
                int,
                best_bottle.xyxy[0]
            )

            bottle_center = (
                (x1 + x2) // 2,
                (y1 + y2) // 2
            )

            last_product_position = bottle_center
            last_product_seen_time = current_time

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
                f"BOTTLE {best_confidence:.0%}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                bottle_color,
                2
            )

        # =======================================
        # POSE DETECTION
        # =======================================

        for result in pose_results:
            if result.keypoints is None:
                continue

            for keypoints in result.keypoints.xy:
                if len(keypoints) <= 12:
                    continue

                left_shoulder = keypoints[5]
                right_shoulder = keypoints[6]

                left_wrist = keypoints[9]
                right_wrist = keypoints[10]

                left_hip = keypoints[11]
                right_hip = keypoints[12]

                # --------------------------------
                # Torso center
                # --------------------------------

                torso_points = []

                for point in [
                    left_shoulder,
                    right_shoulder,
                    left_hip,
                    right_hip
                ]:
                    px = int(point[0])
                    py = int(point[1])

                    if px > 0 and py > 0:
                        torso_points.append((px, py))

                if len(torso_points) >= 2:
                    torso_center = (
                        sum(p[0] for p in torso_points)
                        // len(torso_points),

                        sum(p[1] for p in torso_points)
                        // len(torso_points)
                    )

                    cv2.circle(
                        frame,
                        torso_center,
                        9,
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
                        0.45,
                        (255, 0, 255),
                        2
                    )

                # --------------------------------
                # Wrist detection
                # --------------------------------

                for wrist in [
                    left_wrist,
                    right_wrist
                ]:
                    wrist_x = int(wrist[0])
                    wrist_y = int(wrist[1])

                    if wrist_x <= 0 or wrist_y <= 0:
                        continue

                    wrist_point = (
                        wrist_x,
                        wrist_y
                    )

                    if point_inside_zone(
                        wrist_point,
                        shelf_zone
                    ):
                        hand_in_shelf = True
                        wrist_color = (0, 0, 255)
                    else:
                        wrist_color = (0, 255, 255)

                    cv2.circle(
                        frame,
                        wrist_point,
                        7,
                        wrist_color,
                        -1
                    )

        # =======================================
        # PRODUCT / BODY RELATIONSHIP
        # =======================================

        product_currently_near_body = False

        if (
            bottle_center is not None
            and torso_center is not None
            and not bottle_in_shelf
        ):
            body_distance = calculate_distance(
                bottle_center,
                torso_center
            )

            if body_distance < 180:
                product_currently_near_body = True

                cv2.line(
                    frame,
                    bottle_center,
                    torso_center,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "NEAR BODY",
                    (
                        bottle_center[0],
                        bottle_center[1] + 30
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2
                )

        # =======================================
        # VALID NEAR-BODY EVIDENCE
        # =======================================
        #
        # IMPORTANT FIX:
        #
        # Near-body evidence is ONLY allowed
        # after a real pickup has happened.
        #
        # A bottle near someone's body while
        # WAITING or ON_SHELF does not count.
        # =======================================

        evidence_allowed = product_state in [
            "PICKED_UP",
            "CARRIED"
        ]

        if (
            evidence_allowed
            and product_currently_near_body
        ):
            if near_body_start_time is None:
                near_body_start_time = current_time

            near_body_duration = (
                current_time - near_body_start_time
            )

            if (
                near_body_duration
                >= near_body_confirmation_time
            ):
                if not product_was_near_body:
                    print(
                        "EVIDENCE: Product confirmed near body."
                    )

                product_was_near_body = True
                last_near_body_time = current_time

        else:
            near_body_start_time = None

        # =======================================
        # STATE MACHINE
        # =======================================

        # ---------------------------------------
        # WAITING
        # ---------------------------------------

        if product_state == "WAITING":
            shelf_label = "WAITING FOR PRODUCT"

            # No concealment evidence should
            # survive while waiting.
            product_was_near_body = False
            near_body_start_time = None
            last_near_body_time = None

            if bottle_in_shelf:
                if shelf_entry_time is None:
                    shelf_entry_time = current_time

                if (
                    current_time - shelf_entry_time
                    >= shelf_confirmation_time
                ):
                    product_state = "ON_SHELF"
                    shelf_entry_time = None

                    print(
                        "STATE: Product confirmed on shelf."
                    )

            else:
                shelf_entry_time = None

        # ---------------------------------------
        # ON SHELF
        # ---------------------------------------

        elif product_state == "ON_SHELF":
            shelf_color = (0, 255, 0)
            shelf_label = "PRODUCT READY"

            # Clear old event evidence.
            product_was_near_body = False
            near_body_start_time = None
            last_near_body_time = None

            if hand_in_shelf:
                product_state = "INTERACTION"
                interaction_time = current_time

                print(
                    "STATE: Hand-product interaction."
                )

        # ---------------------------------------
        # INTERACTION
        # ---------------------------------------

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
                pickup_time = current_time

                # New event = clean evidence.
                product_was_near_body = False
                near_body_start_time = None
                last_near_body_time = None

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

        # ---------------------------------------
        # PICKED UP
        # ---------------------------------------

        elif product_state == "PICKED_UP":
            shelf_color = (0, 165, 255)
            shelf_label = "PICKUP CONFIRMED"

            if bottle_in_shelf:
                if shelf_entry_time is None:
                    shelf_entry_time = current_time

                if (
                    current_time - shelf_entry_time
                    >= shelf_confirmation_time
                ):
                    product_state = "ON_SHELF"

                    shelf_entry_time = None
                    pickup_time = None

                    product_was_near_body = False
                    near_body_start_time = None
                    last_near_body_time = None

                    print(
                        "STATE: Product returned to shelf."
                    )

            else:
                shelf_entry_time = None

            if (
                bottle_detected
                and not bottle_in_shelf
                and pickup_time is not None
            ):
                if (
                    current_time - pickup_time
                    >= carry_confirmation_time
                ):
                    product_state = "CARRIED"

                    print(
                        "STATE: Product is being carried."
                    )

        # ---------------------------------------
        # CARRIED
        # ---------------------------------------

        elif product_state == "CARRIED":
            shelf_color = (0, 165, 255)
            shelf_label = "PRODUCT CARRIED"

            # -----------------------------------
            # Returned to shelf
            # -----------------------------------

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
                    pickup_time = None

                    product_was_near_body = False
                    near_body_start_time = None
                    last_near_body_time = None

                    print(
                        "STATE: Product returned to shelf."
                    )

            else:
                shelf_entry_time = None

            # -----------------------------------
            # Product disappeared
            # -----------------------------------

            if (
                not bottle_detected
                and last_product_seen_time is not None
            ):
                missing_time = (
                    current_time
                    - last_product_seen_time
                )

                recent_body_evidence = False

                if last_near_body_time is not None:
                    evidence_age = (
                        current_time
                        - last_near_body_time
                    )

                    if (
                        evidence_age
                        <= (
                            disappearance_threshold
                            + near_body_memory
                        )
                    ):
                        recent_body_evidence = True

                if (
                    missing_time
                    >= disappearance_threshold
                    and product_was_near_body
                    and recent_body_evidence
                ):
                    product_state = "REVIEW"
                    review_alert_frames = 180

                    print(
                        "ALERT: Possible concealment - review footage."
                    )

        # ---------------------------------------
        # REVIEW
        # ---------------------------------------

        elif product_state == "REVIEW":
            shelf_color = (0, 0, 255)
            shelf_label = "REVIEW EVENT"

            if bottle_detected:
                cv2.putText(
                    frame,
                    "PRODUCT VISIBLE AGAIN",
                    (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 255, 255),
                    2
                )

        # =======================================
        # ALERT DISPLAY
        # =======================================

        if pickup_alert_frames > 0:
            cv2.putText(
                frame,
                "PRODUCT PICKUP DETECTED",
                (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.80,
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
                0.70,
                (0, 0, 255),
                3
            )

            review_alert_frames -= 1

        # =======================================
        # CURRENT STATE
        # =======================================

        cv2.putText(
            frame,
            f"STATE: {product_state}",
            (30, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # =======================================
        # LAST KNOWN PRODUCT POSITION
        # =======================================

        if (
            not bottle_detected
            and last_product_position is not None
            and product_state in [
                "PICKED_UP",
                "CARRIED"
            ]
        ):
            cv2.circle(
                frame,
                last_product_position,
                10,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "LAST SEEN",
                (
                    last_product_position[0] + 10,
                    last_product_position[1]
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )

        # =======================================
        # SHELF
        # =======================================

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
            0.60,
            shelf_color,
            2
        )

        cv2.imshow(
            "AI Security Monitor - Concealment Monitor",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("Camera stopped.")


if __name__ == "__main__":
    start_monitor()