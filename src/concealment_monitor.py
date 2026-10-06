import cv2
import time
import math
import os
from datetime import datetime
from collections import deque
from ultralytics import YOLO

from telegram_alert import send_message, send_photo, send_video


def point_inside_zone(point, zone):
    x, y = point
    x1, y1, x2, y2 = zone
    return x1 <= x <= x2 and y1 <= y <= y2


def calculate_distance(point1, point2):
    return math.sqrt(
        (point1[0] - point2[0]) ** 2
        + (point1[1] - point2[1]) ** 2
    )


def create_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_evidence_image(frame, timestamp):
    """Save one annotated evidence frame."""

    evidence_directory = "evidence/images"
    os.makedirs(evidence_directory, exist_ok=True)

    filename = f"possible_concealment_{timestamp}.jpg"
    filepath = os.path.join(evidence_directory, filename)

    success = cv2.imwrite(filepath, frame)

    if success:
        print(f"EVIDENCE IMAGE SAVED: {filepath}")
        return filepath

    print("ERROR: Evidence image could not be saved.")
    return None


def save_video_clip(frames, frame_size, fps, timestamp):
    """Save buffered frames as an evidence video."""

    evidence_directory = "evidence/clips"
    os.makedirs(evidence_directory, exist_ok=True)

    filename = f"possible_concealment_{timestamp}.mp4"
    filepath = os.path.join(evidence_directory, filename)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        filepath,
        fourcc,
        fps,
        frame_size
    )

    if not writer.isOpened():
        print("ERROR: Evidence video could not be created.")
        return None

    for video_frame in frames:
        writer.write(video_frame)

    writer.release()

    print(f"EVIDENCE CLIP SAVED: {filepath}")
    return filepath


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
    # Evidence tracking
    # ---------------------------------------

    evidence_saved_for_event = False
    saved_evidence_path = None

    # ---------------------------------------
    # Video evidence settings
    # ---------------------------------------

    pre_event_seconds = 8.0
    post_event_seconds = 3.0

    # Stores:
    # (timestamp, frame)
    video_buffer = deque()

    recording_event = False
    event_frames = []
    event_end_time = None
    event_timestamp = None

    # ---------------------------------------
    # Detection timing settings
    # ---------------------------------------

    shelf_confirmation_time = 1.0
    pickup_window = 3.0
    carry_confirmation_time = 0.8
    near_body_confirmation_time = 0.7

    # Ignore short YOLO detection failures.
    disappearance_threshold = 3.0
    near_body_memory = 2.0

    pickup_alert_frames = 0
    review_alert_frames = 0

    # ---------------------------------------
    # Camera
    # ---------------------------------------

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not access camera.")
        return

    print("Camera started.")
    print("Concealment monitoring active.")
    print("Evidence image capture active.")
    print("Rolling video buffer active.")
    print("Telegram alerts active.")
    print("")
    print("Near-body evidence is only valid after pickup.")
    print("")
    print("Press Q to stop.")

    # Used to estimate the actual processed FPS.
    processing_times = deque(maxlen=30)

    while True:
        loop_start_time = time.time()

        success, frame = camera.read()

        if not success:
            print("Error: Could not read camera frame.")
            break

        current_time = time.time()

        # Keep a raw copy before annotations are added.
        raw_frame = frame.copy()

        # =======================================
        # ROLLING PRE-EVENT BUFFER
        # =======================================

        video_buffer.append(
            (current_time, raw_frame.copy())
        )

        cutoff_time = current_time - pre_event_seconds

        while (
            video_buffer
            and video_buffer[0][0] < cutoff_time
        ):
            video_buffer.popleft()

        # If an event is already recording,
        # continue collecting post-event footage.
        if recording_event:
            event_frames.append(raw_frame.copy())

            if current_time >= event_end_time:
                saved_video_path = None

                if event_frames:
                    height, width = event_frames[0].shape[:2]

                    frame_size = (
                        width,
                        height
                    )

                    if processing_times:
                        average_processing_time = (
                            sum(processing_times)
                            / len(processing_times)
                        )

                        if average_processing_time > 0:
                            estimated_fps = (
                                1.0
                                / average_processing_time
                            )
                        else:
                            estimated_fps = 10.0
                    else:
                        estimated_fps = 10.0

                    # Keep the saved video FPS
                    # within a reasonable range.
                    estimated_fps = max(
                        1.0,
                        min(
                            estimated_fps,
                            30.0
                        )
                    )

                    saved_video_path = save_video_clip(
                        event_frames,
                        frame_size,
                        estimated_fps,
                        event_timestamp
                    )

                # Send the finished video only after
                # the MP4 has been written to disk.
                if saved_video_path:
                    send_video(
                        saved_video_path,
                        caption=(
                            "🎥 AI SECURITY EVIDENCE\n\n"
                            "Possible concealment event.\n"
                            "Review the attached evidence clip."
                        )
                    )

                recording_event = False
                event_frames = []
                event_end_time = None
                event_timestamp = None

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

        shelf_x1, shelf_y1, shelf_x2, shelf_y2 = (
            shelf_zone
        )

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

                torso_points = []

                body_points = [
                    left_shoulder,
                    right_shoulder,
                    left_hip,
                    right_hip
                ]

                for point in body_points:
                    px = int(point[0])
                    py = int(point[1])

                    if px > 0 and py > 0:
                        torso_points.append(
                            (px, py)
                        )

                if len(torso_points) >= 2:
                    torso_center = (
                        sum(
                            p[0]
                            for p in torso_points
                        )
                        // len(torso_points),

                        sum(
                            p[1]
                            for p in torso_points
                        )
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

                wrists = [
                    left_wrist,
                    right_wrist
                ]

                for wrist in wrists:
                    wrist_x = int(wrist[0])
                    wrist_y = int(wrist[1])

                    if (
                        wrist_x <= 0
                        or wrist_y <= 0
                    ):
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
                        wrist_color = (
                            0,
                            0,
                            255
                        )
                    else:
                        wrist_color = (
                            0,
                            255,
                            255
                        )

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

        evidence_allowed = (
            product_state
            in [
                "PICKED_UP",
                "CARRIED"
            ]
        )

        if (
            evidence_allowed
            and product_currently_near_body
        ):
            if near_body_start_time is None:
                near_body_start_time = current_time

            near_body_duration = (
                current_time
                - near_body_start_time
            )

            if (
                near_body_duration
                >= near_body_confirmation_time
            ):
                if not product_was_near_body:
                    print(
                        "EVIDENCE: Product "
                        "confirmed near body."
                    )

                product_was_near_body = True
                last_near_body_time = current_time

        else:
            near_body_start_time = None

        # =======================================
        # STATE MACHINE
        # =======================================

        if product_state == "WAITING":
            shelf_label = "WAITING FOR PRODUCT"

            product_was_near_body = False
            near_body_start_time = None
            last_near_body_time = None

            if bottle_in_shelf:
                if shelf_entry_time is None:
                    shelf_entry_time = current_time

                if (
                    current_time
                    - shelf_entry_time
                    >= shelf_confirmation_time
                ):
                    product_state = "ON_SHELF"
                    shelf_entry_time = None

                    print(
                        "STATE: Product "
                        "confirmed on shelf."
                    )

            else:
                shelf_entry_time = None

        elif product_state == "ON_SHELF":
            shelf_color = (0, 255, 0)
            shelf_label = "PRODUCT READY"

            product_was_near_body = False
            near_body_start_time = None
            last_near_body_time = None

            evidence_saved_for_event = False
            saved_evidence_path = None

            if hand_in_shelf:
                product_state = "INTERACTION"
                interaction_time = current_time

                print(
                    "STATE: Hand-product "
                    "interaction."
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
                pickup_time = current_time

                product_was_near_body = False
                near_body_start_time = None
                last_near_body_time = None

                evidence_saved_for_event = False
                saved_evidence_path = None

                pickup_alert_frames = 100

                print(
                    "EVENT: Product pickup "
                    "detected."
                )

            elif interaction_age > pickup_window:
                product_state = "ON_SHELF"
                interaction_time = None

                print(
                    "STATE: Interaction expired."
                )

        elif product_state == "PICKED_UP":
            shelf_color = (0, 165, 255)
            shelf_label = "PICKUP CONFIRMED"

            if bottle_in_shelf:
                if shelf_entry_time is None:
                    shelf_entry_time = current_time

                if (
                    current_time
                    - shelf_entry_time
                    >= shelf_confirmation_time
                ):
                    product_state = "ON_SHELF"

                    shelf_entry_time = None
                    pickup_time = None

                    product_was_near_body = False
                    near_body_start_time = None
                    last_near_body_time = None

                    print(
                        "STATE: Product returned "
                        "to shelf."
                    )

            else:
                shelf_entry_time = None

            if (
                bottle_detected
                and not bottle_in_shelf
                and pickup_time is not None
            ):
                if (
                    current_time
                    - pickup_time
                    >= carry_confirmation_time
                ):
                    product_state = "CARRIED"

                    print(
                        "STATE: Product is "
                        "being carried."
                    )

        elif product_state == "CARRIED":
            shelf_color = (0, 165, 255)
            shelf_label = "PRODUCT CARRIED"

            if bottle_in_shelf:
                if shelf_entry_time is None:
                    shelf_entry_time = current_time

                if (
                    current_time
                    - shelf_entry_time
                    >= shelf_confirmation_time
                ):
                    product_state = "ON_SHELF"

                    shelf_entry_time = None
                    interaction_time = None
                    pickup_time = None

                    product_was_near_body = False
                    near_body_start_time = None
                    last_near_body_time = None

                    evidence_saved_for_event = False
                    saved_evidence_path = None

                    print(
                        "STATE: Product returned "
                        "to shelf."
                    )

            else:
                shelf_entry_time = None

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
                    missing_time >= disappearance_threshold
                    and product_was_near_body
                    and recent_body_evidence
                ):
                    product_state = "REVIEW"
                    review_alert_frames = 180

                    print(
                        "ALERT: Possible "
                        "concealment - "
                        "review footage."
                    )

                    # ---------------------------
                    # SAVE STILL IMAGE
                    # ---------------------------

                    if not evidence_saved_for_event:
                        event_timestamp = create_timestamp()

                        saved_evidence_path = (
                            save_evidence_image(
                                frame.copy(),
                                event_timestamp
                            )
                        )

                        if saved_evidence_path:
                            evidence_saved_for_event = True

                            # Send immediate Telegram warning.
                            send_message(
                                "⚠️ AI SECURITY ALERT\n\n"
                                "Possible concealment detected.\n"
                                "Review required."
                            )

                            # Send the evidence image immediately.
                            send_photo(
                                saved_evidence_path,
                                caption=(
                                    "📸 AI SECURITY EVIDENCE\n\n"
                                    "Possible concealment event.\n"
                                    "Review the attached image."
                                )
                            )

                        # -----------------------
                        # START EVENT CLIP
                        # -----------------------

                        event_frames = [
                            buffered_frame.copy()
                            for _, buffered_frame
                            in video_buffer
                        ]

                        recording_event = True

                        event_end_time = (
                            current_time
                            + post_event_seconds
                        )

                        print(
                            "VIDEO: Capturing "
                            "3 seconds of "
                            "post-event footage..."
                        )

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
            and product_state
            in [
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

        # =======================================
        # RECORDING INDICATOR
        # =======================================

        if recording_event:
            cv2.putText(
                frame,
                "SAVING EVENT CLIP...",
                (30, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "AI Security Monitor - Concealment Monitor",
            frame
        )

        processing_time = (
            time.time() - loop_start_time
        )

        if processing_time > 0:
            processing_times.append(
                processing_time
            )

        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("Camera stopped.")


if __name__ == "__main__":
    start_monitor()