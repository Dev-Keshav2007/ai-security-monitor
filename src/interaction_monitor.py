import cv2
import time
from ultralytics import YOLO, YOLOWorld


# ---------------------------------------------------------
# AI SECURITY MONITOR
# Person + Automatic Merchandise Interaction - V2
#
# Goal:
# Detect a person interacting with an automatically
# discovered merchandise zone while preventing duplicate
# events from pose/detection flicker.
# ---------------------------------------------------------


print("Loading AI models...")

shelf_model = YOLOWorld("yolov8s-worldv2.pt")
pose_model = YOLO("yolo11n-pose.pt")


# ---------------------------------------------------------
# Merchandise concepts
# ---------------------------------------------------------

shelf_model.set_classes([
    "retail shelf",
    "merchandise shelf",
    "store shelf",
    "display rack",
    "product display",
    "merchandise display",
    "store merchandise"
])


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

SHELF_CONFIDENCE = 0.15
POSE_CONFIDENCE = 0.45

MIN_ZONE_AREA = 5000

# Wrist must remain inside long enough to confirm
# a real interaction.
INTERACTION_CONFIRMATION_TIME = 0.35

# Once an interaction is confirmed, the hand must remain
# outside all merchandise zones for this long before the
# interaction session can end.
SESSION_END_DELAY = 1.50

# Small pose dropouts are ignored for this amount of time.
WRIST_MEMORY_TIME = 0.60


# ---------------------------------------------------------
# Interaction state
# ---------------------------------------------------------

interaction_candidate_start = None

interaction_session_active = False

last_wrist_in_zone_time = None

session_number = 0

active_person_number = None
active_zone_number = None

session_start_time = None


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def box_area(box):
    x1, y1, x2, y2 = box

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    return width * height


def intersection_area(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)

    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    return width * height


def box_iou(box_a, box_b):

    intersection = intersection_area(
        box_a,
        box_b
    )

    if intersection <= 0:
        return 0.0

    area_a = box_area(box_a)
    area_b = box_area(box_b)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def containment(box_a, box_b):

    intersection = intersection_area(
        box_a,
        box_b
    )

    smaller_area = min(
        box_area(box_a),
        box_area(box_b)
    )

    if smaller_area <= 0:
        return 0.0

    return intersection / smaller_area


def same_fixture(box_a, box_b):

    if box_iou(box_a, box_b) >= 0.20:
        return True

    if containment(box_a, box_b) >= 0.65:
        return True

    return False


def merge_boxes(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    return (
        min(ax1, bx1),
        min(ay1, by1),
        max(ax2, bx2),
        max(ay2, by2)
    )


def remove_duplicate_zones(boxes):

    final_zones = []

    boxes = sorted(
        boxes,
        key=box_area,
        reverse=True
    )

    for candidate in boxes:

        matched = False

        for index, existing in enumerate(
            final_zones
        ):

            if same_fixture(
                candidate,
                existing
            ):

                final_zones[index] = merge_boxes(
                    candidate,
                    existing
                )

                matched = True
                break

        if not matched:

            final_zones.append(
                candidate
            )

    return final_zones


def point_inside_box(point, box):

    x, y = point
    x1, y1, x2, y2 = box

    return (
        x1 <= x <= x2
        and
        y1 <= y <= y2
    )


def valid_keypoint(point):

    x, y = point

    return (
        x > 1
        and
        y > 1
    )


# ---------------------------------------------------------
# Camera
# ---------------------------------------------------------

print("Opening camera...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Could not open camera.")
    raise SystemExit


print()
print("Interaction Monitor V2 started.")
print()
print("BLUE   = merchandise zone")
print("YELLOW = person")
print("GREEN  = wrist outside merchandise")
print("RED    = wrist inside merchandise")
print()
print("V2 uses interaction-session locking.")
print("One continuous reach should create ONE event.")
print()
print("Press Q to stop.")
print()


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------

while True:

    success, frame = camera.read()

    if not success:

        print("ERROR: Could not read camera frame.")
        break


    current_time = time.time()


    # =====================================================
    # STEP 1
    # Automatically discover merchandise zones
    # =====================================================

    shelf_results = shelf_model.predict(
        frame,
        conf=SHELF_CONFIDENCE,
        verbose=False
    )


    raw_zones = []


    for result in shelf_results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )


            candidate = (
                x1,
                y1,
                x2,
                y2
            )


            if box_area(candidate) < MIN_ZONE_AREA:
                continue


            raw_zones.append(
                candidate
            )


    merchandise_zones = remove_duplicate_zones(
        raw_zones
    )


    # -----------------------------------------------------
    # Draw merchandise zones
    # -----------------------------------------------------

    for zone_index, zone in enumerate(
        merchandise_zones
    ):

        zone_number = zone_index + 1

        x1, y1, x2, y2 = zone


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            3
        )


        cv2.putText(
            frame,
            f"MERCHANDISE ZONE {zone_number}",
            (
                x1,
                max(y1 - 10, 25)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 0, 0),
            2
        )


    # =====================================================
    # STEP 2
    # Detect people and wrists
    # =====================================================

    pose_results = pose_model.predict(
        frame,
        conf=POSE_CONFIDENCE,
        verbose=False
    )


    detected_people = 0

    wrist_currently_in_zone = False

    current_person_number = None
    current_zone_number = None


    for result in pose_results:

        if result.boxes is None:
            continue

        if result.keypoints is None:
            continue


        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )


        keypoints = (
            result.keypoints.xy
            .cpu()
            .numpy()
        )


        number_of_people = min(
            len(boxes),
            len(keypoints)
        )


        for person_index in range(
            number_of_people
        ):

            detected_people += 1

            person_number = (
                person_index + 1
            )


            px1, py1, px2, py2 = map(
                int,
                boxes[person_index]
            )


            # ---------------------------------------------
            # Draw person
            # ---------------------------------------------

            cv2.rectangle(
                frame,
                (px1, py1),
                (px2, py2),
                (0, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"PERSON {person_number}",
                (
                    px1,
                    max(py1 - 10, 25)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )


            person_keypoints = (
                keypoints[person_index]
            )


            # COCO pose:
            # 9  = left wrist
            # 10 = right wrist
            wrist_indices = [
                9,
                10
            ]


            for wrist_index in wrist_indices:

                if wrist_index >= len(
                    person_keypoints
                ):
                    continue


                wrist = person_keypoints[
                    wrist_index
                ]


                wrist_point = (
                    int(wrist[0]),
                    int(wrist[1])
                )


                if not valid_keypoint(
                    wrist_point
                ):
                    continue


                wrist_zone_number = None


                # -----------------------------------------
                # Check every merchandise zone
                # -----------------------------------------

                for zone_index, zone in enumerate(
                    merchandise_zones
                ):

                    if point_inside_box(
                        wrist_point,
                        zone
                    ):

                        wrist_zone_number = (
                            zone_index + 1
                        )

                        break


                # -----------------------------------------
                # Wrist is interacting with merchandise
                # -----------------------------------------

                if wrist_zone_number is not None:

                    wrist_currently_in_zone = True

                    current_person_number = (
                        person_number
                    )

                    current_zone_number = (
                        wrist_zone_number
                    )


                    cv2.circle(
                        frame,
                        wrist_point,
                        9,
                        (0, 0, 255),
                        -1
                    )


                    cv2.putText(
                        frame,
                        (
                            f"INTERACTION ZONE "
                            f"{wrist_zone_number}"
                        ),
                        (
                            wrist_point[0] + 10,
                            wrist_point[1] - 10
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.43,
                        (0, 0, 255),
                        2
                    )


                # -----------------------------------------
                # Normal wrist
                # -----------------------------------------

                else:

                    cv2.circle(
                        frame,
                        wrist_point,
                        7,
                        (0, 255, 0),
                        -1
                    )


    # =====================================================
    # STEP 3
    # Interaction session logic
    # =====================================================

    if wrist_currently_in_zone:

        last_wrist_in_zone_time = (
            current_time
        )


        # -----------------------------------------
        # No session yet
        # -----------------------------------------

        if not interaction_session_active:

            if interaction_candidate_start is None:

                interaction_candidate_start = (
                    current_time
                )


            candidate_duration = (
                current_time
                - interaction_candidate_start
            )


            # -------------------------------------
            # Confirm new interaction session
            # -------------------------------------

            if (
                candidate_duration
                >= INTERACTION_CONFIRMATION_TIME
            ):

                interaction_session_active = True

                session_number += 1

                session_start_time = (
                    current_time
                )

                active_person_number = (
                    current_person_number
                )

                active_zone_number = (
                    current_zone_number
                )


                print()
                print(
                    "================================"
                )

                print(
                    f"INTERACTION SESSION "
                    f"{session_number} STARTED"
                )

                print(
                    f"Person: "
                    f"{active_person_number}"
                )

                print(
                    f"Merchandise Zone: "
                    f"{active_zone_number}"
                )

                print(
                    "EVENT: Confirmed "
                    "person-merchandise interaction."
                )

                print(
                    "================================"
                )

                print()


        # -----------------------------------------
        # Session already active
        # -----------------------------------------

        else:

            # Keep session alive.
            interaction_candidate_start = None


    # =====================================================
    # Wrist not currently detected in merchandise
    # =====================================================

    else:

        # -----------------------------------------
        # Candidate never became a real interaction
        # -----------------------------------------

        if not interaction_session_active:

            if interaction_candidate_start is not None:

                candidate_age = (
                    current_time
                    - interaction_candidate_start
                )


                if (
                    candidate_age
                    > WRIST_MEMORY_TIME
                ):

                    interaction_candidate_start = None


        # -----------------------------------------
        # Existing confirmed interaction
        # -----------------------------------------

        else:

            if last_wrist_in_zone_time is not None:

                time_outside_zone = (
                    current_time
                    - last_wrist_in_zone_time
                )


                # ---------------------------------
                # End session only after the wrist
                # has really been away long enough.
                # ---------------------------------

                if (
                    time_outside_zone
                    >= SESSION_END_DELAY
                ):

                    session_duration = (
                        current_time
                        - session_start_time
                    )


                    print()
                    print(
                        "================================"
                    )

                    print(
                        f"INTERACTION SESSION "
                        f"{session_number} ENDED"
                    )

                    print(
                        f"Duration: "
                        f"{session_duration:.1f} seconds"
                    )

                    print(
                        "================================"
                    )

                    print()


                    interaction_session_active = False

                    interaction_candidate_start = None

                    last_wrist_in_zone_time = None

                    active_person_number = None

                    active_zone_number = None

                    session_start_time = None


    # =====================================================
    # STEP 4
    # Status display
    # =====================================================

    if interaction_session_active:

        status_text = (
            f"SESSION {session_number} ACTIVE"
        )

        status_color = (
            0,
            0,
            255
        )


    elif interaction_candidate_start is not None:

        status_text = (
            "CHECKING INTERACTION..."
        )

        status_color = (
            0,
            165,
            255
        )


    else:

        status_text = (
            "MONITORING"
        )

        status_color = (
            0,
            255,
            0
        )


    cv2.putText(
        frame,
        "AI PERSON + MERCHANDISE MONITOR V2",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"People: {detected_people}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        (
            f"Merchandise zones: "
            f"{len(merchandise_zones)}"
        ),
        (20, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        status_text,
        (20, 122),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        status_color,
        2
    )


    # -----------------------------------------------------
    # Active session information
    # -----------------------------------------------------

    if interaction_session_active:

        cv2.putText(
            frame,
            (
                f"Person: "
                f"{active_person_number}"
            ),
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )


        cv2.putText(
            frame,
            (
                f"Origin Zone: "
                f"{active_zone_number}"
            ),
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )


    # =====================================================
    # Show camera
    # =====================================================

    cv2.imshow(
        "AI Security - Interaction Monitor V2",
        frame
    )


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()

print("Interaction Monitor V2 stopped.")