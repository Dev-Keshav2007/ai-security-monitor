import cv2
from collections import deque
from ultralytics import YOLOWorld


# ---------------------------------------------------------
# AI SECURITY MONITOR
# Automatic Merchandise Zone Discovery - Prototype V3
# ---------------------------------------------------------

print("Loading automatic merchandise-zone model...")

model = YOLOWorld("yolov8s-worldv2.pt")


# ---------------------------------------------------------
# Store concepts
# ---------------------------------------------------------

model.set_classes([
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

MIN_CONFIDENCE = 0.15

# Ignore tiny detections that are unlikely to represent
# meaningful merchandise fixtures.
MIN_ZONE_AREA = 5000

# Number of recent frames used to help stabilize detections.
HISTORY_LENGTH = 5

# Two detections must overlap reasonably well before they
# are considered the same merchandise fixture.
MATCH_IOU_THRESHOLD = 0.20

# If one box contains a large portion of another box,
# consider them detections of the same fixture.
CONTAINMENT_THRESHOLD = 0.65


zone_history = deque(maxlen=HISTORY_LENGTH)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def box_area(box):
    x1, y1, x2, y2 = box

    return max(0, x2 - x1) * max(0, y2 - y1)


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


def calculate_iou(box_a, box_b):

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


def calculate_containment(box_a, box_b):

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
    """
    Decide whether two boxes are probably detections
    of the SAME physical shelf/display.

    Unlike V2, proximity by itself is NOT enough.
    """

    iou = calculate_iou(
        box_a,
        box_b
    )

    containment = calculate_containment(
        box_a,
        box_b
    )

    if iou >= MATCH_IOU_THRESHOLD:
        return True

    if containment >= CONTAINMENT_THRESHOLD:
        return True

    return False


def merge_two_boxes(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    return (
        min(ax1, bx1),
        min(ay1, by1),
        max(ax2, bx2),
        max(ay2, by2)
    )


def remove_duplicate_detections(detections):
    """
    Merge detections only when they strongly appear
    to describe the same physical fixture.
    """

    final_boxes = []

    # Process larger detections first.
    detections = sorted(
        detections,
        key=box_area,
        reverse=True
    )

    for candidate in detections:

        matched = False

        for index, existing in enumerate(final_boxes):

            if same_fixture(
                candidate,
                existing
            ):

                final_boxes[index] = merge_two_boxes(
                    candidate,
                    existing
                )

                matched = True
                break

        if not matched:
            final_boxes.append(candidate)

    return final_boxes


def find_historical_match(
    current_box,
    previous_boxes
):

    best_match = None
    best_score = 0.0

    for previous_box in previous_boxes:

        score = calculate_iou(
            current_box,
            previous_box
        )

        if score > best_score:
            best_score = score
            best_match = previous_box

    if best_score >= MATCH_IOU_THRESHOLD:
        return best_match

    return None


def stabilize_box(
    current_box,
    previous_box
):
    """
    Smooth movement between frames so merchandise
    zones do not jump around as much.
    """

    if previous_box is None:
        return current_box

    cx1, cy1, cx2, cy2 = current_box
    px1, py1, px2, py2 = previous_box

    current_weight = 0.65
    previous_weight = 0.35

    return (
        int(
            cx1 * current_weight
            + px1 * previous_weight
        ),
        int(
            cy1 * current_weight
            + py1 * previous_weight
        ),
        int(
            cx2 * current_weight
            + px2 * previous_weight
        ),
        int(
            cy2 * current_weight
            + py2 * previous_weight
        )
    )


def stabilize_zones(
    current_zones,
    previous_zones
):

    stabilized = []

    for current_zone in current_zones:

        previous_match = find_historical_match(
            current_zone,
            previous_zones
        )

        stabilized_zone = stabilize_box(
            current_zone,
            previous_match
        )

        stabilized.append(
            stabilized_zone
        )

    return stabilized


# ---------------------------------------------------------
# Camera
# ---------------------------------------------------------

print("Opening camera...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    raise SystemExit


print()
print("Automatic merchandise-zone discovery V3 started.")
print()
print("GREEN = raw AI shelf/display detection")
print("BLUE  = final individual merchandise zone")
print()
print("Separate shelves should remain separate.")
print()
print("Press Q to stop.")
print()


previous_zones = []


# ---------------------------------------------------------
# Main camera loop
# ---------------------------------------------------------

while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break


    # -----------------------------------------------------
    # Open-vocabulary AI detection
    # -----------------------------------------------------

    results = model.predict(
        frame,
        conf=MIN_CONFIDENCE,
        verbose=False
    )


    raw_boxes = []


    # -----------------------------------------------------
    # Collect raw detections
    # -----------------------------------------------------

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            if confidence < MIN_CONFIDENCE:
                continue


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


            raw_boxes.append(
                candidate
            )


            # Raw AI detection = GREEN
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


    # -----------------------------------------------------
    # Remove duplicate detections
    # -----------------------------------------------------

    individual_zones = remove_duplicate_detections(
        raw_boxes
    )


    # -----------------------------------------------------
    # Stabilize zones
    # -----------------------------------------------------

    stabilized_zones = stabilize_zones(
        individual_zones,
        previous_zones
    )


    previous_zones = stabilized_zones

    zone_history.append(
        stabilized_zones
    )


    # -----------------------------------------------------
    # Draw final zones
    # -----------------------------------------------------

    for zone_number, zone in enumerate(
        stabilized_zones,
        start=1
    ):

        x1, y1, x2, y2 = zone


        # Final merchandise zone = BLUE
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            4
        )


        cv2.putText(
            frame,
            f"MERCHANDISE ZONE {zone_number}",
            (
                x1,
                max(y1 - 10, 25)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 0, 0),
            2
        )


    # -----------------------------------------------------
    # Screen information
    # -----------------------------------------------------

    cv2.putText(
        frame,
        "AUTOMATIC MERCHANDISE DISCOVERY V3",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"AI detections: {len(raw_boxes)}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Individual zones: {len(stabilized_zones)}",
        (20, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "GREEN = AI | BLUE = merchandise zone",
        (20, 119),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )


    # -----------------------------------------------------
    # Show camera
    # -----------------------------------------------------

    cv2.imshow(
        "AI Security - Merchandise Discovery V3",
        frame
    )


    # -----------------------------------------------------
    # Quit
    # -----------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()

print("Merchandise-zone discovery stopped.")