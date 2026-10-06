import cv2
from ultralytics import YOLOWorld


# ---------------------------------------------------------
# AI SECURITY MONITOR
# Generalized Merchandise Detection - Prototype V1
# ---------------------------------------------------------

print("Loading generalized merchandise detector...")

model = YOLOWorld("yolov8s-worldv2.pt")


# ---------------------------------------------------------
# General concepts instead of individual products/SKUs
# ---------------------------------------------------------

model.set_classes([
    "retail product",
    "store product",
    "merchandise",
    "packaged product",
    "drink",
    "snack",
    "box",
    "bottle",
    "can",
    "package"
])


MIN_CONFIDENCE = 0.12
MIN_OBJECT_AREA = 500


# ---------------------------------------------------------
# Camera
# ---------------------------------------------------------

print("Opening camera...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    raise SystemExit


print()
print("Generalized merchandise detection started.")
print()
print("Try different products:")
print("- drinks")
print("- snack bags")
print("- boxes")
print("- cans")
print("- packages")
print("- other store merchandise")
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


    # -----------------------------------------------------
    # Generalized AI detection
    # -----------------------------------------------------

    results = model.predict(
        frame,
        conf=MIN_CONFIDENCE,
        verbose=False
    )


    detected_products = 0


    # -----------------------------------------------------
    # Process detections
    # -----------------------------------------------------

    for result in results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            class_name = model.names[class_id]


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )


            width = x2 - x1
            height = y2 - y1

            area = width * height


            if area < MIN_OBJECT_AREA:
                continue


            detected_products += 1


            # -------------------------------------------------
            # Product bounding box
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )


            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 8, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )


            # Center point
            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            cv2.circle(
                frame,
                (center_x, center_y),
                4,
                (0, 255, 0),
                -1
            )


    # -----------------------------------------------------
    # Display information
    # -----------------------------------------------------

    cv2.putText(
        frame,
        "GENERALIZED MERCHANDISE DETECTION",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Products detected: {detected_products}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "No store-specific training",
        (20, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Press Q to stop",
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
        "AI Security - General Product Detection",
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

print("Generalized merchandise detection stopped.")