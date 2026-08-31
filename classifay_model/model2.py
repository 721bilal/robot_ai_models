from ultralytics import YOLO
import cv2

# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_PATH = r"C:\Users\Bilal_weshah\PycharmProjects\robot_ai_models\classifay_model\runs\classify\runs\plant_health_cls-3\weights\best.pt"

# ============================================================
# CAMERA SETTINGS
# ============================================================

CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("ERROR: Could not open camera!")
    exit()

# Camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ============================================================
# FIXED PLANT REGIONS
# ============================================================

REGIONS = {

    # 1 - Top Left Plant
    "Plant 1": (20, 20, 500, 300),

    # 2 - Top Middle Small Plant
    "Plant 2": (550, 20, 800, 230),

    # 3 - Top Right Plant
    "Plant 3": (950, 0, 1280, 300),

    # 4 - Middle Left Small Plant
    "Plant 4": (200, 300, 400, 490),

    # 5 - Middle Center Plant
    "Plant 5": (520, 300, 900, 570),

    # 6 - Middle Right / Yellow Plant
    "Plant 6": (900, 280, 1280, 620),

    # 7 - Bottom Left Purple Plant
    "Plant 7": (100, 500, 550, 720),
}

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("SMART FARM - LIVE PLANT HEALTH CLASSIFICATION")
print("=" * 70)

print("\nLoading model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully!")

print("\nClasses:")
print(model.names)

print("\nStarting camera...")
print("Press Q to quit.")

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame!")
        break

    # ========================================================
    # PROCESS EVERY PLANT REGION
    # ========================================================

    for plant_name, (x1, y1, x2, y2) in REGIONS.items():

        # ----------------------------------------------------
        # CROP REGION
        # ----------------------------------------------------

        plant_crop = frame[y1:y2, x1:x2]

        # Safety check
        if plant_crop.size == 0:
            continue

        # ----------------------------------------------------
        # RUN CLASSIFICATION
        # ----------------------------------------------------

        results = model.predict(
            source=plant_crop,
            imgsz=224,
            verbose=False
        )

        result = results[0]

        # ----------------------------------------------------
        # GET PREDICTION
        # ----------------------------------------------------

        class_id = result.probs.top1

        confidence = float(result.probs.top1conf)

        class_name = model.names[class_id]

        confidence_percent = confidence * 100

        # ----------------------------------------------------
        # CONVERT CLASS TO STATUS
        # ----------------------------------------------------

        if class_name == "healthy_plant":

            status = "HEALTHY"

        else:

            status = "UNHEALTHY"

        # ----------------------------------------------------
        # DISPLAY LABEL
        # ----------------------------------------------------

        label = (
            f"{plant_name}: "
            f"{status} "
            f"{confidence_percent:.1f}%"
        )

        # ----------------------------------------------------
        # DRAW REGION
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # DRAW LABEL BACKGROUND
        # ----------------------------------------------------

        text_y = max(y1 - 10, 25)

        cv2.putText(
            frame,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    # ========================================================
    # DISPLAY CAMERA
    # ========================================================

    cv2.imshow(
        "Smart Farm - Live Plant Health",
        frame
    )

    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("\nCamera stopped.")