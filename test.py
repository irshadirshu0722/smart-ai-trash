
"""Batch-test bottle detection and classification using YOLO26m."""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_DIR = BASE_DIR / "bottle_test_results"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"

# Replace with your trained model path.
MODEL_PATH = BASE_DIR / "yolo26n.pt"

CONFIDENCE_THRESHOLD = 0.15
IMAGE_SIZE = 640

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}


# ============================================================
# MODEL
# ============================================================

def load_model() -> YOLO:
    """Load the trained YOLO26m model."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Place your trained best.pt inside the project folder."
        )

    print(f"Loading model: {MODEL_PATH.name}")

    model = YOLO(str(MODEL_PATH))

    print("Model loaded successfully.")
    print(f"Classes: {model.names}\n")

    return model


# ============================================================
# DETECTION
# ============================================================

def detect_bottles(model: YOLO, frame):
    """Detect and classify bottles."""

    start = time.perf_counter()

    results = model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=IMAGE_SIZE,
        verbose=False,
    )

    inference_time = time.perf_counter() - start

    result = results[0]
    detections = []

    for box in result.boxes:

        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        class_name = result.names[class_id]

        detections.append({
            "class": class_name,
            "confidence": confidence,
            "box": [x1, y1, x2, y2],
        })

    annotated = result.plot()

    return annotated, detections, inference_time


# ============================================================
# BATCH TESTING
# ============================================================

def main() -> None:

    if not ASSETS_DIR.is_dir():
        raise FileNotFoundError(
            f"Assets folder not found: {ASSETS_DIR}"
        )

    images = sorted(
        path for path in ASSETS_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print(f"No supported images found in: {ASSETS_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()

    report_path = OUTPUT_DIR / "predictions.csv"

    summary_rows = []

    total_detections = 0
    total_inference_seconds = 0.0
    processed_count = 0

    print("=" * 55)
    print("       YOLO26m BOTTLE CLASSIFICATION TEST")
    print("=" * 55)

    print(f"Images: {len(images)}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Output: {OUTPUT_DIR}\n")

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for index, image_path in enumerate(images, start=1):

        frame = cv2.imread(str(image_path))

        if frame is None:

            print(
                f"[{index}/{len(images)}] "
                f"SKIPPED: {image_path.name}"
            )

            summary_rows.append({
                "image": image_path.name,
                "bottle_number": "",
                "class": "",
                "confidence": "",
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
                "status": "unreadable_image",
                "inference_seconds": "",
            })

            continue

        try:

            annotated, detections, elapsed = detect_bottles(
                model,
                frame
            )

        except Exception as error:

            print(f"ERROR: {image_path.name}: {error}")

            summary_rows.append({
                "image": image_path.name,
                "bottle_number": "",
                "class": "",
                "confidence": "",
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
                "status": "inference_error",
                "inference_seconds": "",
            })

            continue

        processed_count += 1
        total_inference_seconds += elapsed
        total_detections += len(detections)

        # Save annotated image
        output_path = (
            ANNOTATED_DIR /
            f"{image_path.stem}_annotated.jpg"
        )

        if not cv2.imwrite(str(output_path), annotated):
            print(f"WARNING: Could not save {output_path}")

        print(f"\n[{index}/{len(images)}] {image_path.name}")
        print(f"  Detections: {len(detections)}")
        print(f"  Inference:  {elapsed:.4f}s")

        # ----------------------------------------------------
        # No detections
        # ----------------------------------------------------

        if not detections:

            print("  No bottles detected.")

            summary_rows.append({
                "image": image_path.name,
                "bottle_number": "",
                "class": "",
                "confidence": "",
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
                "status": "no_detection",
                "inference_seconds": f"{elapsed:.4f}",
            })

        # ----------------------------------------------------
        # Process detections
        # ----------------------------------------------------

        else:

            for bottle_index, detection in enumerate(
                detections,
                start=1
            ):

                x1, y1, x2, y2 = detection["box"]

                class_name = detection["class"]
                confidence = detection["confidence"]

                print(f"\n  Bottle {bottle_index}")
                print(f"    Class:      {class_name}")
                print(f"    Confidence: {confidence:.2%}")
                print(f"    Bounding box: {(x1, y1, x2, y2)}")

                summary_rows.append({
                    "image": image_path.name,
                    "bottle_number": bottle_index,
                    "class": class_name,
                    "confidence": f"{confidence:.4f}",
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "status": "detected",
                    "inference_seconds": f"{elapsed:.4f}",
                })

        print(f"\n  Saved: {output_path.name}")

    # ========================================================
    # CSV REPORT
    # ========================================================

    fieldnames = [
        "image",
        "bottle_number",
        "class",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "status",
        "inference_seconds",
    ]

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    # ========================================================
    # SUMMARY
    # ========================================================

    average_inference = (
        total_inference_seconds / processed_count
        if processed_count else 0.0
    )

    print("\n")
    print("=" * 55)
    print("                 FINAL SUMMARY")
    print("=" * 55)

    print(f"Images found:           {len(images)}")
    print(f"Images processed:       {processed_count}")
    print(f"Total detections:       {total_detections}")
    print(f"Total inference time:   {total_inference_seconds:.3f}s")
    print(f"Average inference:      {average_inference:.3f}s/image")

    print(f"\nAnnotated images: {ANNOTATED_DIR}")
    print(f"CSV report:       {report_path}")

    print("=" * 55)


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(130)
