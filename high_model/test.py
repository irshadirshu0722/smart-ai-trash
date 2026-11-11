"""Batch-test the Bottle-Classification-System pipeline on images in assets/."""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2

from pipeline import run_pipeline


BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_DIR = BASE_DIR / "bottle_test_results"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main() -> None:
    if not ASSETS_DIR.is_dir():
        raise FileNotFoundError(
            f"Assets folder not found: {ASSETS_DIR}\n"
            "Create an 'assets' folder next to test.py and put test images inside it."
        )

    images = sorted(
        path for path in ASSETS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print(f"No supported images found in: {ASSETS_DIR}")
        return

    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = OUTPUT_DIR / "predictions.csv"
    summary_rows: list[dict[str, object]] = []
    total_bottles = 0
    total_inference_seconds = 0.0

    print("\n========== BOTTLE CLASSIFICATION BATCH TEST ==========")
    print(f"Images: {len(images)}")
    print(f"Output: {OUTPUT_DIR}\n")

    for index, image_path in enumerate(images, start=1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[{index}/{len(images)}] SKIPPED: Could not read {image_path.name}")
            summary_rows.append({
                "image": image_path.name,
                "bottle_number": "",
                "condition": "",
                "capacity": "",
                "fullness": "",
                "confidence": "",
                "x1": "", "y1": "", "x2": "", "y2": "",
                "status": "unreadable_image",
                "inference_seconds": "",
            })
            continue

        start = time.perf_counter()
        annotated, result_text, detections = run_pipeline(frame)
        elapsed = time.perf_counter() - start
        total_inference_seconds += elapsed
        total_bottles += len(detections)

        output_path = ANNOTATED_DIR / f"{image_path.stem}_annotated.jpg"
        if not cv2.imwrite(str(output_path), annotated):
            print(f"WARNING: Could not save {output_path}")

        print(f"[{index}/{len(images)}] {image_path.name}")
        print(f"  Bottles detected: {len(detections)}")
        print(f"  Inference time:   {elapsed:.3f}s")

        if not detections:
            print("  No bottle attributes detected.")
            summary_rows.append({
                "image": image_path.name,
                "bottle_number": "",
                "condition": "",
                "capacity": "",
                "fullness": "",
                "confidence": "",
                "x1": "", "y1": "", "x2": "", "y2": "",
                "status": "no_detection",
                "inference_seconds": f"{elapsed:.4f}",
            })
        else:
            for bottle_index, detection in enumerate(detections, start=1):
                box = detection.get("box", ["", "", "", ""])
                condition = detection.get("deformity") or detection.get("defect") or ""
                capacity = detection.get("capacity") or detection.get("size") or ""
                fullness = detection.get("fullness") or detection.get("full") or ""
                confidence = detection.get("confidence", detection.get("conf", ""))

                print(f"  Bottle {bottle_index}:")
                print(f"    Condition: {condition or 'Unknown'}")
                print(f"    Capacity:  {capacity or 'Unknown'}")
                print(f"    Fullness:  {fullness or 'Unknown'}")
                if confidence != "":
                    print(f"    Confidence: {float(confidence):.2%}")

                summary_rows.append({
                    "image": image_path.name,
                    "bottle_number": bottle_index,
                    "condition": condition,
                    "capacity": capacity,
                    "fullness": fullness,
                    "confidence": confidence,
                    "x1": box[0], "y1": box[1],
                    "x2": box[2], "y2": box[3],
                    "status": "detected",
                    "inference_seconds": f"{elapsed:.4f}",
                })

        print(f"  Saved: {output_path.name}\n")

    fieldnames = [
        "image", "bottle_number", "condition", "capacity", "fullness",
        "confidence", "x1", "y1", "x2", "y2", "status", "inference_seconds",
    ]
    with report_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    processed_count = sum(1 for row in summary_rows if row["status"] != "unreadable_image")
    average_seconds = total_inference_seconds / processed_count if processed_count else 0.0

    print("========== SUMMARY ==========")
    print(f"Images found:           {len(images)}")
    print(f"Images processed:       {processed_count}")
    print(f"Total bottles detected: {total_bottles}")
    print(f"Average inference time: {average_seconds:.3f}s/image")
    print(f"Annotated images:       {ANNOTATED_DIR}")
    print(f"CSV report:             {report_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(130)
