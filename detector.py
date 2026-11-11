
"""Bottle detection using custom classification and YOLO models."""

import cv2
import torch
import json
import time
from high_model.pipeline import run_pipeline

from pathlib import Path
from datetime import datetime
from ultralytics import YOLO, YOLOWorld




class BottleDetector:
    """Detect bottles using custom and YOLO-based models."""

    def __init__(
        self,
        mode="high",
        confidence=None,
        save=False,
        save_dir="bottle_results",
        save_format="jpg",
        save_metadata=True,
        show=False
    ):
        """Initialize the selected detection model."""

        self.mode = mode.lower()
        self.save = save
        self.save_dir = Path(save_dir)
        self.save_format = save_format.lower()
        self.save_metadata = save_metadata
        self.show = show

        self.device = (
            0 if torch.cuda.is_available() else -1
        )

        if self.save_format not in ("jpg", "png"):
            raise ValueError(
                "save_format must be jpg or png"
            )

        if self.mode == "high":

            self.confidence = (
                confidence if confidence is not None else 0.25
            )

            print("[MODEL] Loading HIGH classification pipeline")

            self.high_pipeline = run_pipeline

        elif self.mode == "medium":

            self.confidence = (
                confidence if confidence is not None else 0.15
            )

            print("[MODEL] Loading YOLO-World")

            self.model = YOLOWorld(
                "yolov8s-worldv2.pt"
            )

            self.model.set_classes([
                "bottle",
                "thermos flask"
            ])

        elif self.mode == "small":

            self.confidence = (
                confidence if confidence is not None else 0.25
            )

            print("[MODEL] Loading YOLO26 Nano")

            self.model = YOLO("yolo26n.pt")

            self.bottle_ids = [
                class_id
                for class_id, name in self.model.names.items()
                if name.lower() == "bottle"
            ]

            if not self.bottle_ids:
                raise ValueError(
                    "Bottle class not found"
                )

        else:
            raise ValueError(
                "Mode must be high, medium, or small"
            )

        if self.save:
            self.save_dir.mkdir(
                parents=True,
                exist_ok=True
            )

        print(
            f"[MODEL] {self.mode.upper()} loaded successfully"
        )

        print(
            f"[MODEL] Confidence: {self.confidence}"
        )

        print(
            f"[MODEL] Device: "
            f"{'GPU' if self.device == 0 else 'CPU'}"
        )

        print(
            f"[MODEL] Save enabled: {self.save}"
        )

    def _save_result(
        self,
        annotated,
        detections,
        result,
        image_name=None
    ):
        """Save annotated images and detection metadata."""

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        filename = (
            Path(image_name).stem
            if image_name else "bottle"
        )

        output_name = f"{filename}_{timestamp}"

        image_path = (
            self.save_dir
            / f"{output_name}.{self.save_format}"
        )

        if not cv2.imwrite(
            str(image_path),
            annotated
        ):
            raise IOError(
                f"Cannot save image: {image_path}"
            )

        result["saved_image"] = str(image_path)

        if self.save_metadata:

            metadata_path = (
                self.save_dir
                / f"{output_name}.json"
            )

            metadata = {
                "image": image_name,
                "timestamp": datetime.now().isoformat(),
                "mode": self.mode,
                "confidence_threshold": self.confidence,
                "device": (
                    "GPU" if self.device == 0 else "CPU"
                ),
                "detected": result["detected"],
                "count": result["count"],
                "status": result["status"],
                "arduino_value": result["arduino_value"],
                "detections": detections
            }

            with metadata_path.open(
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    metadata,
                    file,
                    indent=4
                )

            result["saved_metadata"] = str(
                metadata_path
            )

    def detect(
        self,
        image,
        image_name=None,
        save=None,
        show=None
    ):
        """Detect bottles and return Arduino-compatible results."""

        if image is None:
            raise ValueError(
                "Input image cannot be None"
            )

        if not hasattr(image, "shape") or len(image.shape) != 3:
            raise ValueError(
                "Expected an OpenCV color image"
            )

        detections = []
        annotated = image.copy()

        should_save = (
            self.save if save is None else save
        )

        should_show = (
            self.show if show is None else show
        )

        print("\n" + "=" * 60)
        print("[DETECTION] Starting inference")
        print(f"[DETECTION] Mode: {self.mode.upper()}")
        print(f"[DETECTION] Image shape: {image.shape}")
        print(f"[DETECTION] Threshold: {self.confidence}")

        start_time = time.perf_counter()

        if self.mode == "high":

            annotated, _, pipeline_detections = self.high_pipeline(image)

            for item in pipeline_detections:

                detections.append({
                    "label": "bottle",

                    "confidence": round(
                        float(
                            item.get("confidence", 0.0)
                        ),
                        3
                    ),

                    "bbox": [
                        int(value)
                        for value in item["box"]
                    ],

                    "deformity": item.get(
                        "deformity", ""
                    ),

                    "capacity": item.get(
                        "capacity", ""
                    ),

                    "fullness": item.get(
                        "fullness", ""
                    )
                })

        elif self.mode == "medium":

            results = self.model.predict(
                source=image,
                imgsz=640,
                conf=self.confidence,
                iou=0.45,
                agnostic_nms=True,
                max_det=10,
                device=self.device,
                augment=False,
                verbose=False
            )

            for box in results[0].boxes:

                class_id = int(box.cls[0])

                detections.append({
                    "label": self.model.names[class_id],

                    "confidence": round(
                        float(box.conf[0]),
                        3
                    ),

                    "bbox": [
                        int(x)
                        for x in box.xyxy[0].tolist()
                    ]
                })

        elif self.mode == "small":

            results = self.model.predict(
                source=image,
                classes=self.bottle_ids,
                conf=self.confidence,
                imgsz=640,
                device=self.device,
                verbose=False
            )

            for box in results[0].boxes:

                detections.append({
                    "label": "bottle",

                    "confidence": round(
                        float(box.conf[0]),
                        3
                    ),

                    "bbox": [
                        int(x)
                        for x in box.xyxy[0].tolist()
                    ]
                })

        if self.mode != "high":

            for detection in detections:

                x1, y1, x2, y2 = detection["bbox"]

                label = detection["label"]
                confidence = detection["confidence"]

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    annotated,
                    f"{label}: {confidence:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        elapsed = (
            time.perf_counter() - start_time
        )

        bottle_found = len(detections) > 0

        status = (
            "BOTTLE DETECTED"
            if bottle_found
            else "NO BOTTLE"
        )

        response = {
            "detected": bottle_found,
            "count": len(detections),
            "status": status,
            "arduino_value": int(bottle_found),
            "detections": detections,
            "image": annotated,
            "saved_image": None,
            "saved_metadata": None
        }

        if should_save:

            self.save_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            self._save_result(
                annotated=annotated,
                detections=detections,
                result=response,
                image_name=image_name
            )

        print("\n[FINAL RESULT]")
        print(f"Status: {status}")
        print(f"Detected: {bottle_found}")
        print(f"Count: {len(detections)}")
        print(
            f"Arduino value: {response['arduino_value']}"
        )
        print(f"Inference time: {elapsed:.3f}s")
        print(f"Saved image: {response['saved_image']}")

        for detection in detections:
            print(f"[OBJECT] {detection}")

        print("=" * 60)

        if should_show:

            cv2.imshow(
                "Bottle Detection",
                annotated
            )

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return response
        
    