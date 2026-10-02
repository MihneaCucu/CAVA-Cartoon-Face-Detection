"""
Face detection using YOLOv8-Face - State-of-the-Art solution for Task 1
Uses YOLOv8 model specifically trained on WIDERFace dataset.
Target performance: >80% AP
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import time
import torch

# Fix path to support imports from utils
import sys
current_dir = Path(__file__).parent
solutie_fisiere_root = current_dir.parent.parent.parent
sys.path.insert(0, str(solutie_fisiere_root / 'utils'))

from paths import PROCESSED_DATA_DIR
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("WARNING: ultralytics not installed. Install with: pip install ultralytics")


class YOLOFaceDetector:
    """Detector of faces based on YOLOv8 antrenat pe WIDERFace."""

    def __init__(self, model_path=None, conf_threshold=0.25, iou_threshold=0.45):

        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        print(f"\n{'='*60}")
        print("YOLOv8-FACE MODEL LOADING")
        print(f"{'='*60}")
        print(f"Confidence threshold: {conf_threshold}")
        print(f"IoU threshold: {iou_threshold}")


        if model_path:
            mp = Path(model_path)
            if mp.exists():
                print(f"YOLOv8-FACE MODEL LOADING: {mp}")
                self.model = YOLO(str(mp))
            else:
                raise FileNotFoundError(f"CRITICAL ERROR: Custom model not found at {mp}")
        else:
            print("YOLOv8n (standard) MODEL LOADING")
            print("NOTA: Pentru >80% AP, recomandăm model antrenat pe WIDERFace")
            self.model = YOLO('yolov8n.pt')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {self.device}")

        print("OK: Model loaded successfully!")
        print(f"{'='*60}\n")

    def detect_faces(self, image_path, visualize=False):
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return []

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print(f"\n{'='*60}")
        print(f"YOLOv8-FACE DETECTION: {Path(image_path).name}")
        print(f"{'='*60}")
        print(f"Image size: {img_rgb.shape[1]}x{img_rgb.shape[0]}")

        start_time = time.time()
        results = self.model.predict(
            img_rgb,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            imgsz=640  # Input size (larger = more accurate but slower)
        )
        detection_time = time.time() - start_time

        detections = []

        if len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())

                    if hasattr(box, 'cls'):
                        cls = int(box.cls[0].cpu().numpy())
                        if cls != 0:
                            continue

                    detections.append([
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                        conf
                    ])

        print(f"Detected faces: {len(detections)}")
        print(f"Detection time: {detection_time:.3f} seconds")
        print(f"{'='*60}")

        if visualize and len(detections) > 0:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle

            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(img_rgb)

            for (xmin, ymin, xmax, ymax, score) in detections:
                rect = Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                               linewidth=2, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, ymin-5, f'{score:.2f}',
                       color='lime', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

            ax.set_title(f'YOLOv8 Face Detection: {len(detections)} faces detected',
                        fontsize=14, fontweight='bold')
            ax.axis('off')
            plt.tight_layout()
            plt.show()

        return detections

    def detect_faces_batch(self, image_paths, verbose=True):
        all_detections = []

        for i, img_path in enumerate(image_paths, 1):
            if verbose and i % 10 == 0:
                print(f"Processing image {i}/{len(image_paths)}")

            detections = self.detect_faces(img_path, visualize=False)
            all_detections.append(detections)

        return all_detections


def test_yolo_face_detector():
    from paths import VAL_DIR

    detector = YOLOFaceDetector(
        model_path=None,
        conf_threshold=0.25,
        iou_threshold=0.45
    )

    test_image = VAL_DIR / "validare" / "0005.jpg"

    if test_image.exists():
        detections = detector.detect_faces(test_image, visualize=True)
        print(f"\nOK: Test complet! Detected {len(detections)} faces.")
    else:
        print(f"Error: Image {test_image} does not exist!")


if __name__ == "__main__":
    test_yolo_face_detector()
