"""
Generate results for evaluation using the YOLOv8 model
"""

import sys
from pathlib import Path
import numpy as np
import glob

current_dir = Path(__file__).parent
solutie_fisiere_root = current_dir.parent.parent.parent
sys.path.insert(0, str(solutie_fisiere_root / 'utils'))

from detect_faces_yolo import YOLOFaceDetector
from paths import VAL_DIR

SPLIT_NAME = 'validare'  # Options: 'validare', 'testare'


def generate_results_yolo(
    test_dir,
    output_dir,
    model_path,
    conf_threshold=0.25,
    iou_threshold=0.45,
    max_images=None
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLOFaceDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )

    test_dir = Path(test_dir)
    test_images = sorted(glob.glob(str(test_dir / "*.jpg")))

    if max_images is not None:
        test_images = test_images[:max_images]

    if len(test_images) == 0:
        print(f"ERROR: No images found in {test_dir}")
        return

    print(f"\n{'='*60}")
    print(f"GENERATE RESULTS YOLOv8 - {len(test_images)} IMAGES")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"{'='*60}\n")

    all_detections = []
    all_scores = []
    all_file_names = []

    total_detections = 0

    for i, img_path in enumerate(test_images, 1):
        img_name = Path(img_path).name

        print(f"[{i}/{len(test_images)}] Processing {img_name}...", end=' ')

        detections = detector.detect_faces(img_path, visualize=False)

        if len(detections) > 0:
            for detection in detections:
                xmin, ymin, xmax, ymax, score = detection

                all_detections.append([int(xmin), int(ymin), int(xmax), int(ymax)])
                all_scores.append(float(score))
                all_file_names.append(img_name)

        total_detections += len(detections)
        print(f"OK: {len(detections)} detections")

    all_detections = np.array(all_detections, dtype=int) if all_detections else np.array([])
    all_scores = np.array(all_scores, dtype=float) if all_scores else np.array([])
    all_file_names = np.array(all_file_names, dtype=str) if all_file_names else np.array([])

    np.save(output_dir / 'detections_all_faces.npy', all_detections)
    np.save(output_dir / 'scores_all_faces.npy', all_scores)
    np.save(output_dir / 'file_names_all_faces.npy', all_file_names)

    print(f"\n{'='*60}")
    print("FINAL STATISTICS")
    print(f"{'='*60}")
    print(f"Total images processed: {len(test_images)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections/image: {total_detections/len(test_images):.2f}")
    print(f"\nOK: Files saved:")
    print(f"  - detections_all_faces.npy ({all_detections.shape})")
    print(f"  - scores_all_faces.npy ({all_scores.shape})")
    print(f"  - file_names_all_faces.npy ({all_file_names.shape})")
    print(f"{'='*60}\n")

    return all_detections, all_scores, all_file_names


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent

    # Define paths based on SPLIT_NAME
    # __file__ is src/task1/bonus/generate_results_yolo.py
    # parents: 0=bonus, 1=task1, 2=src, 3=solutie_fisiere, 4=CAVA-2025-TEMA2
    project_root = Path(__file__).resolve().parents[4]
    if SPLIT_NAME == 'validare':
        test_directory = project_root / 'validare' / 'validare'
    else:
        test_directory = project_root / 'testare' / 'testare'
        if not test_directory.exists():
            test_directory = project_root / 'testare'

    output_directory = project_root / "outputs" / "bonus" / "task1"

    model_path = project_root / "solutie_fisiere" / "yolo_training" / "face_detector7" / "weights" / "best.pt"

    generate_results_yolo(
        test_dir=test_directory,
        output_dir=output_directory,
        model_path=str(model_path),
        conf_threshold=0.25,
        iou_threshold=0.45,
        max_images=None
    )

    print("\nOK: DONE! Results saved.")
    print("Review generated predictions in the outputs directory.")
