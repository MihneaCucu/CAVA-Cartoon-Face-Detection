"""
Generate results for Task 2 using trained YOLOv8 models.
Pipeline:
1. Detect faces with Task 1 model
2. Classify each face with Task 2 models (daphne, fred, shaggy, velma)
"""

import sys
from pathlib import Path
import numpy as np
import glob
import cv2

sys.path.append(str(Path(__file__).parent))
# Add utils to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'utils'))
# Add src/task1/bonus to path to import detect_faces_yolo
sys.path.append(str(Path(__file__).parent.parent.parent / 'task1' / 'bonus'))

from detect_faces_yolo import YOLOFaceDetector
from paths import VAL_DIR, PROJECT_ROOT, CHARACTERS

SPLIT_NAME = 'validare'  # Options: 'validare', 'testare'


def generate_results_task2(
    test_dir,
    output_dir,
    face_detector_path,
    character_models,
    conf_threshold=0.25,
    max_images=None
):
    from ultralytics import YOLO

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("GENERATING RESULTS TASK 2 - CHARACTER CLASSIFICATION")
    print(f"{'='*60}")

    print("\n1. Loading face detector...")
    face_detector = YOLOFaceDetector(
        model_path=face_detector_path,
        conf_threshold=conf_threshold,
        iou_threshold=0.45
    )

    print("\n2. Loading character classification models...")
    classifiers = {}
    for character, model_path in character_models.items():
        print(f"   - {character}: {model_path}")
        classifiers[character] = YOLO(model_path)

    test_dir = Path(test_dir)
    test_images = sorted(glob.glob(str(test_dir / "*.jpg")))

    if max_images is not None:
        test_images = test_images[:max_images]

    if len(test_images) == 0:
        print(f"ERROR: No images found in {test_dir}")
        return

    print(f"\n3. Processing {len(test_images)} images...")
    print(f"{'='*60}\n")

    results_by_character = {char: {'detections': [], 'scores': [], 'file_names': []}
                           for char in CHARACTERS}

    total_faces = 0

    for i, img_path in enumerate(test_images, 1):
        img_name = Path(img_path).name

        print(f"[{i}/{len(test_images)}] {img_name}...", end=' ')

        character_counts = {char: 0 for char in CHARACTERS}

        for character, classifier in classifiers.items():
            results = classifier.predict(
                str(img_path),
                conf=conf_threshold,
                iou=0.45,
                verbose=False
            )

            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf_score = float(box.conf[0].cpu().numpy())

                    results_by_character[character]['detections'].append([
                        int(x1), int(y1), int(x2), int(y2)
                    ])
                    results_by_character[character]['scores'].append(conf_score)
                    results_by_character[character]['file_names'].append(img_name)
                    character_counts[character] += 1
                    total_faces += 1

        char_str = ", ".join([f"{char}:{count}" for char, count in character_counts.items() if count > 0])
        if char_str:
            print(f"OK: ({char_str})")
        else:
            print("OK: 0 faces")

    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")

    task2_dir = output_dir / "task2"
    task2_dir.mkdir(parents=True, exist_ok=True)

    for character in CHARACTERS:
        detections = np.array(results_by_character[character]['detections'], dtype=int) \
                     if results_by_character[character]['detections'] else np.array([])
        scores = np.array(results_by_character[character]['scores'], dtype=float) \
                 if results_by_character[character]['scores'] else np.array([])
        file_names = np.array(results_by_character[character]['file_names'], dtype=str) \
                     if results_by_character[character]['file_names'] else np.array([])

        np.save(task2_dir / f'detections_{character}.npy', detections)
        np.save(task2_dir / f'scores_{character}.npy', scores)
        np.save(task2_dir / f'file_names_{character}.npy', file_names)

        print(f"{character}: {len(detections)} detections saved")

    print(f"\n{'='*60}")
    print(f"Total faces classified: {total_faces}")
    print(f"OK: Results saved in: {task2_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent.parent

    # Define paths based on SPLIT_NAME
    project_root = base_dir.parent
    if SPLIT_NAME == 'validare':
        test_directory = project_root / 'validare' / 'validare'
    else:
        test_directory = project_root / 'testare' / 'testare'
        if not test_directory.exists():
            test_directory = project_root / 'testare'
    output_directory = project_root / "outputs" / "bonus" / "task2"

    face_detector_path = base_dir / "yolo_training" / "face_detector7" / "weights" / "best.pt"

    character_models = {
        'daphne': str(base_dir / "yolo_training_task2" / "daphne_detector2" / "weights" / "best.pt"),
        'fred': str(base_dir / "yolo_training_task2" / "fred_detector" / "weights" / "best.pt"),
        'shaggy': str(base_dir / "yolo_training_task2" / "shaggy_detector" / "weights" / "best.pt"),
        'velma': str(base_dir / "yolo_training_task2" / "velma_detector" / "weights" / "best.pt")
    }

    print("\nVerifying models...")
    if not Path(face_detector_path).exists():
        print(f"ERROR: Model Task 1 not found: {face_detector_path}")
        exit(1)

    for char, path in character_models.items():
        if not Path(path).exists():
            print(f"ERROR: Model {char} not found: {path}")
            exit(1)

    print("OK: All models exist!\n")

    generate_results_task2(
        test_dir=test_directory,
        output_dir=output_directory,
        face_detector_path=str(face_detector_path),
        character_models=character_models,
        conf_threshold=0.25,
        max_images=None
    )

    print("\nOK: DONE! Task 2 results have been saved.")
    print("Review generated predictions in the outputs directory.")
