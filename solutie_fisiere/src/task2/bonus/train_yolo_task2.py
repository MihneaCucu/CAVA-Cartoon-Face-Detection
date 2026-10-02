"""
Script for training YOLOv8 on Task 2 (character classification).
Trains 4 separate models for each character.
"""

import sys
from pathlib import Path
import shutil
import yaml

sys.path.append(str(Path(__file__).parent))

from paths import TRAIN_DIR, VAL_DIR, BASE_DIR, CHARACTERS
from load_annotations import load_annotations


def prepare_character_dataset(character):
    dataset_dir = BASE_DIR / f"yolo_dataset_{character}"

    for split in ['train', 'val']:
        (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PREPARING YOLO DATASET - {character.upper()}")
    print(f"{'='*60}")

    print(f"\n1. Processing training data {character}...")
    process_character_split(
        source_dir=TRAIN_DIR / character,
        annotations_file=TRAIN_DIR / f"{character}_annotations.txt",
        character=character,
        images_dir=dataset_dir / 'images' / 'train',
        labels_dir=dataset_dir / 'labels' / 'train'
    )

    print(f"\n2. Processing validation data {character}...")
    process_character_split(
        source_dir=VAL_DIR / "validare",
        annotations_file=VAL_DIR / f"task2_{character}_gt_validare.txt",
        character=character,
        images_dir=dataset_dir / 'images' / 'val',
        labels_dir=dataset_dir / 'labels' / 'val'
    )

    print(f"\n3. Creating data.yaml for {character}...")
    data_yaml = {
        'path': str(dataset_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': [character]
    }

    with open(dataset_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"\nOK: YOLO dataset prepared for {character} in: {dataset_dir}")
    print(f"{'='*60}\n")

    return dataset_dir


def process_character_split(source_dir, annotations_file, character, images_dir, labels_dir):
    import cv2
    from collections import defaultdict

    is_task2_val = "task2_" in str(annotations_file) and "_gt_validare" in str(annotations_file)

    if is_task2_val:
        annotations_dict = defaultdict(list)
        with open(annotations_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    filename = parts[0]
                    x1, y1, x2, y2 = map(int, parts[1:5])
                    annotations_dict[filename].append((x1, y1, x2, y2, character))
    else:
        annotations_dict = load_annotations(annotations_file, character=character)

    total_images = 0
    total_faces = 0

    for img_name, face_data in annotations_dict.items():
        src_img = source_dir / img_name
        dst_img = images_dir / img_name

        if not src_img.exists():
            continue

        shutil.copy2(src_img, dst_img)

        img = cv2.imread(str(src_img))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        label_file = labels_dir / f"{Path(img_name).stem}.txt"

        with open(label_file, 'w') as f:
            for face in face_data:
                x1, y1, x2, y2, label = face

                if label != character:
                    continue

                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                total_faces += 1

        total_images += 1

    print(f"  OK: Processed {total_images} images with {total_faces} faces {character}")


def train_character_model(character, dataset_dir, epochs=50, imgsz=640, batch=16):
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"TRAINING YOLOv8 MODEL - {character.upper()}")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_dir}")
    print(f"Epochs: {epochs}")
    print(f"{'='*60}\n")

    model = YOLO('yolov8n.pt')
    results = model.train(
        data=str(dataset_dir / 'data.yaml'),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=f'{character}_detector',
        project=str(BASE_DIR / 'yolo_training_task2'),
        patience=10,  # Early stopping
        save=True,
        plots=True,
        val=True,
        verbose=True
    )

    print(f"\nOK: Training complete for {character}!")
    print(f"Model saved in: {BASE_DIR / 'yolo_training_task2' / f'{character}_detector'}")

    return results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TRAINING YOLO TASK 2 - CHARACTER CLASSIFICATION")
    print("="*60)
    print("4 separate models will be trained:")
    for char in CHARACTERS:
        print(f"  - {char.capitalize()}")
    print("="*60 + "\n")

    for character in CHARACTERS:
        print(f"\n{'#'*60}")
        print(f"# CHARACTER: {character.upper()}")
        print(f"{'#'*60}\n")

        dataset_dir = prepare_character_dataset(character)

        train_character_model(
            character=character,
            dataset_dir=dataset_dir,
            epochs=50,
            imgsz=640,
            batch=16
        )

    print("\n" + "="*60)
    print("OK: ALL MODELS HAVE BEEN TRAINED!")
    print("="*60)
    print("\nModels saved in: yolo_training_task2/")
    print("  - daphne_detector/weights/best.pt")
    print("  - fred_detector/weights/best.pt")
    print("  - shaggy_detector/weights/best.pt")
    print("  - velma_detector/weights/best.pt")
    print("\nNext steps:")
    print("1. Create a script for Task 2 which:")
    print("   - Detect faces with Task 1 model")
    print("   - Classify each face with Task 2 models")
    print("2. Generate results for evaluation")
    print("3. Review generated predictions in outputs/.")
