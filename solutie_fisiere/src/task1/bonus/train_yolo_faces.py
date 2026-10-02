"""
Script for training YOLOv8 on the faces dataset.
Prepares data in YOLO format and trains the model.
"""

import sys
from pathlib import Path
import shutil
import yaml

sys.path.append(str(Path(__file__).parent))

# Fix path to support imports from utils
current_dir = Path(__file__).parent
solutie_fisiere_root = current_dir.parent.parent.parent
sys.path.insert(0, str(solutie_fisiere_root / 'utils'))

from paths import TRAIN_DIR, VAL_DIR, PROJECT_ROOT



def load_annotations(annotation_file, character=""):
    from collections import defaultdict

    annotations = defaultdict(list)

    with open(annotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 5:
                filename = parts[0]
                x1, y1, x2, y2 = map(int, parts[1:5])
                annotations[filename].append((x1, y1, x2, y2, 'face'))

    return annotations


def prepare_yolo_dataset():
    dataset_dir = PROJECT_ROOT / "yolo_dataset"

    # CLEAN UP PREVIOUS RUNS (CRITICAL FIX)
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)


    for split in ['train', 'val']:
        (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("PREPARE YOLO DATASET")
    print(f"{'='*60}")

    print("\n1. Process training data...")
    train_annotation_files = [
        TRAIN_DIR / "daphne_annotations.txt",
        TRAIN_DIR / "fred_annotations.txt",
        TRAIN_DIR / "shaggy_annotations.txt",
        TRAIN_DIR / "velma_annotations.txt"
    ]

    process_split_multiple(
        source_dirs=[TRAIN_DIR / char for char in ['daphne', 'fred', 'shaggy', 'velma']],
        annotations_files=train_annotation_files,
        images_dir=dataset_dir / 'images' / 'train',
        labels_dir=dataset_dir / 'labels' / 'train'
    )

    print("\n2. Process validation data...")
    process_split(
        source_dir=VAL_DIR / "validare",
        annotations_file=VAL_DIR / "task1_gt_validare.txt",
        images_dir=dataset_dir / 'images' / 'val',
        labels_dir=dataset_dir / 'labels' / 'val'
    )

    print("\n3. Create data.yaml...")
    data_yaml = {
        'path': str(dataset_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['face']
    }

    with open(dataset_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"\nOK: Dataset YOLO prepared in: {dataset_dir}")
    print(f"{'='*60}\n")

    return dataset_dir


def process_split_multiple(source_dirs, annotations_files, images_dir, labels_dir):
    import cv2

    total_images = 0
    total_faces = 0

    for source_dir, annotations_file in zip(source_dirs, annotations_files):
        print(f"  Procesare {annotations_file.name}...")

        annotations_dict = load_annotations(annotations_file, character="")

        # Helper to extract character name from path (e.g. 'daphne' from '.../train/daphne')
        char_name = source_dir.name

        for img_name, face_data in annotations_dict.items():
            src_img = source_dir / img_name
            # PREFIX FILENAME TO AVOID COLLISION
            new_img_name = f"{char_name}_{img_name}"
            dst_img = images_dir / new_img_name

            if not src_img.exists():
                continue

            # Copy image if not exists
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)
                total_images += 1

            label_file = labels_dir / f"{Path(new_img_name).stem}.txt"

            img = cv2.imread(str(src_img))
            if img is None:
                continue

            img_h, img_w = img.shape[:2]

            with open(label_file, 'a') as f:
                for face in face_data:
                    x1, y1, x2, y2, label = face

                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h

                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    total_faces += 1

    print(f"  OK: Processed {total_images} images with {total_faces} faces")


def process_split(source_dir, annotations_file, images_dir, labels_dir):
    import cv2

    annotations_dict = load_annotations(annotations_file)

    total_images = 0
    total_faces = 0

    for img_name, face_boxes in annotations_dict.items():
        src_img = source_dir / img_name
        dst_img = images_dir / img_name

        if not src_img.exists():
            print(f"  WARNING: {img_name} does not exist!")
            continue

        shutil.copy2(src_img, dst_img)

        img = cv2.imread(str(src_img))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        label_file = labels_dir / f"{Path(img_name).stem}.txt"

        with open(label_file, 'w') as f:
            for box in face_boxes:
                x1, y1, x2, y2, _ = box

                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                total_faces += 1

        total_images += 1

    print(f"  OK: Processed {total_images} images with {total_faces} faces")


def train_yolo_model(dataset_dir, epochs=100, imgsz=640, batch=16):
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print("TRAINING YOLOv8")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_dir}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"{'='*60}\n")

    model = YOLO('yolov8n.pt')

    results = model.train(
        data=str(dataset_dir / 'data.yaml'),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name='face_detector',
        project=str(PROJECT_ROOT / 'yolo_training'),
        patience=20,
        save=True,
        plots=True,
        device='mps'
    )

    print(f"\nOK: Training completed!")
    print(f"Model saved in: {PROJECT_ROOT / 'yolo_training' / 'face_detector'}")

    return results


if __name__ == "__main__":
    dataset_dir = prepare_yolo_dataset()

    print("\nTraining...")

    train_yolo_model(
        dataset_dir=dataset_dir,
        epochs=3,
        imgsz=640,
        batch=16
    )

    print("\nOK: Training completed!")
    print("\nNext steps:")
    print("1. Use the trained model in detect_faces_yolo.py")
    print("2. Run generate_results_yolo.py with the new model")
    print("3. Review the generated predictions in outputs/.")
