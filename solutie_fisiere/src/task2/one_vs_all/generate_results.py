"""
Generate Task 2 Results using One-vs-All Binary Models
Loads Task 1 face detections and classifies each using 4 binary ResNet models
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import sys

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent / 'task1' / 'resnet'))

from model import create_model

SPLIT_NAME = 'validare'  # Options: 'validare', 'testare'


def load_binary_models(device):
    """Load all 4 binary character models"""
    models = {}
    characters = ['daphne', 'fred', 'shaggy', 'velma']

    print("Loading binary models...")
    for char in characters:
        # Models are in 'models' subdirectory relative to this script
        model_path = current_dir / 'models' / f'resnet18_{char}_weighted.pth'
        if not model_path.exists():
            print(f"ERROR: Model not found: {model_path}")
            continue

        model, _ = create_model(num_classes=2, device=device)

        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        models[char] = model
        print(f"OK: Loaded {char}: Val Acc {checkpoint['val_acc']:.2f}%")

    return models


def classify_face(face_patch, models, device, threshold=0.5):
    from torchvision import transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    patch = face_patch.astype(np.float32) / 255.0
    patch = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0)
    patch = normalize(patch).to(device)

    scores = {}
    with torch.no_grad():
        for char, model in models.items():
            output = model(patch)
            prob = F.softmax(output, dim=1)
            scores[char] = prob[0, 1].item()

    max_char = max(scores, key=scores.get)
    max_score = scores[max_char]

    if max_score < threshold:
        return 'unknown', max_score
    else:
        return max_char, max_score


def main():
    print("="*60)
    print("TASK 2 - ONE-VS-ALL DETECTION")
    print("="*60)

    project_root = Path(__file__).parent.parent.parent.parent.parent

    if torch.cuda.is_available():
        device = 'cuda'
        print("OK: Using CUDA")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("OK: Using MPS")
    else:
        device = 'cpu'
        print("OK: Using CPU")

    models = load_binary_models(device)
    if len(models) != 4:
        print(f"ERROR: Expected 4 models, found {len(models)}")
        return

    task1_dir = project_root / 'outputs' / 'task1'
    detections_file = task1_dir / 'detections_all_faces.npy'
    scores_file = task1_dir / 'scores_all_faces.npy'
    filenames_file = task1_dir / 'file_names_all_faces.npy'

    if not detections_file.exists():
        print(f"ERROR: Task 1 detections not found: {detections_file}")
        return

    print(f"\nOK: Loading Task 1 detections...")
    task1_detections = np.load(detections_file)
    task1_scores = np.load(scores_file)
    task1_filenames = np.load(filenames_file)

    print(f"  Total faces detected by Task 1: {len(task1_detections)}")

    score_threshold = 0.6
    mask = task1_scores >= score_threshold
    task1_detections = task1_detections[mask]
    task1_scores = task1_scores[mask]
    task1_filenames = task1_filenames[mask]

    print(f"  After filtering (score >= {score_threshold}): {len(task1_detections)} faces")

    character_detections = {
        'daphne': {'detections': [], 'scores': [], 'filenames': []},
        'fred': {'detections': [], 'scores': [], 'filenames': []},
        'shaggy': {'detections': [], 'scores': [], 'filenames': []},
        'velma': {'detections': [], 'scores': [], 'filenames': []},
        'unknown': {'detections': [], 'scores': [], 'filenames': []}
    }

    if SPLIT_NAME == 'validare':
        data_dir = project_root / 'validare' / 'validare'
    else:
        data_dir = project_root / 'testare' / 'testare'
        if not data_dir.exists():
             data_dir = project_root / 'testare'

    print(f"\nOK: Classifying faces from {data_dir}...")
    for i, (bbox, t1_score, filename) in enumerate(tqdm(zip(task1_detections, task1_scores, task1_filenames),
                                                          total=len(task1_detections))):
        img_path = data_dir / filename
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract face patch
        x_min, y_min, x_max, y_max = bbox.astype(int)

        # Bounds check
        h, w = image_rgb.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        if x_max <= x_min or y_max <= y_min:
            continue

        face_patch = image_rgb[y_min:y_max, x_min:x_max]

        # Resize to 64×64
        face_64 = cv2.resize(face_patch, (64, 64))

        # Classify using argmax (no threshold for maximum recall)
        character, char_score = classify_face(face_64, models, device, threshold=0.05)

        # Store result
        character_detections[character]['detections'].append(bbox)
        character_detections[character]['scores'].append(char_score)
        character_detections[character]['filenames'].append(filename)

    output_dir = project_root / 'outputs' / 'task2'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOK: Saving results...")
    for char in ['daphne', 'fred', 'shaggy', 'velma', 'unknown']:
        detections = np.array(character_detections[char]['detections'])
        scores = np.array(character_detections[char]['scores'])
        filenames = np.array(character_detections[char]['filenames'])

        np.save(output_dir / f'detections_{char}.npy', detections)
        np.save(output_dir / f'scores_{char}.npy', scores)
        np.save(output_dir / f'file_names_{char}.npy', filenames)

        print(f"  {char}: {len(detections)} detections")

    print(f"\n{'='*60}")
    print("OK: TASK 2 DETECTION COMPLETE!")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"\nDetections per character:")
    for char in ['daphne', 'fred', 'shaggy', 'velma', 'unknown']:
        count = len(character_detections[char]['detections'])
        print(f"  {char.capitalize()}: {count}")
    print("\nReview generated predictions in the outputs directory.")
    print("="*60)


if __name__ == "__main__":
    main()
