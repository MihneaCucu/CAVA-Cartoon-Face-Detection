"""
Generate detection results using trained ResNet-18
Sliding window detection
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import sys

current_dir = Path(__file__).parent
solutie_fisiere_root = current_dir.parent.parent.parent
sys.path.insert(0, str(solutie_fisiere_root / 'utils'))
sys.path.insert(0, str(current_dir))

from paths import VAL_DIR, PROCESSED_DATA_DIR
from model import create_model
from torchvision import transforms

SPLIT_NAME = 'validare' # Options: 'validare', 'testare'


def load_trained_model(checkpoint_path, device='auto'):
    """Load trained ResNet model"""
    model, device = create_model(num_classes=2, device=device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"OK: Model loaded from {checkpoint_path}")
    print(f"  Val Acc: {checkpoint['val_acc']:.2f}%")

    return model, device


def detect_faces_resnet(image, model, device, window_sizes=[(32, 32), (35, 45)],
                        step_size=4, scale_factor=1.25, threshold=0.6):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    detections = []
    scores = []
    scale = 1.0
    while True:
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        if new_h < 32 or new_w < 32:
            break

        resized = cv2.resize(image, (new_w, new_h))

        # For each window size
        for win_h, win_w in window_sizes:
            if win_h > new_h or win_w > new_w:
                continue

            # Sliding window
            windows = []
            positions = []

            for y in range(0, new_h - win_h + 1, step_size):
                for x in range(0, new_w - win_w + 1, step_size):
                    window = resized[y:y+win_h, x:x+win_w]
                    window = cv2.resize(window, (64, 64))
                    window = window.astype(np.float32) / 255.0
                    window = torch.from_numpy(window).permute(2, 0, 1)
                    window = normalize(window)

                    windows.append(window)
                    positions.append((x, y, win_w, win_h))

            if len(windows) == 0:
                continue

            batch_size = 64
            for i in range(0, len(windows), batch_size):
                batch = torch.stack(windows[i:i+batch_size]).to(device)

                with torch.no_grad():
                    outputs = model(batch)
                    probs = F.softmax(outputs, dim=1)[:, 1]

                for j, prob in enumerate(probs):
                    if prob >= threshold:
                        x, y, w, h = positions[i + j]

                        x_min = int(x / scale)
                        y_min = int(y / scale)
                        x_max = int((x + w) / scale)
                        y_max = int((y + h) / scale)

                        detections.append([x_min, y_min, x_max, y_max])
                        scores.append(prob.item())

        scale /= scale_factor

    return np.array(detections), np.array(scores)


def non_max_suppression(detections, scores, iou_threshold=0.2):
    if len(detections) == 0:
        return np.array([]), np.array([])

    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / union

        inds = np.where(iou < iou_threshold)[0]
        order = order[inds + 1]

    keep = np.array(keep)
    return detections[keep], scores[keep]


def main():
    print("="*60)
    print("RESNET FACE DETECTOR - GENERATE RESULTS")
    print("="*60)

    checkpoint_path = current_dir.parent.parent.parent / 'models' / 'resnet18_face_detector.pth'
    if not checkpoint_path.exists():
        print(f"ERROR: Model not found: {checkpoint_path}")
        print("Please train the model first!")
        return

    model, device = load_trained_model(checkpoint_path, device='auto')

    project_root = Path(__file__).parent.parent.parent.parent.parent
    if SPLIT_NAME == 'validare':
        data_dir = project_root / 'validare' / 'validare'
    else:
        data_dir = project_root / 'testare' / 'testare'
        if not data_dir.exists():
             data_dir = project_root / 'testare'

    print(f"Directory: {data_dir}")
    image_files = sorted(data_dir.glob('*.jpg'))

    if len(image_files) == 0:
        print(f"ERROR: No images found in {data_dir}")
        return

    print(f"\nProcessing {len(image_files)} images...")

    all_detections = []
    all_scores = []
    all_filenames = []

    for img_path in tqdm(image_files):
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detections, scores = detect_faces_resnet(
            image, model, device,
            window_sizes=[(32, 32), (35, 45)],
            step_size=4,
            scale_factor=1.25,
            threshold=0.8
        )

        if len(detections) > 0:
            detections, scores = non_max_suppression(detections, scores, iou_threshold=0.2)

        for det, score in zip(detections, scores):
            all_detections.append(det)
            all_scores.append(score)
            all_filenames.append(img_path.name)

    all_detections = np.array(all_detections)
    all_scores = np.array(all_scores)
    all_filenames = np.array(all_filenames)

    project_root = Path(__file__).parent.parent.parent.parent.parent
    output_dir = project_root / 'outputs' / 'task1'

    if (output_dir / 'detections_all_faces.npy').exists():
        backup_dir = output_dir.parent / 'task1_hogsvm_backup'
        backup_dir.mkdir(exist_ok=True)
        import shutil
        for f in output_dir.glob('*.npy'):
            shutil.copy(f, backup_dir / f.name)
        print(f"Backup HOG results to: {backup_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / 'detections_all_faces.npy', all_detections)
    np.save(output_dir / 'scores_all_faces.npy', all_scores)
    np.save(output_dir / 'file_names_all_faces.npy', all_filenames)

    print(f"\nOK: Results saved:")
    print(f"  Total detections: {len(all_detections)}")
    print(f"  Average detections/image: {len(all_detections)/len(image_files):.1f}")
    print(f"  Output dir: {output_dir}")


if __name__ == "__main__":
    main()
