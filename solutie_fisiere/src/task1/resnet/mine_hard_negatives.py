"""
Hard Negative Mining for Task 1 ResNet
Extract False Positives as hard negatives for retraining
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from model import create_model
from generate_results_resnet import detect_faces_resnet


def intersection_over_union(bbox_a, bbox_b):
    """Compute IoU between two bboxes"""
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)

    box_a_area = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    box_b_area = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])

    iou = inter_area / float(box_a_area + box_b_area - inter_area + 1e-6)

    return iou


def load_ground_truth(gt_path):
    """Load ground truth annotations"""
    gt_data = {}

    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            filename = parts[0]
            bbox = list(map(int, parts[1:5]))

            if filename not in gt_data:
                gt_data[filename] = []
            gt_data[filename].append(bbox)

    return gt_data


def main():
    print("="*60)
    print("HARD NEGATIVE MINING - Task 1 ResNet")
    print("="*60)

    checkpoint_path = Path('models/resnet18_face_detector.pth')
    if not checkpoint_path.exists():
        print(f"ERROR: Model not found: {checkpoint_path}")
        return

    model, device = create_model(num_classes=2, device='auto')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"OK: Model loaded (Val Acc: {checkpoint['val_acc']:.2f}%)")

    project_root = Path(__file__).parent.parent.parent.parent.parent
    val_dir = project_root / 'validare' / 'validare'
    gt_path = project_root / 'validare' / 'task1_gt_validare.txt'

    print(f"\nOK: Loading ground truth...")
    gt_data = load_ground_truth(gt_path)
    print(f"  GT images: {len(gt_data)}")

    import random
    all_images = sorted(val_dir.glob('*.jpg'))
    random.seed(42)
    image_files = random.sample(all_images, min(5, len(all_images)))
    print(f"\nOK: Processing {len(image_files)} RANDOM images for HNM...")

    hard_negatives = []
    hn_metadata = []

    for img_path in tqdm(image_files):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detections, scores = detect_faces_resnet(image_rgb, model, device, threshold=0.7, step_size=8)

        if len(detections) == 0:
            continue

        gt_bboxes = gt_data.get(img_path.name, [])

        for det, score in zip(detections, scores):
            max_iou = 0.0
            for gt_bbox in gt_bboxes:
                iou = intersection_over_union(det, gt_bbox)
                max_iou = max(max_iou, iou)

            if max_iou < 0.2 and score > 0.7:
                x_min, y_min, x_max, y_max = det.astype(int)

                h, w = image_rgb.shape[:2]
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_max)
                y_max = min(h, y_max)

                if x_max <= x_min or y_max <= y_min:
                    continue

                patch = image_rgb[y_min:y_max, x_min:x_max]

                patch_64 = cv2.resize(patch, (64, 64))

                hard_negatives.append(patch_64)
                hn_metadata.append({
                    'image': img_path.name,
                    'bbox': det,
                    'score': score,
                    'max_iou': max_iou
                })

    hard_negatives = np.array(hard_negatives)

    print(f"\nOK: Extracted {len(hard_negatives)} hard negatives")
    print(f"  Score range: [{hn_metadata[0]['score']:.3f}, {hn_metadata[-1]['score']:.3f}]" if len(hn_metadata) > 0 else "")

    output_dir = project_root / 'processed_data'
    output_dir.mkdir(exist_ok=True)

    np.save(output_dir / 'hard_negative_patches.npy', hard_negatives)
    print(f"\nOK: Saved: {output_dir / 'hard_negative_patches.npy'}")

    print(f"\nOK: Creating visualizations...")
    viz_dir = project_root / 'outputs' / 'diagnostics' / 'hard_negatives'
    viz_dir.mkdir(exist_ok=True)

    n_samples = min(50, len(hard_negatives))
    random_indices = random.sample(range(len(hard_negatives)), n_samples)

    n_cols = 10
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2*n_rows))
    axes = axes.flatten()

    for i, idx in enumerate(random_indices):
        axes[i].imshow(hard_negatives[idx])
        axes[i].axis('off')
        if idx < len(hn_metadata):
            score = hn_metadata[idx]['score']
            iou = hn_metadata[idx]['max_iou']
            axes[i].set_title(f"S:{score:.2f}\nIoU:{iou:.2f}", fontsize=7)

    for i in range(n_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(viz_dir / 'hard_negatives_grid.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {viz_dir / 'hard_negatives_grid.png'}")

    random_20 = random.sample(range(len(hard_negatives)), min(20, len(hard_negatives)))
    for i, idx in enumerate(random_20):
        plt.figure(figsize=(3, 3))
        plt.imshow(hard_negatives[idx])
        plt.axis('off')
        if idx < len(hn_metadata):
            plt.title(f"Score: {hn_metadata[idx]['score']:.3f}, IoU: {hn_metadata[idx]['max_iou']:.3f}")
        plt.savefig(viz_dir / f'hn_{i:03d}.png', dpi=100, bbox_inches='tight')
        plt.close()

    print(f"  Saved 20 random individual samples")

    print(f"\n{'='*60}")
    print(f"OK: HNM COMPLETE!")
    print(f"{'='*60}")
    print(f"Hard negatives: {len(hard_negatives)}")
    print(f"Saved to: {output_dir / 'hard_negative_patches.npy'}")
    print(f"Visualizations: {viz_dir}")
    print(f"\nNext steps:")
    print(f"1. Review visualizations: open {viz_dir}/hard_negatives_grid.png")
    print(f"2. Add to dataset and retrain")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
