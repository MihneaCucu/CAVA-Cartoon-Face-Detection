"""
Mine Hard Negatives using Full Pipeline (Task 1 → Task 2)
Runs complete detection pipeline and mines false positives as hard negatives
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import sys

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent.parent / 'src' / 'task1' / 'resnet'))

from model import create_model
from generate_results_resnet import detect_faces_resnet, non_max_suppression


def load_ground_truth():
    """Load Task 2 ground truth annotations"""
    train_dir = Path('..') / 'antrenare'

    gt_data = {}

    characters = ['daphne', 'fred', 'shaggy', 'velma']

    for char in characters:
        gt_file = train_dir / f'{char}_annotations.txt'

        if not gt_file.exists():
            continue

        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue

                img_name = parts[0]
                x1, y1, x2, y2 = map(int, parts[1:5])
                label = parts[5]

                full_img_name = f'{char}/{img_name}'

                if full_img_name not in gt_data:
                    gt_data[full_img_name] = []

                bbox = (x1, y1, x2, y2)
                gt_data[full_img_name].append((bbox, label))

    return gt_data


def compute_iou(box1, box2):
    """Compute IoU between two boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def mine_hnm_with_pipeline(max_images=100, threshold=0.2, device='mps', visualize=True,
                           non_random=False, specific_images=None):
    print("="*60)
    print(f"EFFICIENT PIPELINE HNM - ALL CHARACTERS")
    print("="*60)
    if non_random and specific_images:
        print(f"Processing {len(specific_images)} specific images")
    else:
        print(f"Processing {max_images} random images from all 4 folders")
    print("Mining HN for ALL 4 characters simultaneously!")

    print("\nOK: Loading Task 1 face detector...")
    task1_model_path = Path('models/resnet18_face_detector.pth')
    task1_model, dev = create_model(num_classes=2, device=device)
    checkpoint = torch.load(task1_model_path, map_location=dev)
    task1_model.load_state_dict(checkpoint['model_state_dict'])
    task1_model.eval()
    print(f"  Task 1 Val Acc: {checkpoint['val_acc']:.2f}%")

    print("\nOK: Loading Task 2 character models (RETRAINED)...")
    task2_models = {}
    for char in ['daphne', 'fred', 'shaggy', 'velma']:
        model_path = Path(f'models/task2/resnet18_{char}.pth')
        model, _ = create_model(num_classes=2, device=dev)
        cp = torch.load(model_path, map_location=dev)
        model.load_state_dict(cp['model_state_dict'])
        model.eval()
        task2_models[char] = model
        print(f"  {char}: Val Acc = {cp['val_acc']:.2f}%")

    print("\nOK: Loading ground truth...")
    gt_data = load_ground_truth()
    print(f"  {len(gt_data)} images with annotations")

    from torchvision import transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    if visualize:
        viz_dir = Path('hnm_pipeline_viz_all')
        viz_dir.mkdir(exist_ok=True)

    colors = {
        'daphne': (0, 165, 255),
        'fred': (255, 0, 0),
        'shaggy': (0, 255, 0),
        'velma': (255, 0, 255),
        'unknown': (128, 128, 128)
    }

    hard_negatives = {
        'daphne': [],
        'fred': [],
        'shaggy': [],
        'velma': []
    }
    train_dir = Path('..') / 'antrenare'
    processed = 0

    print(f"\nOK: Mining with pipeline...")
    print(f"  Total available images: {len(gt_data)}")

    # Select images based on mode
    if non_random and specific_images:
        # Use specific images list
        print(f"  Mode: SPECIFIC IMAGES")
        print(f"  Images to process: {len(specific_images)}")

        # Filter gt_data to only include specific images
        sampled_images = [(img, gt_data[img]) for img in specific_images if img in gt_data]


        print("="*60)
        print(gt_data['velma/0371.jpg'])

        print("="*60)
        print(sampled_images)
        if len(sampled_images) < len(specific_images):
            missing = set(specific_images) - set(img for img, _ in sampled_images)
            print(f"  Warning: {len(missing)} images not found in GT: {missing}")
    else:
        # Get all images from GT data
        all_images = list(gt_data.items())

        # Sample images
        if non_random:
            sampled_images = all_images[:max_images]
            print(f"  Mode: SEQUENTIAL SAMPLING")
        else:
            import random
            sampled_images = random.sample(all_images, min(max_images, len(all_images)))
            print(f"  Mode: RANDOM SAMPLING")

    print(f"  Mining HN for: ALL 4 characters")
    print(f"  Threshold: {threshold}")

    for img_name, gt_annotations in tqdm(sampled_images):

        print("="*60)
        print(f"Processing: {img_name}")
        print("="*60)
        img_path = train_dir / img_name
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Task 1: Detect faces
        detections_array, scores_array = detect_faces_resnet(
            image_rgb,
            task1_model,
            dev,
            window_sizes=[(32, 32), (35, 45)],
            step_size=4,
            scale_factor=1.25,
            threshold=0.6
        )

        # Apply NMS
        if len(detections_array) > 0:
            detections_array, scores_array = non_max_suppression(
                detections_array,
                scores_array,
                iou_threshold=0.2
            )
        else:
            continue

        # Visualization setup
        if visualize:
            vis_image = image.copy()
            hn_count = 0

            # Draw GT bboxes in GREEN
            for gt_bbox, gt_char in gt_annotations:
                gx1, gy1, gx2, gy2 = gt_bbox
                cv2.rectangle(vis_image, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                cv2.putText(vis_image, f"GT:{gt_char}", (gx1, gy1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Task 2: Classify each detected face
        for det_bbox, det_score in zip(detections_array, scores_array):
            x1, y1, x2, y2 = map(int, det_bbox)

            # Extract face patch
            face_patch = image_rgb[y1:y2, x1:x2]
            if face_patch.size == 0:
                continue

            face_64 = cv2.resize(face_patch, (64, 64))

            # Classify with all Task 2 models
            face_norm = face_64.astype(np.float32) / 255.0
            face_tensor = torch.from_numpy(face_norm).permute(2, 0, 1).unsqueeze(0)
            face_tensor = normalize(face_tensor).to(dev)

            scores = {}
            with torch.no_grad():
                for char, model in task2_models.items():
                    output = model(face_tensor)
                    prob = F.softmax(output, dim=1)
                    scores[char] = prob[0, 1].item()

            # Get predicted character
            pred_char = max(scores, key=scores.get)
            pred_score = scores[pred_char]

            # Check if this is a hard negative for ANY character
            if pred_score >= threshold:
                # Find GT label with max IoU
                max_iou = 0.0
                gt_label = None


                print(gt_annotations)

                for gt_bbox, gt_char in gt_annotations:
                    iou = compute_iou((x1, y1, x2, y2), gt_bbox)
                    print("GT bbox:", gt_bbox)
                    print("GT char:", gt_char)
                    print("IoU:", iou)
                    print("Det bbox:", (x1, y1, x2, y2))
                    print("Det char:", pred_char)
                    print()
                    if iou > max_iou:
                        max_iou = iou
                        gt_label = gt_char


                is_hn = False
                hn_for_char = None
                hn_reason = ""

                if gt_label and gt_label != pred_char:
                    hard_negatives[pred_char].append(face_64)
                    is_hn = True
                    hn_for_char = pred_char
                    hn_reason = f"GT:{gt_label} != Pred:{pred_char}"
                    print(hn_reason)
                elif gt_label is None and max_iou < 0.1:
                    hard_negatives[pred_char].append(face_64)
                    is_hn = True
                    hn_for_char = pred_char
                    hn_reason = f"BG->{pred_char}"

                if visualize:
                    if is_hn:
                        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        label_text = f"HN({hn_for_char}): {hn_reason} ({pred_score:.2f})"
                        cv2.putText(vis_image, label_text, (x1, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                        hn_count += 1
                    else:
                        color = colors.get(pred_char, (128, 128, 128))
                        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 1)
                        cv2.putText(vis_image, f"{pred_char} ({pred_score:.2f})", (x1, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        if visualize:
            out_path = viz_dir / f'{Path(img_name).stem}_hn{hn_count}.jpg'
            cv2.imwrite(str(out_path), vis_image)
            if hn_count > 0:
                print(f"  {img_name}: {hn_count} HN → {out_path.name}")

        processed += 1

    output_dir = Path('processed_data')
    total_hn = 0

    print(f"\n{'='*60}")
    print(f"OK: EFFICIENT PIPELINE HNM COMPLETE!")
    print(f"{'='*60}")
    print(f"Random images processed: {processed} (from all 4 folders)")
    print(f"\nHard negatives per character:")

    for char in ['daphne', 'fred', 'shaggy', 'velma']:
        if len(hard_negatives[char]) > 0:
            output_file = output_dir / f'{char}_hard_negatives_pipeline.npy'

            # Load existing HN if file exists
            if output_file.exists():
                existing_hn = np.load(output_file)
                print(f"  {char.capitalize()}: {len(existing_hn)} existing + {len(hard_negatives[char])} new", end='')
                # Append new HN to existing
                combined_hn = np.concatenate([existing_hn, np.array(hard_negatives[char])], axis=0)
                np.save(output_file, combined_hn)
                print(f" = {len(combined_hn)} total HN → {output_file.name}")
                total_hn += len(hard_negatives[char])
            else:
                # Save new HN file
                hn_array = np.array(hard_negatives[char])
                np.save(output_file, hn_array)
                print(f"  {char.capitalize()}: {len(hard_negatives[char])} HN (new file) → {output_file.name}")
                total_hn += len(hard_negatives[char])
        else:
            print(f"  {char.capitalize()}: 0 new HN")

    print(f"\nTotal HN mined: {total_hn}")
    if visualize:
        print(f"Visualizations: {viz_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    non_random = False
    max_images = 5
    specific_images = None

    threshold = 0.2
    device = 'mps'
    visualize = True

    mine_hnm_with_pipeline(max_images, threshold, device, visualize,
                           non_random, specific_images)
