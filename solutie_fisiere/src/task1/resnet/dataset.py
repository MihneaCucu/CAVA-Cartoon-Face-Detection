"""
PyTorch Dataset for Face Detection
Load positive and negative patches for ResNet training
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import sys

current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
sys.path.insert(0, str(src_dir / 'utils'))

from paths import PROCESSED_DIR


class FaceDataset(Dataset):
    """Dataset for face detection with positive and negative patches"""

    def __init__(self, split='train', transform=None, balance_ratio=1.0):
        """
        Args:
            split: 'train' or 'val'
            transform: torchvision transforms
            balance_ratio: negative:positive ratio (1.0 = balanced)
        """
        self.split = split
        self.transform = transform

        positive_patches = np.load(PROCESSED_DIR / 'positive_patches.npy')
        negative_patches = np.load(PROCESSED_DIR / 'negative_patches.npy')

        hard_negatives_path = PROCESSED_DIR / 'hard_negative_patches_combined.npy'
        if hard_negatives_path.exists():
            hard_negatives = np.load(hard_negatives_path)
            print(f"OK: Loaded {len(hard_negatives)} hard negatives")
            negative_patches = np.concatenate([negative_patches, hard_negatives])

        num_positives = len(positive_patches)
        num_negatives = int(num_positives * balance_ratio)

        if num_negatives > len(negative_patches):
            print(f"WARNING: Requested {num_negatives} negatives, but only {len(negative_patches)} available")
            num_negatives = len(negative_patches)

        np.random.seed(42)
        neg_indices = np.random.choice(len(negative_patches), num_negatives, replace=False)
        negative_patches = negative_patches[neg_indices]

        self.patches = np.concatenate([positive_patches, negative_patches], axis=0)
        self.labels = np.concatenate([
            np.ones(len(positive_patches)),
            np.zeros(len(negative_patches))
        ])

        # Train/Val split (90/10)
        np.random.seed(42)
        indices = np.random.permutation(len(self.patches))
        split_idx = int(0.9 * len(indices))

        if split == 'train':
            indices = indices[:split_idx]
        else:
            indices = indices[split_idx:]

        self.patches = self.patches[indices]
        self.labels = self.labels[indices]

        print(f"{split.upper()} dataset: {len(self)} samples")
        print(f"  Positive: {(self.labels == 1).sum()}")
        print(f"  Negative: {(self.labels == 0).sum()}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        label = self.labels[idx]

        patch = patch.astype(np.float32) / 255.0

        patch = torch.from_numpy(patch).permute(2, 0, 1)

        if self.transform:
            patch = self.transform(patch)

        return patch, torch.tensor(label, dtype=torch.long)


def get_dataloaders(batch_size=128, num_workers=4, balance_ratio=1.0):
    from torchvision import transforms

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FaceDataset(split='train', transform=train_transform, balance_ratio=balance_ratio)
    val_dataset = FaceDataset(split='val', transform=val_transform, balance_ratio=balance_ratio)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing FaceDataset...")

    train_loader, val_loader = get_dataloaders(batch_size=32, num_workers=0, balance_ratio=0.65)

    batch, labels = next(iter(train_loader))
    print(f"\nBatch shape: {batch.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Batch range: [{batch.min():.2f}, {batch.max():.2f}]")
    print(f"Labels: {labels[:10]}")

    print("\nOK: Dataset test passed!")
