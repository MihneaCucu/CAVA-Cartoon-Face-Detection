"""
PyTorch Dataset for Binary Character Classification (One-vs-All)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class BinaryCharacterDataset(Dataset):
    """Binary dataset for one character vs rest"""

    def __init__(self, character, split='train', transform=None, use_hnm=False):
        """
        Args:
            character: 'daphne', 'fred', 'shaggy', or 'velma'
            split: 'train' or 'val'
            transform: torchvision transforms
        """
        self.character = character
        self.split = split
        self.transform = transform

        project_root = Path(__file__).parent.parent.parent.parent
        processed_dir = project_root / 'processed_data'

        positive_patches = np.load(processed_dir / f'{character}_positive.npy')
        negative_patches = np.load(processed_dir / f'{character}_negative.npy')

        if use_hnm:
            hn_file = processed_dir_solutie / f'{character}_hard_negatives_pipeline.npy'
            hard_negatives = np.load(hn_file)
            print(f"OK: Loading HN for {character}: {len(hard_negatives)} patches")
            negative_patches = np.concatenate([negative_patches, hard_negatives])
        else:
            print(f"WARNING: No HNM requested for {character}")

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

        print(f"{character.upper()} {split.upper()} dataset: {len(self)} samples")
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


def get_dataloaders(character, batch_size=128, num_workers=0, use_hnm=False):
    from torchvision import transforms

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = BinaryCharacterDataset(character, split='train', transform=train_transform, use_hnm=use_hnm)
    val_dataset = BinaryCharacterDataset(character, split='val', transform=val_transform, use_hnm=use_hnm)

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
    print("Testing BinaryCharacterDataset...")

    for char in ['daphne', 'fred', 'shaggy', 'velma']:
        print(f"\n{'='*60}")
        print(f"Testing {char.upper()}")
        print('='*60)

        train_loader, val_loader = get_dataloaders(char, batch_size=32, num_workers=0)

        batch, labels = next(iter(train_loader))
        print(f"\nBatch shape: {batch.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Batch range: [{batch.min():.2f}, {batch.max():.2f}]")
        print(f"Positive samples: {labels.sum()}/{len(labels)}")

    print("\nOK: All datasets tested successfully!")
