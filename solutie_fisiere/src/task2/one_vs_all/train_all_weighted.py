"""
Train ALL Characters with Sample Weighting for Hard Negatives
Based on Fred's success: 25% → 77% AP (+52%)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, 'src/task1/resnet')
sys.path.insert(0, 'src/task2/one_vs_all')

from model import create_model
from dataset import BinaryCharacterDataset
from tqdm import tqdm


def train_character_weighted(character):
    """Train one character with sample weighting for HN"""

    print("="*60)
    print(f"TRAINING {character.upper()} WITH SAMPLE WEIGHTING")
    print("="*60)

    device = 'mps'
    batch_size = 128
    num_epochs = 30
    learning_rate = 0.001

    print(f"\n1. Loading {character} dataset...")

    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3)
    ])

    train_dataset = BinaryCharacterDataset(
        character=character,
        split='train',
        transform=train_transform,
        use_hnm=True
    )

    val_dataset = BinaryCharacterDataset(
        character=character,
        split='val',
        transform=None,
        use_hnm=True
    )

    print("\n2. Calculating sample weights...")
    print("   Strategy: HN get 2.5x weight")

    train_size = len(train_dataset)

    project_root = Path(__file__).parent.parent.parent.parent
    processed_dir = project_root / 'processed_data'
    positives = np.load(processed_dir / f'{character}_positive.npy')
    negatives = np.load(processed_dir / f'{character}_negative.npy')

    hn_file = Path('processed_data') / f'{character}_hard_negatives_pipeline.npy'
    if hn_file.exists():
        hard_negatives = np.load(hn_file)
    else:
        hard_negatives = np.array([])

    print(f"   {len(positives)} pos, {len(negatives)} neg, {len(hard_negatives)} HN")

    # Create weights for full dataset
    full_weights = []
    full_weights.extend([1.0] * len(positives))
    full_weights.extend([1.0] * len(negatives))
    full_weights.extend([2.5] * len(hard_negatives))

    full_weights = np.array(full_weights)

    # Apply same split as dataset
    np.random.seed(42)
    indices = np.random.permutation(len(full_weights))
    split_idx = int(0.9 * len(indices))
    train_indices = indices[:split_idx]

    sample_weights = full_weights[train_indices]

    print(f"   HN weight: {(sample_weights == 2.5).sum() / len(sample_weights) * 100:.1f}%")

    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"\n3. Creating {character} model...")
    model, _ = create_model(num_classes=2, device=device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"\n4. Training for {num_epochs} epochs...")
    best_val_acc = 0

    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.3f}',
                            'acc': f'{100.*train_correct/train_total:.2f}%'})

        train_acc = 100. * train_correct / train_total

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total

        print(f"Epoch {epoch+1}: Train {train_acc:.2f}%, Val {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = Path(f'src/task2/one_vs_all/models/task2/resnet18_{character}_weighted.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch,
                'train_acc': train_acc
            }, save_path)
            print(f"   OK: Saved (Val: {val_acc:.2f}%)")

        scheduler.step()

    print(f"\nOK: {character.upper()} DONE! Best Val Acc: {best_val_acc:.2f}%\n")
    return best_val_acc


if __name__ == "__main__":
    print("="*60)
    print("TRAINING ALL CHARACTERS WITH SAMPLE WEIGHTING")
    print("="*60)

    characters = ['daphne', 'shaggy', 'velma', 'fred']

    for char in characters:
        train_character_weighted(char)

    print("\n" + "="*60)
    print("OK: ALL CHARACTERS TRAINED!")
    print("="*60)
