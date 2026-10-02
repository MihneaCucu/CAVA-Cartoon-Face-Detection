"""
Train Binary ResNet Model for One Character (One-vs-All)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
from datetime import datetime

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent / 'task1' / 'resnet'))
sys.path.insert(0, str(current_dir))

from model import create_model
import dataset as local_dataset


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.5f}',
            'acc': f'{100.*correct/total:.1f}'
        })

    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss/len(dataloader):.5f}',
                'acc': f'{100.*correct/total:.1f}'
            })

    return running_loss / len(dataloader), 100. * correct / total


def main(args):
    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = output_dir / f'training_{args.character}_{timestamp}.log'

    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_file)

    print("="*60)
    print(f"TRAINING BINARY RESNET - {args.character.upper()}")
    print(f"Log file: {log_file}")
    print("="*60)

    model, device = create_model(num_classes=2, device=args.device)


    print(f"\nLoading {args.character} dataset...")
    train_loader, val_loader = local_dataset.get_dataloaders(
        args.character,
        batch_size=args.batch_size,
        num_workers=0,
        use_hnm=args.use_hnm
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {device}")
    print("="*60)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            checkpoint_dir = Path('models/task2')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_dir / f'resnet18_{args.character}.pth')

            print(f"  OK: New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nWARNING: Early stopping triggered (patience={args.patience})")
                break

        print("-"*60)

    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Character: {args.character.upper()}")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved: models/task2/resnet18_{args.character}.pth")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train binary ResNet for character detection')
    parser.add_argument('--character', type=str, required=True,
                       choices=['daphne', 'fred', 'shaggy', 'velma'],
                       help='Character to train')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs (reduced to prevent overfitting)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                       help='Weight decay (increased for regularization)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use')
    parser.add_argument('--use-hnm', action='store_true',
                       help='Use hard negative mining')

    args = parser.parse_args()
    main(args)
