"""
Create Binary Datasets for One-vs-All Task 2 Strategy
Splits existing character patches into 4 binary datasets
"""

import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
patches_dir = project_root / 'processed_data' / 'patches'
output_dir = project_root / 'processed_data'
CHARACTERS = ['daphne', 'fred', 'shaggy', 'velma']

print("="*60)
print("CREATING BINARY DATASETS FOR ONE-VS-ALL")
print("="*60)

all_patches = {}
for char in CHARACTERS:
    patch_file = patches_dir / f'{char}_patches.npy'
    if patch_file.exists():
        patches = np.load(patch_file)
        all_patches[char] = patches
        print(f"OK: Loaded {char}: {len(patches):,} patches")
    else:
        print(f"ERROR: Missing: {patch_file}")

unknown_file = patches_dir / 'unknown_patches.npy'
background_file = patches_dir / 'negative_patches.npy'

if unknown_file.exists():
    all_patches['unknown'] = np.load(unknown_file)
    print(f"OK: Loaded unknown: {len(all_patches['unknown']):,} patches")

if background_file.exists():
    all_patches['background'] = np.load(background_file)
    print(f"OK: Loaded background: {len(all_patches['background']):,} patches")

print("\n" + "="*60)
print("CREATING BINARY DATASETS")
print("="*60)

for target_char in CHARACTERS:
    print(f"\n--- {target_char.upper()} vs REST ---")

    positive = all_patches[target_char]
    print(f"  Positive ({target_char}): {len(positive):,}")

    negative_patches = []
    for char, patches in all_patches.items():
        if char != target_char:
            negative_patches.append(patches)
            print(f"  Negative ({char}): {len(patches):,}")

    negative = np.concatenate(negative_patches)
    print(f"  Total Negative: {len(negative):,}")
    print(f"  Total Dataset: {len(positive) + len(negative):,}")

    np.save(output_dir / f'{target_char}_positive.npy', positive)
    np.save(output_dir / f'{target_char}_negative.npy', negative)
    print(f"  OK: Saved: {target_char}_positive.npy, {target_char}_negative.npy")

print("\n" + "="*60)
print("OK: BINARY DATASETS CREATED!")
print("="*60)
print(f"\nOutput: {output_dir}")
print("\nNext: Train binary models with train_character.py")
print("="*60)
