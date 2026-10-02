"""
Centralized path configuration for the project
All paths relative to project root
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Models
MODELS_DIR = PROJECT_ROOT / 'models'
TASK1_MODEL = MODELS_DIR / 'resnet18_face_detector.pth'

TASK2_MODELS_DIR = PROJECT_ROOT / 'src' / 'task2' / 'one_vs_all' / 'models'
TASK2_MODELS = {
    'daphne': TASK2_MODELS_DIR / 'resnet18_daphne_weighted.pth',
    'fred': TASK2_MODELS_DIR / 'resnet18_fred_weighted.pth',
    'shaggy': TASK2_MODELS_DIR / 'resnet18_shaggy_weighted.pth',
    'velma': TASK2_MODELS_DIR / 'resnet18_velma_weighted.pth',
}

# Processed Data
PROCESSED_DATA_DIR = PROJECT_ROOT / 'processed_data'

# Task 1 data
TASK1_POSITIVE_PATCHES = PROCESSED_DATA_DIR / 'positive_patches.npy'
TASK1_NEGATIVE_PATCHES = PROCESSED_DATA_DIR / 'negative_patches.npy'
TASK1_HARD_NEGATIVES = PROCESSED_DATA_DIR / 'hard_negative_patches_combined.npy'

# Task 2 data
def get_task2_data_paths(character):
    """Get data paths for a specific character"""
    return {
        'positive': PROCESSED_DATA_DIR / f'{character}_positive.npy',
        'negative': PROCESSED_DATA_DIR / f'{character}_negative.npy',
        'hard_negatives': PROCESSED_DATA_DIR / f'{character}_hard_negatives_pipeline.npy',
    }

# Results directories
RESULTS_DIR = PROJECT_ROOT / 'rezultate'
TASK1_RESULTS_DIR = RESULTS_DIR / 'task1'
TASK2_RESULTS_DIR = RESULTS_DIR / 'task2'

# Data directories (for training/validation)
DATA_ROOT = PROJECT_ROOT.parent  # Go up to CAVA-2025-TEMA2
TRAIN_DIR = DATA_ROOT / 'antrenare'
VAL_DIR = DATA_ROOT / 'validare'

# Character directories
CHARACTERS = ['daphne', 'fred', 'shaggy', 'velma']

def get_character_train_dir(character):
    """Get training directory for a character"""
    return TRAIN_DIR / character

def get_character_annotations(character):
    """Get annotations file for a character"""
    return TRAIN_DIR / f'{character}_annotations.txt'

# Task 1 ground truth
TASK1_GT_TRAIN = TRAIN_DIR / 'task1_gt_antrenare.txt'
TASK1_GT_VAL = VAL_DIR / 'task1_gt_validare.txt'

# Validation images
VAL_IMAGES_DIR = VAL_DIR / 'validare'

# Create directories if they don't exist
def ensure_directories():
    """Create necessary directories if they don't exist"""
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    TASK2_MODELS_DIR.mkdir(exist_ok=True, parents=True)
    PROCESSED_DATA_DIR.mkdir(exist_ok=True, parents=True)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    TASK1_RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    TASK2_RESULTS_DIR.mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    # Print all paths for verification
    print("=== PROJECT PATHS ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"\nTask 1 model: {TASK1_MODEL}")
    print(f"\nTask 2 models:")
    for char, path in TASK2_MODELS.items():
        print(f"  {char}: {path}")
    print(f"\nProcessed data dir: {PROCESSED_DATA_DIR}")
    print(f"\nValidation images: {VAL_IMAGES_DIR}")
