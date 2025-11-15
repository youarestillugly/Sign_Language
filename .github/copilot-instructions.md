# AI Coding Agent Instructions for Deep Learning Project

## Project Overview
ASL Alphabet Classification using TensorFlow/Keras - a deep learning project for recognizing American Sign Language alphabet characters from image data.

**Tech Stack:** TensorFlow/Keras, Python 3.9, GPU-enabled environment (Miniconda: RinzinDema)

## Critical Setup & Environment

- **Python Environment:** `RinzinDema` conda environment at `/home/heru/miniconda3/envs/RinzinDema/`
- **Data Directory:** `/home/heru/miniconda3/envs/RinzinDema/deep_project/Dataset/` (currently empty - needs population)
- **Original Data Source:** Google Drive paths used in notebook (requires local adaptation for development)
- **GPU Support:** Code checks for available GPUs via `tf.config.list_physical_devices('GPU')`

## Development Workflow

1. **Data Preparation:**
   - Expected structure: `Dataset/asl_alphabet_train/` and `Dataset/asl_alphabet_test/` directories
   - Each subdirectory contains class folders (A-Z + space) with images
   - Images are preprocessed to 224×224 pixels (see `img_height`, `img_width` constants)
   - Batch size: 32 (tunable in notebook cell 5)

2. **Model Development Pattern:**
   - Use `tf.keras.preprocessing.image_dataset_from_directory()` for data loading
   - Apply 80/20 train/validation split with `validation_split=0.2`
   - Categorical labels with `label_mode='categorical'` for multi-class classification
   - Use seed=123 for reproducible train/test splits

3. **Running the Notebook:**
   - Execute cells sequentially - GPU detection (cell 2) should run first
   - Cell 4 configures directory paths - **adapt path for your environment**
   - Cell 6 loads training data - ensure Dataset directories exist before running

## Key Patterns & Conventions

- **Path Configuration:** Hardcoded paths in notebook cells (cells 4, 5) - modify for local/remote environments
- **Batch Processing:** Batch size and image dimensions defined as constants (cell 5) for easy tuning
- **Data Augmentation:** `ImageDataGenerator` imported but may not be actively used yet - available for regularization
- **Categorical Classification:** 26 letters + space (27 classes total, inferred from directory structure)

## Common Issues & Solutions

- **"Could not find directory" Error:** Notebook uses Google Drive paths not available locally
  - Solution: Update `base_dir` in cell 4 to point to local Dataset directory
  - Example: `base_dir = '/home/heru/miniconda3/envs/RinzinDema/deep_project/Dataset'`
- **GPU Not Detected:** Cell 2 verifies GPU availability
  - Check: `gpus = tf.config.list_physical_devices('GPU')` returns non-empty list
- **Dataset Missing:** Populate `Dataset/asl_alphabet_train` and `Dataset/asl_alphabet_test` before running training

## File Structure to Know

```
deep_project/
├── Dataset/                    # Image data (subdirs: asl_alphabet_train, asl_alphabet_test)
├── Untitled-1.ipynb           # Main model development notebook
└── .github/
    └── copilot-instructions.md
```

## Next Development Phases

- Model architecture definition (CNN with transfer learning recommended for image classification)
- Validation dataset loading (complement current training setup)
- Training loop with accuracy/loss tracking
- Model evaluation on test set
- Consider: Data augmentation, callbacks (early stopping), fine-tuning strategies
