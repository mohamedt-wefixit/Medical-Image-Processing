# Medical Image Synthetic Dataset Generator

Generates synthetic training datasets by modifying orientation and position parameters in NIfTI medical imaging files.

## Setup and Usage

```bash
# 1. First time setup
python3 start.py

# 2. Activate environment  
source venv/bin/activate

# 3. Process your files
python synthetic_dataset_generator.py your_file.nii.gz
```

## Quick Test

```bash
./quick_test.sh
```

## Process Multiple Files

```bash
python synthetic_dataset_generator.py *.nii.gz *.nii
```

Creates 33 synthetic variants per input file with different orientations and translations. 