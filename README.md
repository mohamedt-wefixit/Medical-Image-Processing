# Medical Image Dataset Generator

Generate synthetic training data from NIfTI medical images using realistic random transformations.

## Usage

```bash
python medical_image_generator.py --setup
python medical_image_generator.py --sample
python medical_image_generator.py your_file.nii.gz
python medical_image_generator.py --directory "MM-WHS 2017 Dataset"
```

## Features

- Single file solution
- YAML configuration
- Realistic mode with random transformations
- Configurable rotation and translation ranges
- Random DOF (degrees of freedom) modification per variant
- Batch processing
- Output validation

## Configuration

Edit `config.yaml` to customize parameters:

- `rotation_range`: Range for random rotations (degrees)
- `translation_range`: Range for random translations (mm) 
- `variants_per_image`: Number of variants to generate per input
- `min_dof_modified`: Minimum DOF to modify per variant
- `max_dof_modified`: Maximum DOF to modify per variant

## Realistic Mode

The generator creates realistic transformations by:
- Randomly selecting 1-6 DOF to modify per variant
- Applying random rotations and translations within specified ranges
- Each variant modifies a different combination of axes
- Preserves original image characteristics while creating diversity

## Dataset Processing

```bash
python medical_image_generator.py --directory "MM-WHS 2017 Dataset"
```

Processes all `.nii` and `.nii.gz` files automatically. 