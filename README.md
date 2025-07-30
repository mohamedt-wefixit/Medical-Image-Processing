# Medical Image Dataset Generator

Generate synthetic training data from NIfTI medical images.

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
- Extended rotations up to ±90°
- Translations up to ±20mm
- Batch processing
- Output validation

## Configuration

Edit `config.yaml` to customize parameters.

Generates ~145 variants per input file.

## Dataset Processing

```bash
python medical_image_generator.py --directory "MM-WHS 2017 Dataset"
```

Processes all `.nii` and `.nii.gz` files automatically. 