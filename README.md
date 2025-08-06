# 6-DOF Medical Image Regression

A PyTorch-based deep learning model for estimating 6 degrees of freedom (3 rotations + 3 translations) from medical images using 3D CNN architecture.

## Overview

This project trains a 3D convolutional neural network to predict 6-DOF transformations (Rx, Ry, Rz, Tx, Ty, Tz) applied to medical images. The model uses synthetic training data generated from the MM-WHS 2017 dataset.

## Quick Start (Production Training)

For production training with the complete dataset:

```bash
# One-command full training
./run_full_training.sh
```

This will:
1. Generate training data from ALL medical image cases (~2000 samples)
2. Train the model at full resolution (256×256×128) for up to 200 epochs
3. Test the trained model performance
4. Save production-ready model

## Manual Step-by-Step Training

If you prefer to run steps manually:

```bash
# 1. Generate training data from all cases
python3 generate_training_data.py

# 2. Train the production model
python3 train_production.py --config full_config.yaml

# 3. Test the trained model
python3 test_trained_model.py
```

## System Requirements

- **Hardware**: 16GB+ RAM, GPU recommended (NVIDIA/Apple Silicon)
- **Storage**: 20GB+ free space for full dataset
- **Python**: 3.8+
- **Training Time**: 8-12 hours (with GPU), 24+ hours (CPU only)

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Setup

Place the MM-WHS 2017 Dataset in the project root:
```
MM-WHS 2017 Dataset/
├── mr_train/
│   ├── mr_train_1001_image.nii.gz
│   ├── mr_train_1002_image.nii.gz
│   └── ... (all training images)
└── mr_test/
    └── ... (test images)
```

## Model Architecture

- **Input**: 3D medical images (256×256×128 resolution)
- **Architecture**: 3D CNN with residual connections
- **Output**: 6-DOF vector [Rx, Ry, Rz, Tx, Ty, Tz]
- **Loss**: Mean Squared Error (MSE)
- **Parameters**: ~2.1M trainable parameters

## Configuration

Modify `full_config.yaml` to adjust training parameters:

- `model.input_size`: Image resolution [256, 256, 128]
- `training.batch_size`: Batch size (adjust based on GPU memory)
- `training.num_epochs`: Maximum training epochs (200)
- `training.learning_rate`: Learning rate (0.0001)

Modify `full_training_data_config.yaml` for data generation:
- `transformations.variants_per_image`: Number of variants per image (100)
- `transformations.rotation_range`: Rotation range in degrees (±60°)
- `transformations.translation_range`: Translation range in mm (±25mm)

## Hardware Acceleration

The model automatically detects and uses available hardware acceleration:
- **Apple Silicon (M1/M2/M3)**: Uses MPS (Metal Performance Shaders)
- **NVIDIA GPUs**: Uses CUDA
- **CPU**: Falls back to CPU processing

## Expected Performance (Full Training)

With the complete dataset, expect:
- **Translation Accuracy**: 2-4mm (excellent for medical imaging)
- **Rotation Accuracy**: 5-10° (good for clinical applications)
- **Training Time**: 8-12 hours on modern GPU

## Results and Monitoring

After training:
- **Model checkpoint**: `checkpoints/best_checkpoint.pth`
- **Training logs**: `runs/full_production_training/`
- **TensorBoard**: `tensorboard --logdir=runs/full_production_training`

## File Structure

```
├── README.md                          # This file
├── MODEL_PERFORMANCE_REPORT.md        # Performance analysis
├── requirements.txt                   # Python dependencies
├── full_config.yaml                   # Production training configuration
├── full_training_data_config.yaml     # Data generation configuration
├── run_full_training.sh              # One-command training script
├── train_production.py               # Production training script
├── generate_training_data.py         # Generate full training data
├── test_trained_model.py             # Model evaluation
├── dof_regressor_model.py            # Core model architecture
├── medical_image_generator.py        # Image transformation engine
└── checkpoints/                      # Saved models
```

## Usage Examples

### Production Training
```bash
./run_full_training.sh
```

### Custom Training
```bash
python3 train_production.py --config full_config.yaml --metadata-dir synthetic_training_data
```

### Model Evaluation
```bash
python3 test_trained_model.py
```

### View Training Progress
```bash
tensorboard --logdir=runs/full_production_training
```

## Performance Optimization

### Memory Optimization
- Reduce `batch_size` if you encounter out-of-memory errors
- Adjust `cache_size` based on available RAM
- Use `num_workers: 0` if experiencing multiprocessing issues

### Speed Optimization
- Use GPU acceleration when available
- Increase `batch_size` on high-memory GPUs
- Enable `mixed_precision: true` for compatible hardware

### Training Optimization
- Increase `variants_per_image` for more diverse training data
- Adjust `learning_rate` based on convergence behavior
- Use `early_stopping` to prevent overfitting

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `batch_size` in `full_config.yaml`
2. **Slow Training**: Ensure GPU acceleration is working
3. **Dataset Not Found**: Verify MM-WHS 2017 Dataset location
4. **Poor Performance**: Increase training data or epochs

### Hardware-Specific Tips

**Apple Silicon (M1/M2/M3)**:
- Set `batch_size: 2` for optimal memory usage
- Disable `mixed_precision` for stability

**NVIDIA GPUs**:
- Increase `batch_size` based on VRAM (4-8 for 8GB+ GPUs)
- Enable `mixed_precision: true` for faster training

**CPU Only**:
- Reduce `batch_size: 1` and `input_size` for feasibility
- Expect significantly longer training times

## Clinical Applications

This model is suitable for:
- **Medical Image Registration**: Automated alignment of medical scans
- **Quality Control**: Detecting misaligned or incorrectly positioned images
- **Preprocessing**: Standardizing image orientations before analysis
- **Research**: Quantifying anatomical variations and movements

## RAS Orientation Enforcement

The model now enforces RAS+ orientation (Right-Anterior-Superior) on all input images:

- **Automatic Conversion**: All images are automatically converted to RAS+ orientation before processing
- **Standardized Coordinates**: Ensures consistent coordinate system across different scanners and acquisition protocols
- **Improved Accuracy**: Eliminates orientation-related errors in 6-DOF estimation
- **Clinical Standard**: RAS+ is the standard orientation for neuroimaging

To use this feature directly:

```python
from medical_image_generator import MedicalImageProcessor
import nibabel as nib

# Load image
img = nib.load("your_image.nii.gz")

# Create processor
processor = MedicalImageProcessor(config={})

# Enforce RAS orientation
ras_img = processor.enforce_ras_orientation(img)

# Save RAS-oriented image
nib.save(ras_img, "ras_image.nii.gz")
```

## License

This project is for research and educational purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review TensorBoard logs for training insights
3. Verify dataset and configuration files
4. Monitor system resources during training 