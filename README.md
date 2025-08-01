# 6-DOF Medical Image Registration Model

A deep learning model for predicting 6 degrees of freedom (3 rotations, 3 translations) from transformed medical images.

## Quick Start

### Test the Pre-trained Model
```bash
python test_model.py
```

### Train on Your Local Machine
```bash
# Generate training data
python generate_training_data.py

# Train the model
python dof_regressor_model.py --metadata-dir synthetic_training_data
```

### Train on Google Colab (Recommended)
1. Upload `Google_Colab_Training.ipynb` to Google Colab
2. Run the first cell to check GPU availability
3. Run the installation cell
4. Run the remaining cells in order
5. Download the trained model (`best_checkpoint.pth`)

The notebook contains detailed instructions and will guide you through each step.

## File Descriptions

### Core Files
- **medical_image_generator.py**: Transforms medical images with known 6-DOF changes
- **generate_training_data.py**: Creates synthetic training data from MR images
- **dof_regressor_model.py**: Defines and trains the 3D CNN model
- **train_dof_model.py**: Master script for the entire pipeline

### Testing & Evaluation
- **demo.py**: Simple demonstration of model usage
- **test_model.py**: Tests model with known transformations
- **evaluate_model.py**: Comprehensive model evaluation

### Configuration
- **config.yaml**: Settings for data generation
- **training_config.yaml**: Settings for model training
- **requirements.txt**: Python dependencies

### Pre-trained Model
- **checkpoints/best_checkpoint.pth**: Ready-to-use model

## Performance

The pre-trained model achieves:
- Rotation accuracy: ~7-11 degrees
- Translation accuracy: ~2-5 mm

With GPU training on full resolution (128×128×64), you can expect:
- Rotation accuracy: ~5-7 degrees
- Translation accuracy: ~1-2 mm

## Configuration Options

### Model & Training (`training_config.yaml`)
- **Resolution**: Adjust `input_size` and `target_size`
- **Training Duration**: Change `num_epochs` (recommend 50-100)
- **Batch Size**: Increase `batch_size` to 4-8 on GPU

### Data Generation (`config.yaml`)
- **Transformation Range**: Adjust `rotation_range` and `translation_range`
- **Data Volume**: Change `variants_per_image` (recommend 50-100)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- nibabel
- scipy
- numpy
- pyyaml

Install dependencies with:
```bash
pip install -r requirements.txt
``` 