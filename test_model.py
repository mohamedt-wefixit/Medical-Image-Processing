#!/usr/bin/env python3
"""
Simple test script for the 6-DOF regression model.
This script loads the trained model and tests it on a sample image.
"""

import os
import numpy as np
import nibabel as nib
import torch
from pathlib import Path
from demo import load_trained_model, predict_6dof
from medical_image_generator import MedicalImageProcessor

def test_on_image(image_path):
    """Test the model on a single image with known transformation"""
    print(f"Testing model on image: {image_path}")
    
    # Load model
    model = load_trained_model('checkpoints/best_checkpoint.pth')
    
    # Create a processor to apply known transformations
    config = {
        'output_dir': 'test_output',
        'log_level': 'INFO',
        'transformations': {
            'rotation_range': [-20, 20],
            'translation_range': [-10, 10]
        }
    }
    processor = MedicalImageProcessor(config=config)
    
    # Load the original image
    img = nib.load(image_path)
    original_data = img.get_fdata()
    original_affine = img.affine
    
    # Extract original 6-DOF
    original_dof = processor.extract_6dof(original_affine)
    print(f"Original 6-DOF: {original_dof}")
    
    # Apply a known transformation
    rotation = [10.0, -5.0, 15.0]  # degrees
    translation = [5.0, -3.0, 7.0]  # mm
    
    # Create new affine with the transformation
    new_affine = processor.create_affine(
        rotation_angles=np.array(rotation),
        translation=np.array(translation),
        original_affine=original_affine
    )
    
    # Create transformed image
    transformed_img = nib.Nifti1Image(original_data, new_affine)
    transformed_file = "test_transformed.nii.gz"
    nib.save(transformed_img, transformed_file)
    
    print(f"Applied transformation:")
    print(f"  Rotation: {rotation}")
    print(f"  Translation: {translation}")
    
    # Predict 6-DOF with our model
    predicted_dof = predict_6dof(model, transformed_file)
    
    print(f"\nPredicted 6-DOF:")
    print(f"  Rotation: {predicted_dof[:3].tolist()}")
    print(f"  Translation: {predicted_dof[3:].tolist()}")
    
    # Calculate error
    error = np.abs(np.array(rotation + translation) - predicted_dof)
    print(f"\nAbsolute Error:")
    print(f"  Rotation (degrees): {error[:3].tolist()}")
    print(f"  Translation (mm): {error[3:].tolist()}")
    print(f"  Mean Error: {np.mean(error):.2f}")
    
    # Clean up
    if os.path.exists(transformed_file):
        os.remove(transformed_file)

if __name__ == "__main__":
    # Check if MM-WHS dataset is available
    mr_train_dir = Path("MM-WHS 2017 Dataset/mr_train")
    
    if mr_train_dir.exists():
        # Find an image file
        image_files = list(mr_train_dir.glob("*_image.nii.gz"))
        if image_files:
            test_image = str(image_files[0])
            test_on_image(test_image)
        else:
            print("No image files found in MM-WHS 2017 Dataset/mr_train")
            print("Please provide a path to a NIfTI image file as an argument")
    else:
        print("MM-WHS dataset not found.")
        print("Usage: python test_model.py [path_to_image.nii.gz]")
        print("If no image path is provided, the script will look for the MM-WHS dataset.") 