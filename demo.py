#!/usr/bin/env python3
"""
Demo script for the 6-DOF regression model.
This script loads the trained model and makes predictions on new images.
"""

import os
import sys
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Union, Tuple, List, Dict
from dof_regressor_model import DOF3DCNN, preprocess_image

def enforce_ras_orientation(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Enforce RAS+ orientation (Right-Anterior-Superior) on the image.
    This is the standard orientation for neuroimaging.
    
    Args:
        img: Input NIfTI image
        
    Returns:
        NIfTI image with RAS+ orientation
    """
    # Get current orientation
    orig_ornt = nib.io_orientation(img.affine)
    
    # Target orientation is RAS (Right, Anterior, Superior)
    # The corresponding codes in nibabel are (L->R, P->A, I->S) = (0, 1, 2)
    # with sign (1, 1, 1) for RAS+
    target_ornt = np.array([[0, 1], [1, 1], [2, 1]])
    
    # Find the transform from current to target orientation
    transform = nib.orientations.ornt_transform(orig_ornt, target_ornt)
    
    # Apply the transform
    reoriented_data = nib.orientations.apply_orientation(img.get_fdata(), transform)
    
    # Create new affine for the reoriented data
    affine = img.affine.dot(nib.orientations.inv_ornt_aff(transform, img.shape))
    
    # Create new image with reoriented data and updated affine
    reoriented_img = nib.Nifti1Image(reoriented_data, affine, img.header)
    
    print(f"Enforced RAS+ orientation. Original orientation: {orig_ornt}")
    
    return reoriented_img

def load_trained_model(checkpoint_path: str) -> torch.nn.Module:
    """
    Load a trained model from a checkpoint file
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Trained model
    """
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 
                         'cpu')
    
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration
    input_size = checkpoint.get('input_size', (256, 256, 128))
    dropout_rate = checkpoint.get('dropout_rate', 0.3)
    
    # Create model with same architecture
    model = DOF3DCNN(input_size=input_size, dropout_rate=dropout_rate).to(device)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from checkpoint at epoch {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    
    return model

def predict_6dof(model: torch.nn.Module, image_path: str) -> np.ndarray:
    """
    Predict 6-DOF parameters for a given image
    
    Args:
        model: Trained model
        image_path: Path to the NIfTI image file
        
    Returns:
        Predicted 6-DOF parameters [rx, ry, rz, tx, ty, tz]
    """
    # Load image
    img = nib.load(image_path)
    
    # Enforce RAS orientation
    img = enforce_ras_orientation(img)
    
    # Preprocess image
    input_tensor = preprocess_image(img, model.input_size)
    
    # Move to same device as model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
    
    # Convert to numpy array
    prediction = output.cpu().numpy()[0]
    
    return prediction

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python demo.py <path_to_image.nii.gz>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    checkpoint_path = 'checkpoints/best_checkpoint.pth'
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
        
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    
    # Load model
    model = load_trained_model(checkpoint_path)
    
    # Make prediction
    prediction = predict_6dof(model, image_path)
    
    # Print results
    print("\nPredicted 6-DOF parameters:")
    print(f"Rotation (degrees): [{prediction[0]:.2f}, {prediction[1]:.2f}, {prediction[2]:.2f}]")
    print(f"Translation (mm): [{prediction[3]:.2f}, {prediction[4]:.2f}, {prediction[5]:.2f}]")

if __name__ == "__main__":
    main() 