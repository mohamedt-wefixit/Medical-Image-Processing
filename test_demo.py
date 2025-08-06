#!/usr/bin/env python3
"""
Test script for demo.py with RAS orientation.
"""

import os
import sys
import numpy as np
from pathlib import Path
from demo import load_trained_model, predict_6dof, enforce_ras_orientation

def test_demo():
    """Test demo with RAS orientation"""
    print("Testing demo with RAS orientation...")
    
    # Find a test image
    dataset_dir = Path("MM-WHS 2017 Dataset")
    if dataset_dir.exists():
        # Look for MR images first
        mr_dir = dataset_dir / "mr_train"
        if mr_dir.exists():
            image_files = list(mr_dir.glob("*.nii.gz"))
            if not image_files:
                print("No .nii.gz files found in mr_train directory")
                return False
        else:
            # Look for any NIfTI files in the dataset directory
            image_files = list(dataset_dir.glob("**/*.nii.gz"))
            if not image_files:
                print("No .nii.gz files found in dataset directory")
                return False
    else:
        print("Dataset directory not found")
        return False
    
    # Use the first image file
    image_path = str(image_files[0])
    print(f"Using test image: {image_path}")
    
    # Check for model checkpoint
    checkpoint_path = "checkpoints/best_checkpoint.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        # Load model
        model = load_trained_model(checkpoint_path)
        
        # Make prediction
        prediction = predict_6dof(model, image_path)
        
        # Print results
        print("\nPredicted 6-DOF parameters:")
        print(f"Rotation (degrees): [{prediction[0]:.2f}, {prediction[1]:.2f}, {prediction[2]:.2f}]")
        print(f"Translation (mm): [{prediction[3]:.2f}, {prediction[4]:.2f}, {prediction[5]:.2f}]")
        
        return True
    
    except Exception as e:
        print(f"Error testing demo: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_demo()
    if success:
        print("\nDemo with RAS orientation works correctly!")
    else:
        print("\nDemo test failed!") 