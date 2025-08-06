#!/usr/bin/env python3
"""
Test script for RAS orientation enforcement.
"""

import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path
from medical_image_generator import MedicalImageProcessor

def test_ras_orientation():
    """Test RAS orientation enforcement"""
    print("Testing RAS orientation enforcement...")
    
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
    
    try:
        # Load the image
        img = nib.load(image_path)
        print(f"Original image shape: {img.shape}")
        
        # Get original orientation
        orig_ornt = nib.io_orientation(img.affine)
        print(f"Original orientation: {orig_ornt}")
        
        # Create processor
        config = {
            'output_dir': 'test_output',
            'log_level': 'INFO'
        }
        processor = MedicalImageProcessor(config=config)
        
        # Enforce RAS orientation
        ras_img = processor.enforce_ras_orientation(img)
        print(f"RAS image shape: {ras_img.shape}")
        
        # Check new orientation
        new_ornt = nib.io_orientation(ras_img.affine)
        print(f"New orientation: {new_ornt}")
        
        # Verify it's RAS
        target_ornt = np.array([[0, 1], [1, 1], [2, 1]])
        is_ras = np.allclose(new_ornt, target_ornt)
        print(f"Is RAS orientation: {is_ras}")
        
        # Save the RAS image for inspection
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "test_ras.nii.gz"
        nib.save(ras_img, output_path)
        print(f"Saved RAS image to: {output_path}")
        
        # Calculate affine matrix
        print("\nAffine matrices:")
        print("Original affine:")
        print(img.affine)
        print("\nRAS affine:")
        print(ras_img.affine)
        
        return is_ras
    
    except Exception as e:
        print(f"Error testing RAS orientation: {e}")
        return False

if __name__ == "__main__":
    success = test_ras_orientation()
    if success:
        print("\nRAS orientation enforcement works correctly!")
    else:
        print("\nRAS orientation test failed!") 