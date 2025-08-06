#!/usr/bin/env python3
"""
Test script for affine matrix calculation and correction.
This demonstrates how to calculate the affine matrix and enforce RAS orientation.
"""

import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def calculate_affine_from_6dof(rotation_angles, translation, reference_affine=None):
    """
    Calculate affine matrix from 6-DOF parameters
    
    Args:
        rotation_angles: [rx, ry, rz] in degrees
        translation: [tx, ty, tz] in mm
        reference_affine: Optional reference affine matrix for scaling
        
    Returns:
        4x4 affine matrix
    """
    # Convert rotation angles to rotation matrix
    r = R.from_euler('xyz', rotation_angles, degrees=True)
    rotation_matrix = r.as_matrix()
    
    # Create affine matrix
    affine = np.eye(4)
    
    if reference_affine is not None:
        # Extract scales from reference affine
        scales = np.sqrt(np.sum(reference_affine[:3, :3]**2, axis=0))
        # Apply scales to rotation matrix
        scaled_rotation = rotation_matrix * scales
        affine[:3, :3] = scaled_rotation
    else:
        affine[:3, :3] = rotation_matrix
    
    # Set translation
    affine[:3, 3] = translation
    
    return affine

def extract_6dof_from_affine(affine):
    """
    Extract 6-DOF parameters from affine matrix
    
    Args:
        affine: 4x4 affine matrix
        
    Returns:
        rotation_angles: [rx, ry, rz] in degrees
        translation: [tx, ty, tz] in mm
    """
    # Extract rotation and scaling
    rotation_scaling = affine[:3, :3]
    translation = affine[:3, 3]
    
    # Extract scales
    scales = np.sqrt(np.sum(rotation_scaling**2, axis=0))
    
    # Normalize to get rotation matrix
    rotation_matrix = rotation_scaling / scales
    
    # Handle negative determinant
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, 0] *= -1
        scales[0] *= -1
    
    # Convert to Euler angles
    try:
        r = R.from_matrix(rotation_matrix)
        rotation_angles = r.as_euler('xyz', degrees=True)
    except ValueError:
        print("Warning: Invalid rotation matrix, using identity")
        rotation_angles = np.array([0.0, 0.0, 0.0])
    
    return rotation_angles, translation

def enforce_ras_orientation(img):
    """
    Enforce RAS+ orientation on a NIfTI image
    
    Args:
        img: Input NIfTI image
        
    Returns:
        NIfTI image with RAS+ orientation
    """
    # Get current orientation
    orig_ornt = nib.io_orientation(img.affine)
    print(f"Original orientation: {orig_ornt}")
    
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
    
    return reoriented_img

def test_affine_calculation():
    """Test affine matrix calculation and correction"""
    print("Testing affine matrix calculation and RAS orientation...")
    
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
        
        # Get original affine
        original_affine = img.affine
        print("\nOriginal affine matrix:")
        print(original_affine)
        
        # Extract 6-DOF from original affine
        original_rotation, original_translation = extract_6dof_from_affine(original_affine)
        print(f"\nExtracted 6-DOF from original affine:")
        print(f"Rotation (degrees): {original_rotation}")
        print(f"Translation (mm): {original_translation}")
        
        # Enforce RAS orientation
        ras_img = enforce_ras_orientation(img)
        print(f"\nRAS image shape: {ras_img.shape}")
        
        # Get RAS affine
        ras_affine = ras_img.affine
        print("\nRAS affine matrix:")
        print(ras_affine)
        
        # Extract 6-DOF from RAS affine
        ras_rotation, ras_translation = extract_6dof_from_affine(ras_affine)
        print(f"\nExtracted 6-DOF from RAS affine:")
        print(f"Rotation (degrees): {ras_rotation}")
        print(f"Translation (mm): {ras_translation}")
        
        # Create a new affine from 6-DOF
        new_affine = calculate_affine_from_6dof(ras_rotation, ras_translation, ras_affine)
        print("\nRecreated affine matrix from 6-DOF:")
        print(new_affine)
        
        # Check if the recreated affine is close to the RAS affine
        is_close = np.allclose(new_affine, ras_affine, rtol=1e-5, atol=1e-5)
        print(f"\nRecreated affine matches RAS affine: {is_close}")
        
        # Save the RAS image for inspection
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "test_affine_ras.nii.gz"
        nib.save(ras_img, output_path)
        print(f"\nSaved RAS image to: {output_path}")
        
        return is_close
    
    except Exception as e:
        print(f"Error testing affine calculation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_affine_calculation()
    if success:
        print("\nAffine matrix calculation and RAS orientation works correctly!")
    else:
        print("\nAffine matrix calculation test failed!") 