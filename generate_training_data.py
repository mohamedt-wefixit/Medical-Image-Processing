#!/usr/bin/env python3
"""
Generate synthetic training data from MM-WHS MR dataset
"""

import os
import sys
import yaml
from pathlib import Path
import subprocess
import logging

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_training_config():
    """Create optimized configuration for training data generation"""
    training_config = {
        'output_dir': 'synthetic_training_data',
        'log_level': 'INFO',
        'transformations': {
            'rotation_range': [-45, 45],  # More realistic range for medical images
            'translation_range': [-20, 20],  # Reduced translation range
            'variants_per_image': 50,  # More variants for better training
            'min_dof_modified': 1,
            'max_dof_modified': 3  # Focus on 1-3 DOF changes for easier learning
        },
        'dataset': {
            'extensions': ['.nii', '.nii.gz'],
            'batch_size': 5,
            'validate_outputs': True
        },
        'sample': {
            'enabled': False,  # Don't create sample data
            'dimensions': [128, 128, 64],
            'voxel_size': [1.0, 1.0, 1.0],
            'add_noise': False,
            'noise_level': 0.05
        }
    }
    
    config_path = 'training_data_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(training_config, f, default_flow_style=False, indent=2)
    
    return config_path

def main():
    logger = setup_logging()
    
    # Check if MM-WHS MR training data exists
    mr_train_dir = Path("MM-WHS 2017 Dataset/mr_train")
    if not mr_train_dir.exists():
        logger.error(f"MR training directory not found: {mr_train_dir}")
        logger.error("Please ensure the MM-WHS 2017 Dataset is properly extracted")
        sys.exit(1)
    
    # Find MR image files (not labels)
    mr_image_files = list(mr_train_dir.glob("*_image.nii.gz"))
    
    if not mr_image_files:
        logger.error("No MR image files found in training directory")
        sys.exit(1)
    
    logger.info(f"Found {len(mr_image_files)} MR training images")
    
    # Create training configuration
    config_path = create_training_config()
    logger.info(f"Created training configuration: {config_path}")
    
    # Generate synthetic data for each MR training image
    logger.info("Starting synthetic data generation...")
    
    try:
        # Run the medical image generator for MR training data
        # Only process the image files (not labels) by passing them individually
        cmd = [
            sys.executable, "medical_image_generator.py",
            "--config", config_path
        ]
        
        # Add only image files (exclude label files)
        for img_file in mr_image_files:
            cmd.append(str(img_file))
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Synthetic data generation completed successfully!")
            logger.info("Output:")
            logger.info(result.stdout)
        else:
            logger.error("Synthetic data generation failed!")
            logger.error("Error output:")
            logger.error(result.stderr)
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Failed to run synthetic data generation: {e}")
        sys.exit(1)
    
    # Verify output
    output_dir = Path("synthetic_training_data")
    if output_dir.exists():
        variant_files = list(output_dir.glob("*_variant_*.nii.gz"))
        metadata_files = list(output_dir.glob("*_metadata.json"))
        
        logger.info(f"Generated {len(variant_files)} variant images")
        logger.info(f"Generated {len(metadata_files)} metadata files")
        
        if len(variant_files) > 0 and len(metadata_files) > 0:
            logger.info("✅ Synthetic training data generation successful!")
            logger.info(f"Data saved to: {output_dir}")
            logger.info("You can now run: python dof_regressor_model.py --metadata-dir synthetic_training_data")
        else:
            logger.warning("⚠️ Synthetic data generation completed but no output files found")
    else:
        logger.error("❌ Output directory not created")

if __name__ == "__main__":
    main() 