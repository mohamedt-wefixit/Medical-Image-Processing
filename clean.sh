#!/bin/bash
# Cleanup script for Medical Image Synthetic Dataset Generator

echo "Cleaning up generated files..."

# Remove generated datasets
rm -rf synthetic_dataset/
rm -rf real_*/

# Remove test files
rm -f sample_brain.nii.gz
rm -f *.log

# Remove Python cache
rm -rf __pycache__/

echo "Cleanup complete. System is ready for fresh use."
echo "Run 'python3 start.py' to get started." 