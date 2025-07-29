#!/usr/bin/env python3
"""
Medical Image Synthetic Dataset Generator - Setup Script
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header():
    """Print the startup header."""
    print("=" * 60)
    print("MEDICAL IMAGE SYNTHETIC DATASET GENERATOR")
    print("=" * 60)
    print("Generate training data by modifying orientation parameters")
    print("in NIfTI medical imaging files.\n")


def check_environment():
    """Check if the environment is set up correctly."""
    print("Checking environment...")
    
    # Check if virtual environment exists
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Virtual environment not found. Creating...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("Virtual environment created.")
    else:
        print("Virtual environment found.")
    
    # Check if dependencies are installed
    print("Installing/checking dependencies...")
    subprocess.run([
        "venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"
    ], check=True, capture_output=True)
    print("Dependencies ready.")
    print()


def show_usage_examples():
    """Show usage examples."""
    print("USAGE EXAMPLES:")
    print("-" * 20)
    print()
    
    print("1. Create sample data for testing:")
    print("   source venv/bin/activate")
    print("   python synthetic_dataset_generator.py --create-sample")
    print()
    
    print("2. Process your own NIfTI file:")
    print("   source venv/bin/activate")
    print("   python synthetic_dataset_generator.py your_brain_scan.nii.gz")
    print()
    
    print("3. Process multiple files with custom output:")
    print("   source venv/bin/activate")
    print("   python synthetic_dataset_generator.py --output-dir my_dataset file1.nii file2.nii.gz")
    print()
    
    print("4. Quick test with sample data:")
    print("   ./quick_test.sh")
    print()


def create_quick_test_script():
    """Create a quick test script."""
    script_content = """#!/bin/bash
# Quick Test Script for Medical Image Synthetic Dataset Generator

echo "Starting quick test..."

# Activate virtual environment
source venv/bin/activate

# Create sample data
echo "Creating sample NIfTI file..."
python synthetic_dataset_generator.py --create-sample

# Generate synthetic dataset
echo "Generating synthetic dataset..."
python synthetic_dataset_generator.py sample_brain.nii.gz

# Show results
echo ""
echo "Test complete. Results:"
echo "Generated files:"
ls -la synthetic_dataset/ | head -10
echo ""
echo "Total variants created:"
ls synthetic_dataset/*.nii.gz | wc -l
echo ""
echo "Synthetic dataset generation successful."
"""
    
    with open("quick_test.sh", "w") as f:
        f.write(script_content)
    
    # Make executable
    os.chmod("quick_test.sh", 0o755)
    print("Created quick_test.sh script")


def show_file_structure():
    """Show the expected file structure."""
    print("PROJECT STRUCTURE:")
    print("-" * 20)
    print("ai medical images/")
    print("├── synthetic_dataset_generator.py  # Main generator")
    print("├── requirements.txt               # Dependencies")
    print("├── start.py                      # This startup script")
    print("├── quick_test.sh                 # Quick test script")
    print("├── venv/                         # Virtual environment")
    print("└── your_nifti_files.nii.gz      # Your medical images")
    print()


def main():
    """Main startup function."""
    print_header()
    
    try:
        check_environment()
        create_quick_test_script()
        show_file_structure()
        show_usage_examples()
        
        print("READY TO GO")
        print("=" * 60)
        print("Place your NIfTI files in this directory and run the commands above.")
        print("For help: python synthetic_dataset_generator.py --help")
        print()
        
    except Exception as e:
        print(f"Error during setup: {e}")
        print("Please check your Python installation and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main() 