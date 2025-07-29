#!/bin/bash
# Quick Test Script for Medical Image Synthetic Dataset Generator

echo "🧠 Starting quick test..."

# Activate virtual environment
source venv/bin/activate

# Create sample data
echo "📊 Creating sample NIfTI file..."
python synthetic_dataset_generator.py --create-sample

# Generate synthetic dataset
echo "🔄 Generating synthetic dataset..."
python synthetic_dataset_generator.py sample_brain.nii.gz

# Show results
echo ""
echo "✅ Test complete! Results:"
echo "📁 Generated files:"
ls -la synthetic_dataset/ | head -10
echo ""
echo "📊 Total variants created:"
ls synthetic_dataset/*.nii.gz | wc -l
echo ""
echo "🎉 Synthetic dataset generation successful!"
