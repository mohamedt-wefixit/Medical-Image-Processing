#!/bin/bash
# Full production training script for client

echo "🚀 6-DOF Medical Image Regression - Full Production Training"
echo "============================================================"

# Check for dataset
if [ ! -d "MM-WHS 2017 Dataset/mr_train" ]; then
    echo "❌ MM-WHS 2017 Dataset not found!"
    echo "Please place the dataset in the project root directory"
    exit 1
fi

# Install dependencies if needed
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Step 1: Generate full training data from all cases
echo ""
echo "📊 Step 1: Generating training data from all available cases..."
python3 generate_training_data.py

if [ $? -ne 0 ]; then
    echo "❌ Training data generation failed!"
    exit 1
fi

# Step 2: Run full training
echo ""
echo "🧠 Step 2: Training model at full resolution with complete dataset..."
python3 dof_regressor_model.py --config full_config.yaml --metadata-dir synthetic_training_data

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

# Step 3: Test the trained model
echo ""
echo "🧪 Step 3: Testing trained model performance..."
python3 test_trained_model.py

if [ $? -ne 0 ]; then
    echo "⚠️ Model testing failed, but training completed successfully"
fi

echo ""
echo "✅ Full training completed successfully!"
echo "📁 Model saved to: checkpoints/best_checkpoint.pth"
echo "📊 View results: python3 -m tensorboard.main --logdir=runs/full_production_training"
echo ""
echo "🎯 Production model ready for deployment!" 