#!/usr/bin/env python3
"""
Demo script showing how to use the trained 6-DOF regression model
"""

import os
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import json

# Import our model
from dof_regressor_model import DOF3DCNN, MedicalImageDataset

def load_trained_model(model_path: str, input_size=(128, 128, 64)):
    """Load a trained model from checkpoint"""
    print(f"📂 Loading model from: {model_path}")
    
    # Create model with same architecture
    model = DOF3DCNN(input_size=input_size, dropout_rate=0.3)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   - Trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"   - Final loss: {checkpoint.get('val_loss', checkpoint.get('loss', 'unknown'))}")
    
    return model

def predict_6dof(model, image_path: str, target_size=(128, 128, 64)):
    """Predict 6-DOF transformation from a medical image"""
    print(f"\n🔍 Analyzing image: {Path(image_path).name}")
    
    # Load and preprocess image
    img = nib.load(image_path)
    data = img.get_fdata()
    
    print(f"   Original size: {data.shape}")
    
    # Create dataset for single image (handles preprocessing)
    dataset = MedicalImageDataset([image_path], [np.zeros(6)], target_size=target_size, normalize=True)
    processed_data, _ = dataset[0]
    
    # Add batch dimension
    input_tensor = processed_data.unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        prediction = model(input_tensor)
        dof_pred = prediction.cpu().numpy()[0]
    
    print(f"   Processed size: {processed_data.shape}")
    print(f"   📊 Predicted 6-DOF:")
    print(f"      Rotations (degrees): RX={dof_pred[0]:.2f}°, RY={dof_pred[1]:.2f}°, RZ={dof_pred[2]:.2f}°")
    print(f"      Translations (mm):   TX={dof_pred[3]:.2f}mm, TY={dof_pred[4]:.2f}mm, TZ={dof_pred[5]:.2f}mm")
    
    return dof_pred

def demo_with_synthetic_data():
    """Demo using synthetic test data"""
    print("\n🧪 Demo with synthetic test data...")
    
    # Look for synthetic test data
    test_data_dir = Path("synthetic_test_data")
    if not test_data_dir.exists():
        print("❌ No synthetic test data found. Generate some first with:")
        print("   python evaluate_model.py --generate-test-data")
        return
    
    # Find test metadata
    metadata_files = list(test_data_dir.glob("*_metadata.json"))
    if not metadata_files:
        print("❌ No test metadata found.")
        return
    
    # Load first test sample
    with open(metadata_files[0], 'r') as f:
        metadata = json.load(f)
    
    if not metadata['variants']:
        print("❌ No test variants found.")
        return
    
    # Get first variant
    variant = metadata['variants'][0]
    image_path = variant['output_file']
    true_rotation = variant['applied_rotation']
    true_translation = variant['applied_translation']
    
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        return
    
    print(f"   📋 Ground truth:")
    print(f"      Rotations (degrees): RX={true_rotation[0]:.2f}°, RY={true_rotation[1]:.2f}°, RZ={true_rotation[2]:.2f}°")
    print(f"      Translations (mm):   TX={true_translation[0]:.2f}mm, TY={true_translation[1]:.2f}mm, TZ={true_translation[2]:.2f}mm")
    
    return image_path, np.array(true_rotation + true_translation)

def main():
    """Main demo function"""
    print("🚀 Medical Image 6-DOF Regression - Demo")
    print("=" * 50)
    
    # Check for trained model
    model_path = "checkpoints/best_checkpoint.pth"
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found: {model_path}")
        print("\n📝 Please train the model first:")
        print("   1. Generate training data: python generate_training_data.py")
        print("   2. Train model: python dof_regressor_model.py")
        print("   3. Run this demo again")
        return
    
    # Load trained model
    try:
        model = load_trained_model(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Demo with synthetic test data
    demo_result = demo_with_synthetic_data()
    if demo_result:
        test_image_path, true_dof = demo_result
        predicted_dof = predict_6dof(model, test_image_path)
        
        # Calculate accuracy
        errors = np.abs(predicted_dof - true_dof)
        print(f"\n📊 Prediction Accuracy:")
        print(f"   Rotation errors: RX={errors[0]:.2f}°, RY={errors[1]:.2f}°, RZ={errors[2]:.2f}°")
        print(f"   Translation errors: TX={errors[3]:.2f}mm, TY={errors[4]:.2f}mm, TZ={errors[5]:.2f}mm")
        print(f"   Mean absolute error: {np.mean(errors):.2f}")
        
        if np.mean(errors) < 10:
            print("   🎉 Excellent accuracy!")
        elif np.mean(errors) < 30:
            print("   ✅ Good accuracy!")
        else:
            print("   ⚠️ Model may need more training")
    
    print(f"\n💡 Usage for new images:")
    print(f"   from demo import load_trained_model, predict_6dof")
    print(f"   model = load_trained_model('checkpoints/best_checkpoint.pth')")
    print(f"   dof = predict_6dof(model, 'your_image.nii.gz')")

if __name__ == "__main__":
    main() 