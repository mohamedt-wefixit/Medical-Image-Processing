#!/usr/bin/env python3
"""
Master script for training 6-DOF regression model on medical images
"""

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path


def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed"""
    logger = logging.getLogger(__name__)
    
    try:
        import torch
        import nibabel
        import sklearn
        logger.info("✅ All dependencies are available")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("Please install requirements: pip install -r requirements.txt")
        return False


def check_dataset():
    """Check if MM-WHS dataset is available"""
    logger = logging.getLogger(__name__)
    
    mr_train_dir = Path("MM-WHS 2017 Dataset/mr_train")
    mr_test_dir = Path("MM-WHS 2017 Dataset/mr_test")
    
    if not mr_train_dir.exists():
        logger.error(f"❌ MR training data not found: {mr_train_dir}")
        logger.error("Please ensure the MM-WHS 2017 Dataset is properly extracted")
        return False
    
    if not mr_test_dir.exists():
        logger.warning(f"⚠️ MR test data not found: {mr_test_dir}")
        logger.warning("Test evaluation will not be available")
    
    # Count training images
    train_images = list(mr_train_dir.glob("*_image.nii.gz"))
    logger.info(f"✅ Found {len(train_images)} MR training images")
    
    if len(train_images) == 0:
        logger.error("❌ No training images found")
        return False
    
    return True


def generate_training_data():
    """Generate synthetic training data"""
    logger = logging.getLogger(__name__)
    
    logger.info("🔄 Generating synthetic training data...")
    
    try:
        result = subprocess.run([
            sys.executable, "generate_training_data.py"
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            logger.info("✅ Training data generation completed")
            logger.info(result.stdout)
            return True
        else:
            logger.error("❌ Training data generation failed")
            logger.error(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Training data generation timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Training data generation error: {e}")
        return False


def train_model():
    """Train the 6-DOF regression model"""
    logger = logging.getLogger(__name__)
    
    logger.info("🔄 Training 6-DOF regression model...")
    
    try:
        result = subprocess.run([
            sys.executable, "dof_regressor_model.py",
            "--metadata-dir", "synthetic_training_data"
        ], capture_output=False, text=True)  # Show output in real-time
        
        if result.returncode == 0:
            logger.info("✅ Model training completed")
            return True
        else:
            logger.error("❌ Model training failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Model training error: {e}")
        return False


def evaluate_model():
    """Evaluate the trained model"""
    logger = logging.getLogger(__name__)
    
    # Check if trained model exists
    checkpoint_dir = Path("checkpoints")
    best_model = checkpoint_dir / "best_checkpoint.pth"
    
    if not best_model.exists():
        logger.warning("⚠️ No trained model found for evaluation")
        return False
    
    logger.info("🔄 Evaluating trained model...")
    
    try:
        result = subprocess.run([
            sys.executable, "evaluate_model.py",
            "--model", str(best_model),
            "--generate-test-data"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Model evaluation completed")
            logger.info(result.stdout)
            return True
        else:
            logger.error("❌ Model evaluation failed")
            logger.error(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"❌ Model evaluation error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Complete 6-DOF regression training pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
1. Check dependencies and dataset
2. Generate synthetic training data from MR images
3. Train 3D CNN model for 6-DOF regression
4. Evaluate model on test data

This script automates the entire process for medical image 6-DOF estimation.
        """
    )
    
    parser.add_argument('--skip-data-generation', action='store_true',
                       help='Skip synthetic data generation (use existing data)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training (use existing model)')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip model evaluation')
    parser.add_argument('--data-only', action='store_true',
                       help='Only generate training data, skip training')
    parser.add_argument('--train-only', action='store_true',
                       help='Only train model, skip data generation and evaluation')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    logger.info("🚀 Starting 6-DOF Medical Image Regression Pipeline")
    logger.info("="*60)
    
    # Step 1: Check dependencies and dataset
    logger.info("📋 Step 1: Checking dependencies and dataset...")
    if not check_dependencies():
        sys.exit(1)
    
    if not check_dataset():
        sys.exit(1)
    
    # Step 2: Generate synthetic training data
    if not args.skip_data_generation and not args.train_only:
        logger.info("\n📊 Step 2: Generating synthetic training data...")
        if not generate_training_data():
            logger.error("Failed to generate training data")
            sys.exit(1)
    else:
        logger.info("\n📊 Step 2: Skipping data generation...")
        # Check if training data exists
        if not Path("synthetic_training_data").exists():
            logger.error("❌ Training data not found. Run without --skip-data-generation")
            sys.exit(1)
    
    if args.data_only:
        logger.info("✅ Data generation completed. Exiting as requested.")
        return
    
    # Step 3: Train model
    if not args.skip_training and not args.data_only:
        logger.info("\n🧠 Step 3: Training 6-DOF regression model...")
        if not train_model():
            logger.error("Failed to train model")
            sys.exit(1)
    else:
        logger.info("\n🧠 Step 3: Skipping model training...")
    
    if args.train_only:
        logger.info("✅ Model training completed. Exiting as requested.")
        return
    
    # Step 4: Evaluate model
    if not args.skip_evaluation:
        logger.info("\n📈 Step 4: Evaluating trained model...")
        if not evaluate_model():
            logger.warning("Model evaluation failed or skipped")
    else:
        logger.info("\n📈 Step 4: Skipping model evaluation...")
    
    logger.info("\n🎉 Pipeline completed successfully!")
    logger.info("="*60)
    logger.info("Results:")
    logger.info("  - Training data: synthetic_training_data/")
    logger.info("  - Trained model: checkpoints/best_checkpoint.pth")
    logger.info("  - Training logs: runs/dof_regression/")
    logger.info("  - Evaluation plots: evaluation_plots/")
    logger.info("  - Evaluation results: evaluation_results.json")


if __name__ == "__main__":
    main() 