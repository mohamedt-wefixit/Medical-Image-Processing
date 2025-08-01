#!/usr/bin/env python3
"""
Evaluate trained 6-DOF regression model on test data
"""

import os
import sys
import json
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import logging
import argparse
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import pandas as pd

# Import our model and dataset classes
from dof_regressor_model import DOF3DCNN, MedicalImageDataset, load_training_config


class ModelEvaluator:
    """Evaluator for trained 6-DOF regression model"""
    
    def __init__(self, model_path: str, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Load model
        self.model = self.load_model(model_path)
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_model(self, model_path: str) -> nn.Module:
        """Load trained model from checkpoint"""
        self.logger.info(f"Loading model from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model with same architecture
        model = DOF3DCNN(
            input_size=tuple(self.config['model']['input_size']),
            dropout_rate=self.config['model']['dropout_rate']
        ).to(self.device)
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        self.logger.info(f"Model loaded successfully from epoch {checkpoint['epoch']}")
        self.logger.info(f"Best validation loss: {checkpoint['val_loss']:.4f}")
        
        return model
    
    def load_test_dataset(self, metadata_dir: str) -> Tuple[List[str], List[np.ndarray]]:
        """Load test dataset from metadata files"""
        metadata_files = list(Path(metadata_dir).glob("*_metadata.json"))
        
        image_paths = []
        labels = []
        original_files = []
        
        self.logger.info(f"Loading test dataset from {len(metadata_files)} metadata files...")
        
        for metadata_file in tqdm(metadata_files, desc="Loading test metadata"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            for variant in metadata['variants']:
                image_path = variant['output_file']
                if os.path.exists(image_path):
                    # Get applied transformation (relative to original)
                    applied_rot = variant['applied_rotation']
                    applied_trans = variant['applied_translation']
                    
                    # Combine into 6-DOF vector [rx, ry, rz, tx, ty, tz]
                    dof_vector = np.array(applied_rot + applied_trans, dtype=np.float32)
                    
                    image_paths.append(image_path)
                    labels.append(dof_vector)
                    original_files.append(metadata['input_file'])
        
        self.logger.info(f"Loaded {len(image_paths)} test image-label pairs")
        return image_paths, labels, original_files
    
    def create_test_loader(self, image_paths: List[str], labels: List[np.ndarray]) -> DataLoader:
        """Create test data loader"""
        test_dataset = MedicalImageDataset(
            image_paths, labels,
            target_size=tuple(self.config['model']['input_size']),
            normalize=True, augment=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
        
        return test_loader
    
    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions on test data"""
        self.model.eval()
        predictions = []
        targets = []
        
        self.logger.info("Generating predictions on test data...")
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Predicting'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                predictions.extend(output.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        return np.array(predictions), np.array(targets)
    
    def calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """Calculate evaluation metrics"""
        # Overall metrics
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)
        
        # Per-DOF metrics
        dof_names = ['RX', 'RY', 'RZ', 'TX', 'TY', 'TZ']
        dof_units = ['°', '°', '°', 'mm', 'mm', 'mm']
        
        dof_metrics = {}
        for i, (name, unit) in enumerate(zip(dof_names, dof_units)):
            dof_mae = mean_absolute_error(targets[:, i], predictions[:, i])
            dof_mse = mean_squared_error(targets[:, i], predictions[:, i])
            dof_rmse = np.sqrt(dof_mse)
            dof_r2 = r2_score(targets[:, i], predictions[:, i])
            
            dof_metrics[name] = {
                'mae': dof_mae,
                'mse': dof_mse,
                'rmse': dof_rmse,
                'r2': dof_r2,
                'unit': unit
            }
        
        metrics = {
            'overall': {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            },
            'per_dof': dof_metrics
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print evaluation metrics"""
        self.logger.info("\n" + "="*50)
        self.logger.info("EVALUATION RESULTS")
        self.logger.info("="*50)
        
        # Overall metrics
        overall = metrics['overall']
        self.logger.info(f"\nOverall Performance:")
        self.logger.info(f"  MAE:  {overall['mae']:.4f}")
        self.logger.info(f"  MSE:  {overall['mse']:.4f}")
        self.logger.info(f"  RMSE: {overall['rmse']:.4f}")
        self.logger.info(f"  R²:   {overall['r2']:.4f}")
        
        # Per-DOF metrics
        self.logger.info(f"\nPer-DOF Performance:")
        for dof_name, dof_metrics in metrics['per_dof'].items():
            unit = dof_metrics['unit']
            self.logger.info(f"  {dof_name}:")
            self.logger.info(f"    MAE:  {dof_metrics['mae']:.3f} {unit}")
            self.logger.info(f"    RMSE: {dof_metrics['rmse']:.3f} {unit}")
            self.logger.info(f"    R²:   {dof_metrics['r2']:.4f}")
    
    def plot_results(self, predictions: np.ndarray, targets: np.ndarray, 
                    save_dir: str = "evaluation_plots"):
        """Create visualization plots"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        dof_names = ['RX (°)', 'RY (°)', 'RZ (°)', 'TX (mm)', 'TY (mm)', 'TZ (mm)']
        
        # 1. Scatter plots for each DOF
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (ax, dof_name) in enumerate(zip(axes, dof_names)):
            ax.scatter(targets[:, i], predictions[:, i], alpha=0.6, s=20)
            
            # Perfect prediction line
            min_val = min(targets[:, i].min(), predictions[:, i].min())
            max_val = max(targets[:, i].max(), predictions[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
            
            ax.set_xlabel(f'True {dof_name}')
            ax.set_ylabel(f'Predicted {dof_name}')
            ax.set_title(f'{dof_name} Predictions')
            ax.grid(True, alpha=0.3)
            
            # Calculate and display R²
            r2 = r2_score(targets[:, i], predictions[:, i])
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path / 'dof_scatter_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Error distribution plots
        errors = predictions - targets
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (ax, dof_name) in enumerate(zip(axes, dof_names)):
            ax.hist(errors[:, i], bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel(f'Error in {dof_name}')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{dof_name} Error Distribution')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_error = np.mean(errors[:, i])
            std_error = np.std(errors[:, i])
            ax.text(0.05, 0.95, f'μ = {mean_error:.3f}\nσ = {std_error:.3f}', 
                   transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path / 'error_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Correlation heatmap
        plt.figure(figsize=(10, 8))
        
        # Combine predictions and targets for correlation analysis
        all_data = np.hstack([targets, predictions])
        labels = [f'True {name}' for name in dof_names] + [f'Pred {name}' for name in dof_names]
        
        correlation_matrix = np.corrcoef(all_data.T)
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   xticklabels=labels, yticklabels=labels, fmt='.2f')
        plt.title('Correlation Matrix: True vs Predicted DOF Values')
        plt.tight_layout()
        plt.savefig(save_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Error magnitude vs target value
        error_magnitudes = np.linalg.norm(errors, axis=1)
        target_magnitudes = np.linalg.norm(targets, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(target_magnitudes, error_magnitudes, alpha=0.6)
        plt.xlabel('Target 6-DOF Magnitude')
        plt.ylabel('Prediction Error Magnitude')
        plt.title('Prediction Error vs Target Magnitude')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(target_magnitudes, error_magnitudes, 1)
        p = np.poly1d(z)
        plt.plot(target_magnitudes, p(target_magnitudes), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(save_path / 'error_vs_magnitude.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Plots saved to: {save_path}")
    
    def save_results(self, predictions: np.ndarray, targets: np.ndarray, 
                    image_paths: List[str], metrics: Dict, save_path: str = "evaluation_results.json"):
        """Save detailed evaluation results"""
        # Create detailed results
        results = {
            'metrics': metrics,
            'predictions': predictions.tolist(),
            'targets': targets.tolist(),
            'image_paths': image_paths,
            'num_samples': len(predictions),
            'model_config': self.config
        }
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Detailed results saved to: {save_path}")
        
        # Also save as CSV for easy analysis
        dof_names = ['RX', 'RY', 'RZ', 'TX', 'TY', 'TZ']
        
        # Create DataFrame
        data = {}
        for i, name in enumerate(dof_names):
            data[f'True_{name}'] = targets[:, i]
            data[f'Pred_{name}'] = predictions[:, i]
            data[f'Error_{name}'] = predictions[:, i] - targets[:, i]
        
        data['Image_Path'] = image_paths
        
        df = pd.DataFrame(data)
        csv_path = save_path.replace('.json', '.csv')
        df.to_csv(csv_path, index=False)
        
        self.logger.info(f"Results also saved as CSV: {csv_path}")
    
    def evaluate(self, test_metadata_dir: str, save_plots: bool = True):
        """Main evaluation function"""
        self.logger.info("Starting model evaluation...")
        
        # Load test data
        image_paths, labels, original_files = self.load_test_dataset(test_metadata_dir)
        test_loader = self.create_test_loader(image_paths, labels)
        
        # Generate predictions
        predictions, targets = self.predict(test_loader)
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, targets)
        
        # Print results
        self.print_metrics(metrics)
        
        # Create plots
        if save_plots:
            self.plot_results(predictions, targets)
        
        # Save detailed results
        self.save_results(predictions, targets, image_paths, metrics)
        
        return metrics, predictions, targets


def generate_test_data(test_dir: str = "MM-WHS 2017 Dataset/mr_test"):
    """Generate synthetic test data from MR test images"""
    logger = logging.getLogger(__name__)
    
    # Check if test directory exists
    test_path = Path(test_dir)
    if not test_path.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return False
    
    # Find test image files
    test_images = list(test_path.glob("*_image.nii.gz"))
    if not test_images:
        logger.error(f"No test images found in {test_dir}")
        return False
    
    logger.info(f"Found {len(test_images)} test images")
    
    # Create test configuration (similar to training but fewer variants)
    test_config = {
        'output_dir': 'synthetic_test_data',
        'log_level': 'INFO',
        'transformations': {
            'rotation_range': [-45, 45],
            'translation_range': [-20, 20],
            'variants_per_image': 10,  # Fewer variants for testing
            'min_dof_modified': 1,
            'max_dof_modified': 3
        },
        'dataset': {
            'extensions': ['.nii', '.nii.gz'],
            'batch_size': 5,
            'validate_outputs': True
        },
        'sample': {
            'enabled': False
        }
    }
    
    config_path = 'test_data_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f, default_flow_style=False, indent=2)
    
    # Generate test data
    try:
        import subprocess
        cmd = [
            sys.executable, "medical_image_generator.py",
            "--config", config_path
        ]
        
        # Add test image files
        for img_file in test_images:
            cmd.append(str(img_file))
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Test data generation completed successfully!")
            return True
        else:
            logger.error("Test data generation failed!")
            logger.error(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"Failed to generate test data: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained 6-DOF regression model')
    parser.add_argument('--model', '-m', required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--test-data', '-t', default='synthetic_test_data',
                       help='Directory containing test metadata files')
    parser.add_argument('--config', '-c', default='training_config.yaml',
                       help='Training configuration file')
    parser.add_argument('--generate-test-data', action='store_true',
                       help='Generate synthetic test data before evaluation')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Generate test data if requested
    if args.generate_test_data:
        logger.info("Generating synthetic test data...")
        if not generate_test_data():
            logger.error("Failed to generate test data")
            sys.exit(1)
        args.test_data = 'synthetic_test_data'
    
    # Check if model checkpoint exists
    if not os.path.exists(args.model):
        logger.error(f"Model checkpoint not found: {args.model}")
        sys.exit(1)
    
    # Check if test data exists
    if not os.path.exists(args.test_data):
        logger.error(f"Test data directory not found: {args.test_data}")
        logger.info("Use --generate-test-data to create synthetic test data")
        sys.exit(1)
    
    # Load configuration
    config = load_training_config(args.config)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model, config)
    
    # Run evaluation
    try:
        metrics, predictions, targets = evaluator.evaluate(
            args.test_data, 
            save_plots=not args.no_plots
        )
        logger.info("✅ Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main() 