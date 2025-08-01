#!/usr/bin/env python3
"""
Production training script for full dataset at full resolution
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json

# Import from our existing modules
from dof_regressor_model import DOF3DCNN, MedicalImageDataset


class ProductionTrainer:
    """Production trainer for full dataset"""
    
    def __init__(self, config):
        self.config = config
        
        # Set device
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("✅ Using Apple MPS acceleration")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("✅ Using CUDA acceleration")
        else:
            self.device = torch.device('cpu')
            print("⚠️ Using CPU (no hardware acceleration)")
        
        # Setup logging
        self.setup_logging()
        
        # Create model
        self.model = DOF3DCNN(
            input_size=tuple(config['model']['input_size']),
            dropout_rate=config['model']['dropout_rate']
        ).to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Setup loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=15
        )
        
        # Tensorboard
        self.writer = SummaryWriter(config['training']['log_dir'])
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Gradient accumulation steps
        self.grad_accumulation_steps = config['training'].get('gradient_accumulation', 1)
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config['training']['log_dir'])
        log_dir.mkdir(exist_ok=True, parents=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_dataset_from_metadata(self, metadata_dir):
        """Load dataset from metadata files"""
        metadata_files = list(Path(metadata_dir).glob("*_metadata.json"))
        
        image_paths = []
        labels = []
        
        self.logger.info(f"Loading dataset from {len(metadata_files)} metadata files...")
        
        for metadata_file in tqdm(metadata_files, desc="Loading metadata"):
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
        
        self.logger.info(f"Loaded {len(image_paths)} image-label pairs")
        return image_paths, labels
    
    def create_data_loaders(self, image_paths, labels):
        """Create train/validation data loaders"""
        from torch.utils.data import DataLoader
        from sklearn.model_selection import train_test_split
        
        # Split data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, 
            test_size=self.config['training']['val_split'],
            random_state=42
        )
        
        self.logger.info(f"Train samples: {len(train_paths)}, Val samples: {len(val_paths)}")
        
        # Create datasets
        train_dataset = MedicalImageDataset(
            train_paths, train_labels,
            target_size=tuple(self.config['model']['input_size']),
            normalize=False, augment=True
        )
        
        val_dataset = MedicalImageDataset(
            val_paths, val_labels,
            target_size=tuple(self.config['model']['input_size']),
            normalize=False, augment=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['training']['batch_size'],
            shuffle=True, 
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Clear gradients
            if batch_idx % self.grad_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Scale loss by accumulation steps
            loss = loss / self.grad_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights after accumulation steps
            if (batch_idx + 1) % self.grad_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                self.optimizer.step()
                if batch_idx % self.grad_accumulation_steps != 0:
                    self.optimizer.zero_grad()
            
            # Track full loss (not scaled)
            running_loss += loss.item() * self.grad_accumulation_steps
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item() * self.grad_accumulation_steps:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log to tensorboard
            if batch_idx % self.config['training']['log_interval'] == 0:
                step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Loss/Train_Batch', loss.item() * self.grad_accumulation_steps, step)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], step)
        
        epoch_loss = running_loss / len(train_loader)
        self.train_losses.append(epoch_loss)
        return epoch_loss
    
    def validate(self, val_loader, epoch):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target).item()
                
                val_loss += loss
                predictions.extend(output.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        val_loss /= len(val_loader)
        self.val_losses.append(val_loss)
        
        # Calculate metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        
        # Per-DOF metrics
        dof_names = ['RX', 'RY', 'RZ', 'TX', 'TY', 'TZ']
        dof_mae = [mean_absolute_error(targets[:, i], predictions[:, i]) for i in range(6)]
        
        self.logger.info(f'Validation - Loss: {val_loss:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}')
        for i, (name, mae_val) in enumerate(zip(dof_names, dof_mae)):
            self.logger.info(f'{name} MAE: {mae_val:.4f}')
        
        # Log to tensorboard
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        self.writer.add_scalar('Metrics/MAE', mae, epoch)
        self.writer.add_scalar('Metrics/RMSE', rmse, epoch)
        
        for i, (name, mae_val) in enumerate(zip(dof_names, dof_mae)):
            self.writer.add_scalar(f'DOF_MAE/{name}', mae_val, epoch)
        
        return val_loss, mae, rmse
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / 'latest_checkpoint.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best_checkpoint.pth')
            self.logger.info(f'New best model saved with validation loss: {val_loss:.4f}')
    
    def train(self, metadata_dir):
        """Main training loop"""
        self.logger.info("Starting production training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Gradient accumulation steps: {self.grad_accumulation_steps}")
        self.logger.info(f"Effective batch size: {self.config['training']['batch_size'] * self.grad_accumulation_steps}")
        self.logger.info(f"Image resolution: {self.config['model']['input_size']}")
        
        # Load dataset
        image_paths, labels = self.load_dataset_from_metadata(metadata_dir)
        train_loader, val_loader = self.create_data_loaders(image_paths, labels)
        
        self.logger.info(f"Starting training for {self.config['training']['num_epochs']} epochs...")
        
        # Training loop
        for epoch in range(self.config['training']['num_epochs']):
            self.logger.info(f"\nEpoch {epoch+1}/{self.config['training']['num_epochs']}")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, mae, rmse = self.validate(val_loader, epoch)
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping
            if self.config['training']['early_stopping']['enabled']:
                patience = self.config['training']['early_stopping']['patience']
                if len(self.val_losses) > patience:
                    recent_losses = self.val_losses[-patience:]
                    if all(loss >= self.best_val_loss for loss in recent_losses):
                        self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Production training for full dataset')
    parser.add_argument('--config', '-c', default='full_config.yaml', 
                       help='Training configuration file')
    parser.add_argument('--metadata-dir', '-m', default='synthetic_training_data',
                       help='Directory containing metadata files')
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} not found!")
        sys.exit(1)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if training data exists
    if not Path(args.metadata_dir).exists():
        print("Error: Training data not found!")
        print("Run: python3 generate_training_data.py first")
        sys.exit(1)
    
    # Initialize trainer
    trainer = ProductionTrainer(config)
    
    # Start training
    try:
        trainer.train(args.metadata_dir)
        print("✅ Production training completed successfully!")
        print(f"📁 Best model saved to: checkpoints/best_checkpoint.pth")
        print(f"📊 View results: tensorboard --logdir={config['training']['log_dir']}")
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main() 