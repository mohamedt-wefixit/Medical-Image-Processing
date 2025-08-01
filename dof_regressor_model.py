import os
import sys
import json
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import logging
import argparse
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MedicalImageDataset(Dataset):
    """Dataset for medical images and their 6-DOF labels"""
    
    def __init__(self, image_paths: List[str], labels: List[np.ndarray], 
                 target_size: Tuple[int, int, int] = (128, 128, 64),
                 normalize: bool = False, augment: bool = False):
        """
        Args:
            image_paths: List of paths to image files
            labels: List of 6-DOF labels (Rx, Ry, Rz, Tx, Ty, Tz)
            target_size: Target size for resizing images
            normalize: Whether to normalize images
            augment: Whether to apply data augmentation
        """
        self.image_paths = image_paths
        self.labels = labels
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment
        
        # Default normalization values
        self.mean = 0.0
        self.std = 1.0
        
        # Precompute normalization stats if needed
        if self.normalize:
            self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        """Compute mean and std for normalization"""
        print("Computing normalization statistics...")
        all_values = []
        
        # Sample a subset for efficiency
        sample_indices = np.random.choice(len(self.image_paths), 
                                        min(10, len(self.image_paths)), 
                                        replace=False)
        
        for idx in tqdm(sample_indices, desc="Computing stats"):
            img = nib.load(self.image_paths[idx])
            data = img.get_fdata()
            all_values.extend(data.flatten())
        
        all_values = np.array(all_values)
        self.mean = np.mean(all_values)
        self.std = np.std(all_values)
        print(f"Normalization stats - Mean: {self.mean:.2f}, Std: {self.std:.2f}")
    
    def _preprocess_image(self, data: np.ndarray) -> np.ndarray:
        """Preprocess medical image data"""
        # Handle different input shapes
        if len(data.shape) == 4:
            data = data[:, :, :, 0]  # Take first channel if 4D
        
        # Resize to target size
        if data.shape != self.target_size:
            # Simple interpolation for resizing
            from scipy.ndimage import zoom
            zoom_factors = [t/s for t, s in zip(self.target_size, data.shape)]
            data = zoom(data, zoom_factors, order=1)
        
        # Normalize
        if self.normalize:
            data = (data - self.mean) / (self.std + 1e-8)
        
        # Clip outliers
        data = np.clip(data, -5, 5)
        
        return data.astype(np.float32)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = nib.load(img_path)
        data = img.get_fdata()
        
        # Preprocess
        data = self._preprocess_image(data)
        
        # Apply data augmentation if enabled
        if self.augment:
            # Add random noise
            noise_level = 0.05
            noise = np.random.normal(0, noise_level, data.shape)
            data = data + noise
            
            # Random intensity shift
            intensity_shift = np.random.uniform(-0.1, 0.1)
            data = data + intensity_shift
            
            # Clip to reasonable range
            data = np.clip(data, -5, 5)
        
        # Convert to tensor and add channel dimension
        data = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)  # (1, D, H, W)
        
        # Get label
        label = torch.from_numpy(self.labels[idx].astype(np.float32)).float()
        
        return data, label


class DOF3DCNN(nn.Module):
    """3D CNN for 6-DOF regression with residual connections"""
    
    def __init__(self, input_size: Tuple[int, int, int] = (128, 128, 64), 
                 dropout_rate: float = 0.3):
        super(DOF3DCNN, self).__init__()
        
        self.input_size = input_size
        
        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(1, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.GroupNorm(8, 32)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.GroupNorm(8, 64)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.GroupNorm(16, 128)
        
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.GroupNorm(32, 256)
        
        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.GroupNorm(32, 512)
        
        # Residual connections
        self.res1 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32)
        )
        
        self.res2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64)
        )
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        
        # Fully connected layers for regression
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 6)  # 6 DOF output
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual connection 1
        identity = x
        x = self.res1(x) + identity
        x = F.relu(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Residual connection 2
        identity = x
        x = self.res2(x) + identity
        x = F.relu(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(self.dropout(x)))
        x = F.relu(self.fc2(self.dropout(x)))
        x = F.relu(self.fc3(self.dropout(x)))
        x = self.fc_out(x)
        
        return x


class DOFTrainer:
    """Trainer class for 6-DOF regression model"""
    
    def __init__(self, config: Dict):
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"Using device: {self.device}")
        
        # Setup logging
        self.setup_logging()
        
        # Setup model
        self.model = DOF3DCNN(
            input_size=tuple(config['model']['input_size']),
            dropout_rate=config['model']['dropout_rate']
        ).to(self.device)
        
        # Setup loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Tensorboard
        self.writer = SummaryWriter(config['training']['log_dir'])
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config['training']['log_dir'])
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_dataset_from_metadata(self, metadata_dir: str) -> Tuple[List[str], List[np.ndarray]]:
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
    
    def create_data_loaders(self, image_paths: List[str], labels: List[np.ndarray]):
        """Create train/validation data loaders"""
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
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Log to tensorboard
            if batch_idx % self.config['training']['log_interval'] == 0:
                step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Loss/Train_Batch', loss.item(), step)
        
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
                val_loss += self.criterion(output, target).item()
                
                predictions.extend(output.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        val_loss /= len(val_loader)
        self.val_losses.append(val_loss)
        
        # Calculate metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        
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
    
    def train(self, metadata_dir: str):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        # Load dataset
        image_paths, labels = self.load_dataset_from_metadata(metadata_dir)
        
        # Use full dataset (comment out subsetting)
        # if len(image_paths) > 200:
        #     self.logger.info(f"Using subset of data for faster training: 200/{len(image_paths)} samples")
        #     indices = np.random.choice(len(image_paths), 200, replace=False)
        #     image_paths = [image_paths[i] for i in indices]
        #     labels = [labels[i] for i in indices]
        
        train_loader, val_loader = self.create_data_loaders(image_paths, labels)
        
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
        self.writer.close()
    
    def plot_training_history(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training History')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Validation Loss')
        
        plt.tight_layout()
        plt.savefig(Path(self.config['training']['log_dir']) / 'training_history.png')
        plt.show()


def load_training_config(config_path: str = "training_config.yaml") -> Dict:
    """Load training configuration"""
    default_config = {
        'model': {
            'input_size': [128, 128, 64],
            'dropout_rate': 0.3
        },
        'training': {
            'batch_size': 4,
            'num_epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'val_split': 0.2,
            'num_workers': 4,
            'log_interval': 10,
            'log_dir': 'runs/dof_regression',
            'checkpoint_dir': 'checkpoints',
            'early_stopping': {
                'enabled': True,
                'patience': 15
            }
        },
        'data': {
            'metadata_dir': 'synthetic_dataset',
            'target_size': [128, 128, 64]
        }
    }
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Merge with defaults
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if subkey not in config[key]:
                        config[key][subkey] = subvalue
    else:
        config = default_config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"Created default training config: {config_path}")
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Train 3D CNN for 6-DOF regression')
    parser.add_argument('--config', '-c', default='training_config.yaml', 
                       help='Training configuration file')
    parser.add_argument('--metadata-dir', '-m', default='synthetic_dataset',
                       help='Directory containing metadata files')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                       help='GPU device ID')
    
    args = parser.parse_args()
    
    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    # Load configuration
    config = load_training_config(args.config)
    
    # Override metadata directory if specified
    if args.metadata_dir:
        config['data']['metadata_dir'] = args.metadata_dir
    
    # Initialize trainer
    trainer = DOFTrainer(config)
    
    # Start training
    try:
        trainer.train(config['data']['metadata_dir'])
        trainer.plot_training_history()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main() 