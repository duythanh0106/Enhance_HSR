"""
Main Training Script for Hyperspectral Super-Resolution
Run: python train.py --config proposed
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import numpy as np

from config import Config, ConfigBaseline, ConfigProposed, ConfigAblation, ConfigLightweight, ConfigSpecTrans
from data.dataset import HyperspectralDataset
from utils.metrics import MetricsCalculator
from utils.losses import L1Loss, L2Loss, CombinedLoss, AdaptiveCombinedLoss

# Import models
from models.essa_original import ESSA
from models.essa_improved import ESSA_SSAM
from models.essa_ssam_spectrans import ESSA_SSAM_SpecTrans


class Trainer:
    """Trainer class cho training và validation"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Set random seed
        self.set_seed(config.seed)
        
        # Create directories
        config.create_dirs()
        
        # Build model
        self.model = self.build_model()
        
        # Build data loaders
        self.train_loader, self.val_loader = self.build_dataloaders()
        
        # Build optimizer and scheduler
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        
        # Build loss function
        self.criterion = self.build_loss()
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator(data_range=1.0)
        
        # Training state
        self.current_epoch = 0
        self.best_psnr = 0.0
        self.train_losses = []
        self.val_metrics = []
        
        # Resume if needed
        if config.resume and config.resume_checkpoint:
            self.load_checkpoint(config.resume_checkpoint)
    
    def set_seed(self, seed):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def build_model(self):
        """Build model based on config"""
        if self.config.model_name == 'ESSA_Original':
            model = ESSA(
                inch=self.config.num_spectral_bands,
                dim=self.config.feature_dim,
                upscale=self.config.upscale_factor
            )
        elif self.config.model_name == 'ESSA_SSAM':
            model = ESSA_SSAM(
                inch=self.config.num_spectral_bands,
                dim=self.config.feature_dim,
                upscale=self.config.upscale_factor,
                fusion_mode=self.config.fusion_mode
            )
        elif self.config.model_name == 'ESSA_SSAM_SpecTrans':
            model = ESSA_SSAM_SpecTrans(
                inch=self.config.num_spectral_bands,
                dim=self.config.feature_dim,
                upscale=self.config.upscale_factor,
                fusion_mode=self.config.fusion_mode,
                use_spectrans=self.config.use_spectrans,
                spectrans_depth=self.config.spectrans_depth
            )
        else:
            raise ValueError(f"Unknown model: {self.config.model_name}")
        
        model = model.to(self.device)
        
        # Print model info
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {self.config.model_name}")
        print(f"Parameters: {num_params:,}")
        
        return model
    
    def build_dataloaders(self):
        """Build train and validation dataloaders with FIXED splits"""
        # Training dataset (FIXED split)
        train_dataset = HyperspectralDataset(
            data_root=self.config.data_root,
            dataset_type=self.config.dataset_type,
            split='train',  # ⭐ FIXED train split
            patch_size=self.config.patch_size,
            upscale=self.config.upscale_factor,
            augment=self.config.use_augmentation
        )
        
        # Validation dataset (FIXED split)
        val_dataset = HyperspectralDataset(
            data_root=self.config.data_root,
            dataset_type=self.config.dataset_type,
            split='val',  # ⭐ FIXED val split
            patch_size=self.config.patch_size,
            upscale=self.config.upscale_factor,
            augment=False  # NO augmentation for validation
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def build_optimizer(self):
        """Build optimizer"""
        if self.config.optimizer.lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        return optimizer
    
    def build_scheduler(self):
        """Build learning rate scheduler"""
        if self.config.lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.lr_min
            )
        elif self.config.lr_scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.5
            )
        elif self.config.lr_scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10
            )
        else:
            scheduler = None
        
        return scheduler
    
    def build_loss(self):
        """Build loss function"""
        if self.config.loss_type == 'l1':
            criterion = L1Loss()
        elif self.config.loss_type == 'l2':
            criterion = L2Loss()
        elif self.config.loss_type == 'combined':
            criterion = CombinedLoss(
                lambda_l1=self.config.lambda_l1,
                lambda_sam=self.config.lambda_sam,
                lambda_ssim=self.config.lambda_ssim
            )
        elif self.config.loss_type == 'adaptive':
            criterion = AdaptiveCombinedLoss()
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        return criterion
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.num_epochs}')
        
        for i, (lr, hr) in enumerate(pbar):
            lr = lr.to(self.device)
            hr = hr.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            sr = self.model(lr)
            
            # Compute loss
            if self.config.loss_type in ['combined', 'adaptive']:
                loss, loss_dict = self.criterion(sr, hr)
                if i % self.config.log_interval == 0:
                    pbar.set_postfix({
                        'loss': f"{loss_dict['total']:.4f}",
                        'l1': f"{loss_dict['l1']:.4f}",
                        'sam': f"{loss_dict['sam']:.4f}"
                    })
            else:
                loss = self.criterion(sr, hr)
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        total_psnr = 0.0
        total_ssim = 0.0
        total_sam = 0.0
        total_ergas = 0.0
        
        with torch.no_grad():
            for lr, hr in tqdm(self.val_loader, desc='Validating'):
                lr = lr.to(self.device)
                hr = hr.to(self.device)
                
                # Forward pass
                sr = self.model(lr)
                
                # Compute metrics
                metrics = self.metrics_calculator.calculate_all(
                    sr, hr, scale=self.config.upscale_factor
                )
                
                total_psnr += metrics['PSNR']
                total_ssim += metrics['SSIM']
                total_sam += metrics['SAM']
                total_ergas += metrics['ERGAS']
        
        # Average metrics
        num_samples = len(self.val_loader)
        avg_metrics = {
            'PSNR': total_psnr / num_samples,
            'SSIM': total_ssim / num_samples,
            'SAM': total_sam / num_samples,
            'ERGAS': total_ergas / num_samples
        }
        
        return avg_metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config.to_dict()
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.config.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved! PSNR: {metrics['PSNR']:.2f} dB")
        
        # Save periodic checkpoints
        if epoch % self.config.save_checkpoint_every == 0:
            epoch_path = os.path.join(self.config.checkpoint_dir, f'epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint to resume training"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("Starting Training")
        print("="*70)
        
        for epoch in range(self.current_epoch + 1, self.config.num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            if epoch % self.config.validate_every == 0:
                val_metrics = self.validate()
                self.val_metrics.append(val_metrics)
                
                # Print metrics
                print(f"\nEpoch {epoch}:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val PSNR: {val_metrics['PSNR']:.2f} dB")
                print(f"  Val SSIM: {val_metrics['SSIM']:.4f}")
                print(f"  Val SAM: {val_metrics['SAM']:.4f}°")
                print(f"  Val ERGAS: {val_metrics['ERGAS']:.4f}")
                
                # Save checkpoint
                is_best = val_metrics['PSNR'] > self.best_psnr
                if is_best:
                    self.best_psnr = val_metrics['PSNR']
                
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Update learning rate
            if self.scheduler:
                if self.config.lr_scheduler == 'plateau':
                    self.scheduler.step(val_metrics['PSNR'])
                else:
                    self.scheduler.step()
        
        print("\n" + "="*70)
        print("Training Completed!")
        print(f"Best PSNR: {self.best_psnr:.2f} dB")
        print("="*70)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Hyperspectral SR Model')
    parser.add_argument('--config', type=str, default='default',
                       choices=['default', 'baseline', 'proposed', 'spectrans', 'lightweight'],
                       help='Configuration preset')
    parser.add_argument('--data_root', type=str, default=None,
                       help='Override data root path')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config == 'baseline':
        config = ConfigBaseline()
    elif args.config == 'proposed':
        config = ConfigProposed()
    elif args.config == 'spectrans':
        config = ConfigSpecTrans()
    elif args.config == 'lightweight':
        config = ConfigLightweight()
    else:
        config = Config()
    
    # Override with command line arguments
    if args.data_root:
        config.data_root = args.data_root
    
    if args.resume:
        config.resume = True
        config.resume_checkpoint = args.resume
    
    # Print configuration
    config.print_config()
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()