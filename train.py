"""
Main Training Script for Hyperspectral Super-Resolution
Run: python train.py --config proposed
"""

import os
import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from config import (
    Config,
    ConfigBaseline,
    ConfigProposed,
    ConfigLightweight,
    ConfigSpecTrans,
    ConfigUniversalBest,
)
from data.dataset import HyperspectralDataset
from models.factory import build_model_from_config, load_state_dict_compat
from utils.metrics import MetricsCalculator
from utils.losses import L1Loss, L2Loss, CombinedLoss, AdaptiveCombinedLoss
from utils.device import resolve_device


class ModelEMA:
    """EMA tracker cho tham số model, dùng cho validation/checkpoint ổn định hơn."""

    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup = None

    def update(self, model):
        one_minus_decay = 1.0 - self.decay
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=one_minus_decay)

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name])

    def restore(self, model):
        if self.backup is None:
            return
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = None

    def state_dict(self):
        return {
            'decay': self.decay,
            'shadow': {k: v.detach().clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state):
        self.decay = float(state.get('decay', self.decay))
        loaded_shadow = state.get('shadow', {})
        if loaded_shadow:
            self.shadow = {k: v.detach().clone() for k, v in loaded_shadow.items()}


class Trainer:
    """Trainer class cho training và validation"""
    
    def __init__(self, config):
        self.config = config

        self.device = resolve_device(config.device)
        self.use_amp = bool(config.mixed_precision and self.device.type == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.log_file = None

    
        # Set random seed
        self.set_seed(config.seed)
        
        # Build data loaders
        self.train_loader, self.val_loader = self.build_dataloaders()

        # Build model (after data loader so num_spectral_bands is auto-detected)
        self.model = self.build_model()
        self.ema = ModelEMA(self.model, decay=self.config.ema_decay) if self.config.use_ema else None
        
        # Build optimizer and scheduler
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        if self.config.warmup_epochs > 0:
            self._set_learning_rate(self.config.warmup_start_lr)
        
        # Build loss function
        self.criterion = self.build_loss()
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator(data_range=1.0)
        
        # Training state
        self.current_epoch = 0
        self.best_psnr = 0.0
        self.best_score = float('-inf')
        self.best_epoch = 0
        self.train_losses = []
        self.val_metrics = []
        
        # Resume if needed
        if config.resume and config.resume_checkpoint:
            self.load_checkpoint(config.resume_checkpoint)
    
    def set_seed(self, seed):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        np.random.seed(seed)

    def _set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = float(lr)

    def _apply_warmup(self, epoch):
        warmup_epochs = int(getattr(self.config, 'warmup_epochs', 0))
        if warmup_epochs <= 0:
            return False

        if epoch <= warmup_epochs:
            start_lr = float(getattr(self.config, 'warmup_start_lr', 1e-6))
            target_lr = float(self.config.learning_rate)
            progress = float(epoch) / float(max(1, warmup_epochs))
            lr = start_lr + (target_lr - start_lr) * progress
            self._set_learning_rate(lr)
            return True

        # Ensure we return to base LR after warmup phase.
        if epoch == warmup_epochs + 1:
            self._set_learning_rate(self.config.learning_rate)
        return False

    @staticmethod
    def format_duration(seconds):
        total = int(seconds)
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    @staticmethod
    def _remove_dir_if_effectively_empty(path):
        if not os.path.isdir(path):
            return
        entries = [e for e in os.listdir(path) if e != '.DS_Store']
        if not entries:
            ds_store = os.path.join(path, '.DS_Store')
            if os.path.exists(ds_store):
                os.remove(ds_store)
            os.rmdir(path)

    def cleanup_empty_outputs(self):
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
        self._remove_dir_if_effectively_empty(os.path.join(self.config.log_dir, 'images'))
        self._remove_dir_if_effectively_empty(self.config.log_dir)
        self._remove_dir_if_effectively_empty(self.config.checkpoint_dir)

    def _ensure_output_dirs(self):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)

    def _open_log_file(self):
        self._ensure_output_dirs()
        if self.log_file is None:
            log_path = os.path.join(self.config.log_dir, 'training.log')
            self.log_file = open(log_path, 'a', encoding='utf-8', buffering=1)
            self.log_file.write("=== Training Log Started ===\n")

    def _log(self, message):
        print(message)
        if self.log_file is not None:
            self.log_file.write(message + "\n")
    
    def build_model(self):
        """Build model based on config"""
        model = build_model_from_config(self.config)
        model = model.to(self.device)
        
        # Print model info
        num_params = sum(p.numel() for p in model.parameters())
        self._log(f"Model: {self.config.model_name}")
        self._log(f"Parameters: {num_params:,}")
        self._log(f"Using device: {self.device}")
        
        return model
    
    def build_dataloaders(self):
        """Build train and validation dataloaders from split.json."""

        split_kwargs = {
            "data_root": self.config.data_root,
            "upscale": self.config.upscale_factor,
            "split_seed": self.config.split_seed,
            "train_ratio": self.config.train_ratio,
            "val_ratio": self.config.val_ratio,
            "test_ratio": self.config.test_ratio,
            "force_regenerate_split": self.config.regenerate_split,
        }

        train_dataset = HyperspectralDataset(
            patch_size=self.config.patch_size,
            augment=self.config.use_augmentation,
            split='train',
            virtual_samples_per_epoch=self.config.train_virtual_samples_per_epoch,
            **split_kwargs
        )

        try:
            val_dataset = HyperspectralDataset(
                patch_size=self.config.patch_size,
                augment=False,
                split='val',
                virtual_samples_per_epoch=self.config.val_virtual_samples_per_epoch,
                **split_kwargs
            )
        except ValueError as exc:
            message = str(exc)
            if "No images found for split 'val'" not in message:
                raise
            self._log("⚠️ Validation split is empty. Falling back to split='train' for validation.")
            val_dataset = HyperspectralDataset(
                patch_size=self.config.patch_size,
                augment=False,
                split='train',
                virtual_samples_per_epoch=self.config.val_virtual_samples_per_epoch,
                **split_kwargs
            )

        # Sync config with detected number of spectral bands
        if self.config.num_spectral_bands != train_dataset.num_bands:
            print(
                f"Updating num_spectral_bands: "
                f"{self.config.num_spectral_bands} -> {train_dataset.num_bands} (auto-detected)"
            )
        self.config.num_spectral_bands = train_dataset.num_bands
        if val_dataset.num_bands != train_dataset.num_bands:
            raise ValueError(
                f"Mismatch num_bands between train ({train_dataset.num_bands}) "
                f"and val ({val_dataset.num_bands}) splits."
            )

        num_workers = max(0, int(self.config.num_workers))
        pin_memory = self.device.type == 'cuda'
        loader_kwargs = {
            'num_workers': num_workers,
            'pin_memory': pin_memory
        }
        if num_workers > 0:
            loader_kwargs['persistent_workers'] = True

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            **loader_kwargs
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            **loader_kwargs
        )

        train_size = len(train_dataset)
        val_size = len(val_dataset)
        train_images = len(train_dataset.image_paths)
        val_images = len(val_dataset.image_paths)
        self._log(f"Total samples: {train_size + val_size}")
        self._log(f"Train images: {train_images}, Val images: {val_images}")
        self._log(f"Train samples: {train_size}, Val samples: {val_size}")
        self._log(f"DataLoader workers: {num_workers}, pin_memory: {pin_memory}")

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

    @staticmethod
    def _clamp01(value):
        return max(0.0, min(1.0, float(value)))

    def _get_two_phase_lambdas(self, epoch):
        """
        Return (lambda_l1, lambda_sam, lambda_ssim, phase_label) for current epoch.
        Phase 1: prioritize stable pixel convergence.
        Phase 2: increase spectral/structural regularization.
        """
        target_l1 = float(self.config.lambda_l1)
        target_sam = float(self.config.lambda_sam)
        target_ssim = float(self.config.lambda_ssim)

        phase1_ratio = float(getattr(self.config, 'loss_phase1_ratio', 0.4))
        phase1_epochs = max(1, int(round(self.config.num_epochs * phase1_ratio)))
        transition_epochs = max(0, int(getattr(self.config, 'loss_phase_transition_epochs', 0)))

        phase1_sam = target_sam * float(getattr(self.config, 'loss_phase1_sam_scale', 0.35))
        phase1_ssim = target_ssim * float(getattr(self.config, 'loss_phase1_ssim_scale', 0.25))

        if epoch <= phase1_epochs:
            return target_l1, phase1_sam, phase1_ssim, 'phase1'

        if transition_epochs <= 0:
            return target_l1, target_sam, target_ssim, 'phase2'

        transition_step = epoch - phase1_epochs
        if transition_step >= transition_epochs:
            return target_l1, target_sam, target_ssim, 'phase2'

        alpha = float(transition_step) / float(max(1, transition_epochs))
        sam = phase1_sam + (target_sam - phase1_sam) * alpha
        ssim = phase1_ssim + (target_ssim - phase1_ssim) * alpha
        return target_l1, sam, ssim, 'transition'

    def _apply_loss_schedule(self, epoch):
        """
        Dynamically adjust CombinedLoss weights during training.
        Returns a tuple: (applied, phase_label, l1, sam, ssim)
        """
        if self.config.loss_type != 'combined':
            return False, 'static', None, None, None
        if not getattr(self.config, 'use_two_phase_loss', False):
            return False, 'static', self.config.lambda_l1, self.config.lambda_sam, self.config.lambda_ssim
        if not hasattr(self.criterion, 'set_weights'):
            return False, 'static', self.config.lambda_l1, self.config.lambda_sam, self.config.lambda_ssim

        l1, sam, ssim, phase = self._get_two_phase_lambdas(epoch)
        self.criterion.set_weights(lambda_l1=l1, lambda_sam=sam, lambda_ssim=ssim)
        return True, phase, l1, sam, ssim

    def compute_selection_score(self, metrics):
        """Compute score used to determine best checkpoint."""
        mode = str(getattr(self.config, 'best_selection_metric', 'psnr')).lower()
        if mode == 'psnr':
            return float(metrics['PSNR'])

        weights = getattr(self.config, 'best_score_weights', {}) or {}
        refs = getattr(self.config, 'best_score_refs', {}) or {}
        w_psnr = float(weights.get('psnr', 0.45))
        w_ssim = float(weights.get('ssim', 0.25))
        w_sam = float(weights.get('sam', 0.20))
        w_ergas = float(weights.get('ergas', 0.10))

        psnr_ref = float(refs.get('psnr', 50.0))
        sam_ref = float(refs.get('sam', 10.0))
        ergas_ref = float(refs.get('ergas', 20.0))

        psnr_score = self._clamp01(float(metrics['PSNR']) / max(psnr_ref, 1e-6))
        ssim_score = self._clamp01(float(metrics['SSIM']))
        sam_score = self._clamp01(1.0 - float(metrics['SAM']) / max(sam_ref, 1e-6))
        ergas_score = self._clamp01(1.0 - float(metrics['ERGAS']) / max(ergas_ref, 1e-6))

        composite = (
            w_psnr * psnr_score
            + w_ssim * ssim_score
            + w_sam * sam_score
            + w_ergas * ergas_score
        )
        return float(composite)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        epoch_start = time.time()
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.num_epochs}')
        
        for i, (lr, hr) in enumerate(pbar):
            lr = lr.to(self.device, non_blocking=self.device.type == 'cuda')
            hr = hr.to(self.device, non_blocking=self.device.type == 'cuda')
            
            # Forward pass
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
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
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.config.gradient_clip_norm and self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.gradient_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.gradient_clip_norm and self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.gradient_clip_norm
                    )
                self.optimizer.step()

            if self.ema is not None:
                self.ema.update(self.model)
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(self.train_loader)
        train_time = time.time() - epoch_start
        return avg_loss, train_time
    
    def validate(self):
        """Validate the model"""
        if self.ema is not None:
            self.ema.apply_shadow(self.model)
        try:
            self.model.eval()
            
            total_psnr = 0.0
            total_ssim = 0.0
            total_sam = 0.0
            total_ergas = 0.0
            
            with torch.no_grad():
                for lr, hr in tqdm(self.val_loader, desc='Validating'):
                    lr = lr.to(self.device, non_blocking=self.device.type == 'cuda')
                    hr = hr.to(self.device, non_blocking=self.device.type == 'cuda')

                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
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
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)
    
    def save_checkpoint(self, epoch, metrics, is_best=False, selection_score=None):
        """Save model checkpoint"""
        self._ensure_output_dirs()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'selection_score': selection_score,
            'best_selection_metric': self.config.best_selection_metric,
            'best_psnr': self.best_psnr,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'config': self.config.to_dict()
        }
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
        
        # Save latest checkpoint
        latest_path = os.path.join(self.config.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            mode = str(getattr(self.config, 'best_selection_metric', 'psnr')).upper()
            if selection_score is None:
                selection_score = self.compute_selection_score(metrics)
            if mode == 'PSNR':
                self._log(f"💾 Best model saved! PSNR: {metrics['PSNR']:.2f} dB")
            else:
                self._log(
                    f"💾 Best model saved! Score({mode}): {selection_score:.6f} | "
                    f"PSNR: {metrics['PSNR']:.2f} dB"
                )
        
        # Save periodic checkpoints
        if epoch % self.config.save_checkpoint_every == 0:
            epoch_path = os.path.join(self.config.checkpoint_dir, f'epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint to resume training"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        _, converted_keys = load_state_dict_compat(
            self.model, checkpoint['model_state_dict'], strict=True
        )
        if converted_keys:
            self._log(f"Converted {len(converted_keys)} legacy weight tensors for compatibility.")
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.ema is not None and checkpoint.get('ema_state_dict') is not None:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        resumed_metrics = checkpoint.get('metrics', {}) or {}
        self.best_psnr = float(checkpoint.get('best_psnr', resumed_metrics.get('PSNR', self.best_psnr)))
        if checkpoint.get('best_score') is not None:
            self.best_score = float(checkpoint['best_score'])
        elif checkpoint.get('selection_score') is not None:
            self.best_score = float(checkpoint['selection_score'])
        elif resumed_metrics:
            self.best_score = float(self.compute_selection_score(resumed_metrics))
        else:
            self.best_score = float('-inf')
        self.best_epoch = int(checkpoint.get('best_epoch', self.current_epoch))
        self._log(f"Resumed from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop"""
        self._open_log_file()
        self._log("\n" + "="*70)
        self._log("Starting Training")
        self._log("="*70)

        training_start = time.time()
        try:
            for epoch in range(self.current_epoch + 1, self.config.num_epochs + 1):
                epoch_start = time.time()
                in_warmup = self._apply_warmup(epoch)
                _, loss_phase, curr_lambda_l1, curr_lambda_sam, curr_lambda_ssim = self._apply_loss_schedule(epoch)

                # Train
                train_loss, train_time = self.train_epoch(epoch)
                self.train_losses.append(train_loss)

                val_metrics = None
                val_time = 0.0
                selection_score = None

                # Validate
                if epoch % self.config.validate_every == 0:
                    val_start = time.time()
                    val_metrics = self.validate()
                    val_time = time.time() - val_start
                    self.val_metrics.append(val_metrics)

                    # Save checkpoint
                    selection_score = self.compute_selection_score(val_metrics)
                    is_best = selection_score > self.best_score
                    if is_best:
                        self.best_score = selection_score
                        self.best_psnr = val_metrics['PSNR']
                        self.best_epoch = epoch
                    
                    self.save_checkpoint(epoch, val_metrics, is_best, selection_score)

                epoch_total_time = time.time() - epoch_start

                # Print metrics + timing (readable summary block)
                self._log("=" * 70)
                self._log(f"Epoch {epoch}/{self.config.num_epochs} Summary")
                self._log("-" * 70)
                self._log(f"  Train Loss      : {train_loss:.4f}")
                if val_metrics is not None:
                    self._log(f"  Val PSNR        : {val_metrics['PSNR']:.2f} dB")
                    self._log(f"  Val SSIM        : {val_metrics['SSIM']:.4f}")
                    self._log(f"  Val SAM         : {val_metrics['SAM']:.4f}°")
                    self._log(f"  Val ERGAS       : {val_metrics['ERGAS']:.4f}")
                    mode = str(getattr(self.config, 'best_selection_metric', 'psnr')).upper()
                    if selection_score is not None:
                        self._log(f"  Selection Score : {selection_score:.6f} ({mode})")
                current_lr = self.optimizer.param_groups[0]['lr']
                warmup_note = " (warmup)" if in_warmup else ""
                self._log(f"  Learning Rate   : {current_lr:.8f}{warmup_note}")
                if curr_lambda_l1 is not None:
                    self._log(
                        f"  Loss Weights    : L1={float(curr_lambda_l1):.4f}, "
                        f"SAM={float(curr_lambda_sam):.4f}, SSIM={float(curr_lambda_ssim):.4f} ({loss_phase})"
                    )
                self._log(f"  Train Time      : {self.format_duration(train_time)} ({train_time:.2f} seconds)")
                if val_metrics is not None:
                    self._log(f"  Validate Time   : {self.format_duration(val_time)} ({val_time:.2f} seconds)")
                self._log(f"  Epoch Total Time: {self.format_duration(epoch_total_time)} ({epoch_total_time:.2f} seconds)")
                self._log("=" * 70 + "\n")
                
                # Update learning rate
                if self.scheduler:
                    if self.config.lr_scheduler == 'plateau' and val_metrics is not None:
                        self.scheduler.step(val_metrics['PSNR'])
                    elif self.config.lr_scheduler != 'plateau' and not in_warmup:
                        self.scheduler.step()

            total_train_time = time.time() - training_start
            best_ckpt_path = os.path.join(self.config.checkpoint_dir, 'best.pth')
            latest_ckpt_path = os.path.join(self.config.checkpoint_dir, 'latest.pth')
            log_path = os.path.join(self.config.log_dir, 'training.log')
            self._log("\n" + "="*70)
            self._log("Training Completed!")
            self._log(f"Best PSNR: {self.best_psnr:.2f} dB")
            mode = str(getattr(self.config, 'best_selection_metric', 'psnr')).upper()
            self._log(f"Best Score ({mode}): {self.best_score:.6f} (epoch {self.best_epoch})")
            self._log(f"Total Training Time: {self.format_duration(total_train_time)} ({total_train_time:.2f} seconds)")
            self._log("\nSaved Outputs:")
            self._log(f"  Checkpoint Dir : {self.config.checkpoint_dir}")
            self._log(f"  Best Checkpoint: {best_ckpt_path}")
            self._log(f"  Latest Checkpoint: {latest_ckpt_path}")
            self._log(f"  Training Log   : {log_path}")
            self._log("="*70)
        finally:
            if self.log_file is not None:
                self.log_file.close()
                self.log_file = None


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Hyperspectral SR Model')
    parser.add_argument('--config', type=str, default='default',
                       choices=['default', 'baseline', 'proposed', 'spectrans', 'lightweight', 'universal_best'],
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
    elif args.config == 'universal_best':
        config = ConfigUniversalBest()
    else:
        config = Config()
    
    # Override with command line arguments
    if args.data_root:
        config.data_root = args.data_root
        if hasattr(config, 'apply_dataset_profile'):
            config.apply_dataset_profile()
        config.refresh_output_paths()
    
    if args.resume:
        config.resume = True
        config.resume_checkpoint = args.resume
    
    # Print configuration
    config.print_config()
    
    # Create trainer and start training
    trainer = Trainer(config)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        trainer.cleanup_empty_outputs()
    except Exception:
        trainer.cleanup_empty_outputs()
        raise


if __name__ == '__main__':
    main()
