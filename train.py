"""
Main training script cho Hyperspectral Super-Resolution.

Cung cấp hai class chính:
  ModelEMA — Exponential Moving Average tracker cho validation/checkpoint ổn định
  Trainer  — Quản lý toàn bộ training loop: data, model, optimizer, scheduler, checkpoint

Chạy: python train.py --config cave
      python train.py --config cave_x2 --data_root ./data/CAVE
      python train.py --config default --resume ./checkpoints/exp/latest.pth

QUAN TRỌNG:
  - build_config(preset) load config từ config.py — xem danh sách presets với --help
  - Checkpoint lưu atomic (tmp + os.replace) để tránh corrupt nếu crash
  - EMA weights dùng cho validation/best checkpoint, không dùng khi train
"""

import os
import argparse
import random
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from config import CONFIG_PRESET_CHOICES, build_config
from data.dataset import HyperspectralDataset, build_split_kwargs, load_dataset_with_fallback
from models.factory import build_model_from_config, load_state_dict_compat
from utils.metrics import MetricsCalculator
from utils.losses import L1Loss, L2Loss, CombinedLoss, AdaptiveCombinedLoss
from utils.device import resolve_device
from utils.scoring import compute_selection_score_from_config
from utils.time_utils import format_duration


class ModelEMA:
    """EMA tracker cho tham số model, dùng cho validation/checkpoint ổn định hơn."""

    def __init__(self, model, decay=0.999):
        """Khởi tạo EMA shadow weights từ model hiện tại.

        Args:
            model: Model có parameters cần theo dõi.
            decay: Hệ số EMA — shadow = decay*shadow + (1-decay)*param (mặc định 0.999).
        """
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup = None

    def update(self, model):
        """Cập nhật shadow weights từ model parameters sau mỗi train step."""
        one_minus_decay = 1.0 - self.decay
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=one_minus_decay)

    def apply_shadow(self, model):
        """Thay thế model parameters bằng EMA shadow weights — backup params trước."""
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Khôi phục model parameters từ backup (sau khi apply_shadow)."""
        if self.backup is None:
            return
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = None

    def state_dict(self):
        """Trả về dict chứa decay và shadow weights để lưu vào checkpoint."""
        return {
            'decay': self.decay,
            'shadow': {k: v.detach().clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state):
        """Khôi phục EMA state từ checkpoint dict (decay và shadow weights)."""
        self.decay = float(state.get('decay', self.decay))
        loaded_shadow = state.get('shadow', {})
        if loaded_shadow:
            self.shadow = {k: v.detach().clone() for k, v in loaded_shadow.items()}


class Trainer:
    """Trainer class cho training và validation"""
    
    def __init__(self, config):
        """Khởi tạo Trainer — build datasets, model, optimizer, scheduler, loss.

        Args:
            config: Config object với tất cả hyperparameters (xem config.py).
        """
        self.config = config

        self.device = resolve_device(config.device)
        self.use_amp = bool(config.mixed_precision and self.device.type == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.log_file = None
        self._device_runtime_logged = False

        # Open log file early — ALL _log() calls (build, dataloader, model) are recorded.
        # train() calls _open_log_file() again but it is a no-op if already open.
        self._open_log_file()

        # Log config vào file ngay sau khi log mở — dùng self._log để ghi cả stdout lẫn file.
        self.config.print_config(print_fn=self._log)

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
        self.no_improve_validations = 0
        self.train_losses = []
        self.val_metrics = []
        
        # Resume if needed
        if config.resume and config.resume_checkpoint:
            self.load_checkpoint(config.resume_checkpoint)
    
    def set_seed(self, seed):
        """Đặt random seed cho torch, numpy và Python random để reproducibility."""
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        np.random.seed(seed)
        random.seed(seed)  # seed Python built-in random (dùng bởi nhiều augmentation lib)

    def _set_learning_rate(self, lr):
        """Đặt learning rate trực tiếp cho tất cả optimizer param groups."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = float(lr)

    def _apply_warmup(self, epoch):
        """Điều chỉnh LR theo warmup schedule; trả về True nếu epoch đang trong warmup."""
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
        """Delegate sang utils.format_duration — chuyển giây thành HH:MM:SS."""
        return format_duration(seconds)

    @staticmethod
    def _remove_dir_if_effectively_empty(path):
        """Xóa thư mục path nếu rỗng (bỏ qua .DS_Store)."""
        if not os.path.isdir(path):
            return
        entries = [e for e in os.listdir(path) if e != '.DS_Store']
        if not entries:
            ds_store = os.path.join(path, '.DS_Store')
            if os.path.exists(ds_store):
                os.remove(ds_store)
            os.rmdir(path)

    def cleanup_empty_outputs(self):
        """Xóa các thư mục output rỗng khi train bị interrupt hoặc lỗi sớm.

        Log file KHÔNG bị xóa nếu đã có nội dung — giữ lại thông tin
        debug khi training crash giữa chừng.
        """
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

        # Chỉ xóa images dir nếu rỗng
        self._remove_dir_if_effectively_empty(os.path.join(self.config.log_dir, 'images'))

        # Chỉ xóa log_dir nếu KHÔNG có training.log (hoặc log rỗng)
        log_path = os.path.join(self.config.log_dir, 'training.log')
        log_has_content = os.path.exists(log_path) and os.path.getsize(log_path) > 0
        if not log_has_content:
            self._remove_dir_if_effectively_empty(self.config.log_dir)

        # Chỉ xóa checkpoint_dir nếu rỗng
        self._remove_dir_if_effectively_empty(self.config.checkpoint_dir)

    def _ensure_output_dirs(self):
        """Tạo checkpoint_dir và log_dir nếu chưa tồn tại."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)

    def _open_log_file(self):
        """Mở training.log trong append mode — no-op nếu đã mở."""
        self._ensure_output_dirs()
        if self.log_file is None:
            log_path = os.path.join(self.config.log_dir, 'training.log')
            self.log_file = open(log_path, 'a', encoding='utf-8', buffering=1)
            self.log_file.write("=== Training Log Started ===\n")

    def _log(self, message):
        """Print message và ghi vào training.log."""
        print(message)
        if self.log_file is not None:
            self.log_file.write(message + "\n")

    def _log_runtime_device_snapshot(self, lr, hr, sr, loss):
        """Log một lần duy nhất thông tin device và MPS memory khi bắt đầu train."""
        if self._device_runtime_logged:
            return
        if not bool(getattr(self.config, 'log_device_runtime', True)):
            return

        model_device = next(self.model.parameters()).device
        self._log(
            "Runtime Device Check | "
            f"model={model_device}, lr={lr.device}, hr={hr.device}, "
            f"sr={sr.device}, loss={loss.device}"
        )

        if self.device.type == 'mps':
            fallback_env = os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0 (default)')
            self._log(f"MPS fallback env (PYTORCH_ENABLE_MPS_FALLBACK): {fallback_env}")
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'current_allocated_memory'):
                try:
                    allocated = torch.mps.current_allocated_memory()
                    if hasattr(torch.mps, 'driver_allocated_memory'):
                        driver_allocated = torch.mps.driver_allocated_memory()
                        self._log(
                            f"MPS memory | current_allocated={allocated} bytes, "
                            f"driver_allocated={driver_allocated} bytes"
                        )
                    else:
                        self._log(f"MPS memory | current_allocated={allocated} bytes")
                except Exception:
                    self._log("MPS memory stats unavailable on this PyTorch build.")

        self._device_runtime_logged = True
    
    def build_model(self):
        """Build model từ config, di chuyển lên device, log params và device."""
        model = build_model_from_config(self.config)
        model = model.to(self.device)
        
        # Print model info
        num_params = sum(p.numel() for p in model.parameters())
        self._log(f"Model: {self.config.model_name}")
        self._log(f"Parameters: {num_params:,}")
        self._log(f"Using device: {self.device}")
        
        return model
    
    def build_dataloaders(self):
        """Build train và val DataLoaders; tự detect num_bands và sync vào config."""

        split_kwargs = build_split_kwargs(
            upscale=self.config.upscale_factor,
            split_seed=self.config.split_seed,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
            force_regenerate_split=self.config.regenerate_split,
        )

        # Dùng 'trainval' nếu val_ratio=0 hoặc config set use_trainval=True
        use_trainval = (
            getattr(self.config, 'use_trainval', False)
            or float(self.config.val_ratio) == 0.0
        )
        train_split = 'trainval' if use_trainval else 'train'
        normalization_mode = getattr(self.config, 'normalization_mode', 'per_image_minmax')
        normalization_scale = float(getattr(self.config, 'normalization_scale', 65535.0))

        train_dataset = HyperspectralDataset(
            data_root=self.config.data_root,
            patch_size=self.config.patch_size,
            augment=self.config.use_augmentation,
            split=train_split,
            virtual_samples_per_epoch=self.config.train_virtual_samples_per_epoch,
            cache_in_memory=bool(getattr(self.config, 'cache_in_memory', False)),
            normalization_mode=normalization_mode,
            normalization_scale=normalization_scale,
            **split_kwargs
        )

        if use_trainval:
            # Không có val set riêng — dùng lại train set để monitor
            self._log("INFO: val_ratio=0 hoac use_trainval=True -> dung train set de validate.")
            val_dataset = HyperspectralDataset(
                data_root=self.config.data_root,
                patch_size=self.config.patch_size,
                augment=False,
                split=train_split,
                virtual_samples_per_epoch=min(
                    50, self.config.val_virtual_samples_per_epoch or 50
                ),
                cache_in_memory=bool(getattr(self.config, 'cache_in_memory', False)),
                normalization_mode=normalization_mode,
                normalization_scale=normalization_scale,
                **split_kwargs,
            )
        else:
            val_dataset, _ = load_dataset_with_fallback(
                dataset_cls=HyperspectralDataset,
                primary_split='val',
                fallback_split='train',
                log_fn=self._log,
                data_root=self.config.data_root,
                patch_size=self.config.patch_size,
                augment=False,
                virtual_samples_per_epoch=self.config.val_virtual_samples_per_epoch,
                cache_in_memory=bool(getattr(self.config, 'cache_in_memory', False)),
                normalization_mode=normalization_mode,
                normalization_scale=normalization_scale,
                **split_kwargs,
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
        """Build optimizer theo config; khôi phục state nếu rebuild giữa chừng."""
        previous_optimizer_state = None
        previous_optimizer = getattr(self, 'optimizer', None)
        if previous_optimizer is not None:
            try:
                previous_optimizer_state = previous_optimizer.state_dict()
            except Exception:
                previous_optimizer_state = None

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

        if previous_optimizer_state is not None:
            try:
                optimizer.load_state_dict(previous_optimizer_state)
                self._log("Restored optimizer state after rebuild.")
            except Exception as exc:
                self._log(f"⚠️ Failed to restore optimizer state after rebuild: {exc}")
        
        return optimizer
    
    def build_scheduler(self):
        """Build LR scheduler theo config; khôi phục state nếu cùng loại khi rebuild."""
        previous_scheduler_state = None
        previous_scheduler_type = None
        previous_scheduler = getattr(self, 'scheduler', None)
        if previous_scheduler is not None:
            try:
                previous_scheduler_state = previous_scheduler.state_dict()
                previous_scheduler_type = type(previous_scheduler)
            except Exception:
                previous_scheduler_state = None
                previous_scheduler_type = None

        # Registry: add new scheduler types here — no other code needs to change.
        _SCHEDULER_REGISTRY = {
            'cosine':  lambda: optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.num_epochs, eta_min=self.config.lr_min,
            ),
            'step':    lambda: optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(getattr(self.config, 'lr_step_size', 30)),
                gamma=float(getattr(self.config, 'lr_step_gamma', 0.5)),
            ),
            'plateau': lambda: optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=10,
            ),
        }
        factory = _SCHEDULER_REGISTRY.get(self.config.lr_scheduler)
        scheduler = factory() if factory is not None else None

        if (
            scheduler is not None
            and previous_scheduler_state is not None
            and previous_scheduler_type is type(scheduler)
        ):
            try:
                scheduler.load_state_dict(previous_scheduler_state)
                self._log("Restored scheduler state after rebuild.")
            except Exception as exc:
                self._log(f"⚠️ Failed to restore scheduler state after rebuild: {exc}")
        
        return scheduler
    
    def build_loss(self):
        """Build loss function theo config.loss_type."""
        # Registry: add new loss types here — no other code needs to change.
        _LOSS_REGISTRY = {
            'l1':       lambda: L1Loss(),
            'l2':       lambda: L2Loss(),
            'combined': lambda: CombinedLoss(
                lambda_l1=self.config.lambda_l1,
                lambda_sam=self.config.lambda_sam,
                lambda_ssim=self.config.lambda_ssim,
            ),
            'adaptive': lambda: AdaptiveCombinedLoss(),
        }
        factory = _LOSS_REGISTRY.get(self.config.loss_type)
        if factory is None:
            raise ValueError(
                f"Unknown loss type '{self.config.loss_type}'. "
                f"Known: {sorted(_LOSS_REGISTRY)}"
            )
        return factory()

    def _get_two_phase_lambdas(self, epoch):
        """Tính lambda_l1/sam/ssim và tên phase cho two-phase loss schedule.

        Args:
            epoch: Epoch hiện tại (1-indexed).

        Returns:
            tuple: (lambda_l1, lambda_sam, lambda_ssim, phase_name) trong đó
                   phase_name là 'phase1', 'phase2', hoặc 'transition'.
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
        """Áp dụng two-phase loss schedule nếu được bật.

        Args:
            epoch: Epoch hiện tại (1-indexed).

        Returns:
            tuple: (scheduled, phase, lambda_l1, lambda_sam, lambda_ssim) —
                   scheduled=False nếu loss không hỗ trợ set_weights hoặc chưa bật.
        """
        if not hasattr(self.criterion, 'set_weights'):
            return False, 'static', None, None, None
        if not getattr(self.config, 'use_two_phase_loss', False):
            return False, 'static', self.config.lambda_l1, self.config.lambda_sam, self.config.lambda_ssim

        l1, sam, ssim, phase = self._get_two_phase_lambdas(epoch)
        self.criterion.set_weights(lambda_l1=l1, lambda_sam=sam, lambda_ssim=ssim)
        return True, phase, l1, sam, ssim

    def compute_selection_score(self, metrics):
        """Tính selection score từ metrics dict theo config.best_selection_metric.

        Args:
            metrics: Dict chứa PSNR, SSIM, SAM, ERGAS.

        Returns:
            float: Scalar score — cao hơn là tốt hơn.
        """
        return compute_selection_score_from_config(metrics, self.config)

    def _is_early_stopping_enabled(self):
        """Trả về True nếu early stopping được bật trong config."""
        return bool(getattr(self.config, 'use_early_stopping', False))

    def _early_stopping_patience(self):
        """Trả về số validations không cải thiện cho phép trước khi dừng."""
        return max(1, int(getattr(self.config, 'early_stopping_patience', 1)))

    def _early_stopping_start_epoch(self):
        """Trả về epoch đầu tiên mà early stopping bắt đầu được áp dụng."""
        return max(0, int(getattr(self.config, 'early_stopping_start_epoch', 0)))

    def _early_stopping_min_delta(self):
        """Trả về ngưỡng cải thiện tối thiểu để reset bộ đếm early stopping."""
        return float(getattr(self.config, 'early_stopping_min_delta', 0.0))
    
    def train_epoch(self, epoch):
        """Chạy một epoch train với AMP và gradient clipping.

        Args:
            epoch: Epoch hiện tại (1-indexed), dùng cho progress bar.

        Returns:
            tuple: (avg_loss, train_time) — loss trung bình và thời gian epoch (giây).
        """
        epoch_start = time.time()
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch}/{self.config.num_epochs}',
            disable=bool(getattr(self.config, 'quiet_tqdm', False)),
        )
        
        for i, (lr, hr) in enumerate(pbar):
            lr = lr.to(self.device, non_blocking=self.device.type == 'cuda')
            hr = hr.to(self.device, non_blocking=self.device.type == 'cuda')
            
            # Forward pass
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                sr = self.model(lr)
            
                # All loss functions return (loss, loss_dict) — unpack uniformly.
                loss, loss_dict = self.criterion(sr, hr)
                if i % self.config.log_interval == 0:
                    postfix = {'loss': f"{loss_dict['total']:.4f}"}
                    postfix.update({
                        k: f"{v:.4f}" for k, v in loss_dict.items()
                        if k != 'total' and isinstance(v, float)
                    })
                    pbar.set_postfix(postfix)

            if i == 0:
                self._log_runtime_device_snapshot(lr, hr, sr, loss)
            
            # Backward pass
            if self.use_amp:
                if not torch.isfinite(loss):
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
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
                if not torch.isfinite(loss):
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
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
        """Đánh giá model trên val set với EMA weights (nếu có); trả về avg metrics."""
        if self.ema is not None:
            self.ema.apply_shadow(self.model)
        try:
            self.model.eval()
            
            total_psnr = 0.0
            total_ssim = 0.0
            total_sam = 0.0
            total_ergas = 0.0
            
            with torch.no_grad():
                for lr, hr in tqdm(
                    self.val_loader,
                    desc='Validating',
                    disable=bool(getattr(self.config, 'quiet_tqdm', False)),
                ):
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
        """Lưu checkpoint atomic vào latest.pth và (nếu is_best) best.pth.

        Args:
            epoch: Epoch hiện tại.
            metrics: Dict metrics validation (PSNR, SSIM, SAM, ERGAS).
            is_best: Nếu True, copy sang best.pth.
            selection_score: Score đã tính sẵn; tính lại nếu None và is_best=True.
        """
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
            'no_improve_validations': self.no_improve_validations,
            'config': self.config.to_dict()
        }
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
        
        def _atomic_save(obj, path):
            """Write to .tmp then rename — prevents corrupt files on crash."""
            tmp = path + '.tmp'
            torch.save(obj, tmp)
            os.replace(tmp, path)

        latest_path = os.path.join(self.config.checkpoint_dir, 'latest.pth')
        _atomic_save(checkpoint, latest_path)

        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best.pth')
            _atomic_save(checkpoint, best_path)
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

        if epoch % self.config.save_checkpoint_every == 0:
            epoch_path = os.path.join(self.config.checkpoint_dir, f'epoch_{epoch}.pth')
            _atomic_save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Khôi phục model, optimizer, scheduler và training state từ checkpoint.

        Args:
            checkpoint_path: Đường dẫn tới file .pth checkpoint.
        """
        # weights_only=False vì checkpoint chứa config object (Python dict/class).
        # Chỉ load từ nguồn đáng tin cậy.
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
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
        self.no_improve_validations = int(checkpoint.get('no_improve_validations', 0))
        self._log(f"Resumed from epoch {self.current_epoch}")
    
    def train(self):
        """Chạy toàn bộ training loop với validation, early stopping và checkpoint."""
        self._open_log_file()
        self._log("\n" + "="*70)
        self._log("Starting Training")
        self._log("="*70)
        if self._is_early_stopping_enabled():
            self._log(
                f"Early stopping enabled | patience={self._early_stopping_patience()} "
                f"| min_delta={self._early_stopping_min_delta()} "
                f"| start_epoch={self._early_stopping_start_epoch()}"
            )

        training_start = time.time()
        early_stop_triggered = False
        early_stop_epoch = None
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
                    min_delta = self._early_stopping_min_delta()
                    is_best = selection_score > (self.best_score + min_delta)
                    if is_best:
                        self.best_score = selection_score
                        self.best_psnr = val_metrics['PSNR']
                        self.best_epoch = epoch
                        self.no_improve_validations = 0
                    else:
                        if self._is_early_stopping_enabled() and epoch >= self._early_stopping_start_epoch():
                            self.no_improve_validations += 1
                    
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
                    if self._is_early_stopping_enabled() and epoch >= self._early_stopping_start_epoch():
                        self._log(
                            f"  Early Stop Count: {self.no_improve_validations}/"
                            f"{self._early_stopping_patience()}"
                        )
                self._log(f"  Epoch Total Time: {self.format_duration(epoch_total_time)} ({epoch_total_time:.2f} seconds)")
                self._log("=" * 70 + "\n")
                
                # Update learning rate
                if self.scheduler:
                    if self.config.lr_scheduler == 'plateau' and val_metrics is not None:
                        self.scheduler.step(val_metrics['PSNR'])
                    elif self.config.lr_scheduler != 'plateau' and not in_warmup:
                        self.scheduler.step()

                if (
                    val_metrics is not None
                    and self._is_early_stopping_enabled()
                    and epoch >= self._early_stopping_start_epoch()
                    and self.no_improve_validations >= self._early_stopping_patience()
                ):
                    early_stop_triggered = True
                    early_stop_epoch = epoch
                    self._log(
                        f"⏹ Early stopping triggered at epoch {epoch} "
                        f"(no improvement for {self.no_improve_validations} validations)."
                    )
                    break

            total_train_time = time.time() - training_start
            best_ckpt_path = os.path.join(self.config.checkpoint_dir, 'best.pth')
            latest_ckpt_path = os.path.join(self.config.checkpoint_dir, 'latest.pth')
            log_path = os.path.join(self.config.log_dir, 'training.log')
            self._log("\n" + "="*70)
            self._log("Training Completed!")
            self._log(f"Best PSNR: {self.best_psnr:.2f} dB")
            mode = str(getattr(self.config, 'best_selection_metric', 'psnr')).upper()
            self._log(f"Best Score ({mode}): {self.best_score:.6f} (epoch {self.best_epoch})")
            if early_stop_triggered:
                self._log(f"Stopped Early At Epoch: {early_stop_epoch}")
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
    """Parse CLI args, build config, khởi tạo Trainer và bắt đầu training."""
    parser = argparse.ArgumentParser(description='Train Hyperspectral SR Model')
    parser.add_argument('--config', type=str, default='default',
                       help=(
                           'Config preset. Built-in: ' +
                           ', '.join(CONFIG_PRESET_CHOICES) +
                           '. Dataset presets: cave, harvard, chikusei, pavia '
                           '(thêm _x2 hoặc _x4, ví dụ: cave_x2, pavia_x4).'
                       ))
    parser.add_argument('--data_root', type=str, default=None,
                       help='Override data root path')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume')
    
    args = parser.parse_args()
    
    # Load configuration
    config = build_config(args.config)
    
    # Override with command line arguments
    if args.data_root:
        config.data_root = args.data_root
        if hasattr(config, 'apply_dataset_profile'):
            config.apply_dataset_profile()
        config.refresh_output_paths()
    
    if args.resume:
        config.resume = True
        config.resume_checkpoint = args.resume

    # Create trainer and start training (Trainer.__init__ logs config to file + stdout)
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
