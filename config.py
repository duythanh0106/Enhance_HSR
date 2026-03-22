"""
Configuration file for Hyperspectral Super-Resolution Training
Chỉnh sửa file này để thay đổi hyperparameters
"""

import os
from datetime import datetime


def infer_dataset_name(data_root):
    """Execute `infer_dataset_name`.

    Args:
        data_root: Input parameter `data_root`.

    Returns:
        Any: Output produced by this function.
    """
    normalized = os.path.normpath(str(data_root or "")).strip()
    if not normalized or normalized == ".":
        return "dataset"
    name = os.path.basename(normalized)
    return name or "dataset"


class Config:
    """Base configuration"""
    
    def __init__(self):
        # ============================================================
        # DATASET SETTINGS
        # ============================================================
        """Initialize the `Config` instance.

        Args:
            None.

        Returns:
            None: This method initializes state and returns no value.
        """
        self.data_root = './data/Harvard'  # Root folder containing hyperspectral .mat files.
        self.dataset_name = infer_dataset_name(self.data_root)  # Dataset label inferred from `data_root`.
        self.num_spectral_bands = 31  # Expected input/output spectral channels (auto-updated from data at runtime).
        # Split settings used to create/read `split.json`.
        self.split_seed = 42  # Random seed used for reproducible train/val/test split generation.
        self.train_ratio = 0.8  # Fraction of files assigned to training split.
        self.val_ratio = 0.1  # Fraction of files assigned to validation split.
        self.test_ratio = 0.1  # Fraction of files assigned to test split.
        self.regenerate_split = False  # If True, force re-create split.json using seed and ratios above.
        self.cache_in_memory = False  # Cache loaded hyperspectral cubes in RAM to reduce repeated disk I/O.
        
        # ============================================================
        # MODEL SETTINGS
        # ============================================================
        self.model_name = 'ESSA_SSAM'  # Model architecture key used by model factory.
        self.feature_dim = 64  # Base feature width (channel dimension inside network blocks).
        self.upscale_factor = 4  # Super-resolution scale factor (e.g., 2/4/8).
        
        # SSAM attention fusion strategy (used by ESSA_SSAM variants).
        # Options: 'sequential', 'parallel', 'adaptive'
        self.fusion_mode = 'sequential'  # How spatial and spectral attention outputs are fused.
        
        # ============================================================
        # TRAINING SETTINGS
        # ============================================================
        self.batch_size = 1  # Number of LR-HR pairs per optimization step.
        self.patch_size = 64  # HR patch size sampled for training.
        self.num_epochs = 1000  # Maximum number of training epochs.
        self.num_workers = 4  # DataLoader worker processes for parallel data loading.
        # Virtual samples per epoch (0 disables and uses actual split size).
        self.train_virtual_samples_per_epoch = 0  # Effective train iterations per epoch when sampling from limited scenes.
        self.val_virtual_samples_per_epoch = 0  # Effective validation iterations per epoch.
        self.gradient_clip_norm = 1.0  # Max gradient norm; <=0 disables gradient clipping.
        
        # Learning-rate policy.
        self.learning_rate = 2e-4  # Initial/base learning rate.
        self.lr_scheduler = 'cosine'  # LR scheduler type: 'cosine', 'step', or 'plateau'.
        self.lr_min = 1e-6  # Minimum LR floor (mainly for cosine schedule).
        self.warmup_epochs = 0  # Number of warmup epochs before using base LR schedule (0 disables warmup).
        self.warmup_start_lr = 1e-6  # Starting LR used at warmup epoch 1.
        
        # Optimizer settings.
        self.optimizer = 'adamw'  # Optimizer type: 'adam' or 'adamw'.
        self.weight_decay = 0.0  # L2 regularization coefficient.
        self.betas = (0.9, 0.999)  # Adam/AdamW momentum coefficients.
        
        # ============================================================
        # LOSS SETTINGS
        # ============================================================
        self.loss_type = 'combined'  # Training loss mode: 'l1', 'l2', 'combined', or 'adaptive'.
        
        # Combined-loss component weights.
        self.lambda_l1 = 1.0  # Pixel reconstruction weight.
        self.lambda_sam = 0.1  # Spectral angle penalty weight.
        self.lambda_ssim = 0.5  # Structural similarity penalty weight.
        # Two-phase loss schedule (stability first, then stronger spectral/structure regularization).
        self.use_two_phase_loss = False  # Enable dynamic weight schedule for combined loss.
        self.loss_phase1_ratio = 0.4  # Fraction of total epochs assigned to phase-1 settings.
        self.loss_phase1_sam_scale = 0.35  # Scale factor applied to lambda_sam during phase-1.
        self.loss_phase1_ssim_scale = 0.25  # Scale factor applied to lambda_ssim during phase-1.
        self.loss_phase_transition_epochs = 20  # Transition length from phase-1 to target weights (0 = hard switch).
        
        # ============================================================
        # DATA AUGMENTATION
        # ============================================================
        self.use_augmentation = True  # Enable random flip/rotation augmentation in training dataset.
        
        # ============================================================
        # VALIDATION & CHECKPOINT
        # ============================================================
        self.validate_every = 5  # Run validation every N epochs (5 = less noisy for small val sets).
        self.save_checkpoint_every = 10  # Save periodic checkpoint every N epochs.
        # Best-checkpoint selection strategy.
        self.best_selection_metric = 'composite'  # Composite more robust than PSNR alone on small val sets.
        self.best_score_weights = {
            'psnr': 0.45,
            'ssim': 0.25,
            'sam': 0.20,
            'ergas': 0.10,
        }  # Weights used when `best_selection_metric='composite'`.
        self.best_score_refs = {
            'psnr': 50.0,   # PSNR normalization reference.
            'sam': 10.0,    # SAM normalization reference (lower-is-better metric).
            'ergas': 20.0,  # ERGAS normalization reference (lower-is-better metric).
        }
        # Early-stopping controls.
        self.use_early_stopping = False  # Stop training automatically when validation stops improving.
        self.early_stopping_patience = 50  # Allowed consecutive non-improving validations.
        self.early_stopping_min_delta = 0.0  # Minimum score improvement required to reset patience.
        self.early_stopping_start_epoch = 0  # Earliest epoch where early-stopping is allowed.
        
        # Output naming and directory keys.
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # Run identifier used in experiment/checkpoint/log paths.
        self.refresh_output_paths()
        
        # ============================================================
        # LOGGING
        # ============================================================
        self.log_interval = 10  # Progress logging frequency in training iterations.
        self.save_images = True  # Save qualitative output images during evaluation/validation workflows.
        self.use_ema = True   # EMA weights give more stable val metrics (critical for small val sets).
        self.ema_decay = 0.999  # EMA decay factor (higher = smoother but slower adaptation).
        
        # ============================================================
        # RESUME TRAINING
        # ============================================================
        self.resume = False  # Resume training state from a saved checkpoint.
        self.resume_checkpoint = ''  # Checkpoint path used when `resume=True`.
        
        # ============================================================
        # EVALUATION
        # ============================================================
        # (Reserved for future evaluation options)
        
        # ============================================================
        # HARDWARE / RUNTIME
        # ============================================================
        self.device = 'auto'  # Device preference: 'auto', 'cuda', 'mps', or 'cpu'.
        self.mixed_precision = False  # Enable AMP mixed precision (effective on CUDA).
        self.log_device_runtime = True  # Print one-time runtime device/memory snapshot during training.
        self.seed = 42  # Global randomness seed for reproducibility.
    
    def create_dirs(self):
        """Execute `create_dirs`.

        Args:
            None.

        Returns:
            None: This function returns no value.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'images'), exist_ok=True)

    def refresh_output_paths(self):
        """Execute `refresh_output_paths`.

        Args:
            None.

        Returns:
            None: This function returns no value.
        """
        self.dataset_name = infer_dataset_name(self.data_root)
        self.experiment_name = (
            f'{self.model_name}_{self.dataset_name}_x{self.upscale_factor}_{self.timestamp}'
        )
        self.checkpoint_dir = os.path.join('./checkpoints', self.experiment_name)
        self.log_dir = os.path.join('./logs', self.experiment_name)
    
    def print_config(self):
        """Execute `print_config`.

        Args:
            None.

        Returns:
            None: This function returns no value.
        """
        print("=" * 70)
        print("CONFIGURATION")
        print("=" * 70)
        
        print("\n📊 Dataset Settings:")
        print(f"  Dataset: {self.dataset_name}")
        print(f"  Data Root: {self.data_root}")
        print(f"  Spectral Bands: {self.num_spectral_bands}")
        print(f"  Split Seed: {self.split_seed}")
        print(f"  Split Ratio (train/val/test): {self.train_ratio}/{self.val_ratio}/{self.test_ratio}")
        print(f"  Regenerate Split: {self.regenerate_split}")
        print(f"  Cache In Memory: {self.cache_in_memory}")
        
        print("\n🏗️  Model Settings:")
        print(f"  Model: {self.model_name}")
        print(f"  Feature Dim: {self.feature_dim}")
        print(f"  Upscale Factor: {self.upscale_factor}")
        if self.model_name == 'ESSA_SSAM':
            print(f"  Fusion Mode: {self.fusion_mode}")
        
        print("\n🎯 Training Settings:")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Patch Size: {self.patch_size}")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Train Virtual Samples/Epoch: {self.train_virtual_samples_per_epoch}")
        print(f"  Val Virtual Samples/Epoch: {self.val_virtual_samples_per_epoch}")
        print(f"  Grad Clip Norm: {self.gradient_clip_norm}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Warmup Epochs: {self.warmup_epochs}")
        print(f"  Warmup Start LR: {self.warmup_start_lr}")
        print(f"  LR Scheduler: {self.lr_scheduler}")
        print(f"  Use EMA: {self.use_ema}")
        print(f"  Runtime Device Log: {self.log_device_runtime}")
        if self.use_ema:
            print(f"  EMA Decay: {self.ema_decay}")
        
        print("\n💡 Loss Settings:")
        print(f"  Loss Type: {self.loss_type}")
        if self.loss_type == 'combined':
            print(f"  λ_L1: {self.lambda_l1}")
            print(f"  λ_SAM: {self.lambda_sam}")
            print(f"  λ_SSIM: {self.lambda_ssim}")
            print(f"  Two-Phase Loss: {self.use_two_phase_loss}")
            if self.use_two_phase_loss:
                print(f"  Phase1 Ratio: {self.loss_phase1_ratio}")
                print(f"  Phase1 SAM Scale: {self.loss_phase1_sam_scale}")
                print(f"  Phase1 SSIM Scale: {self.loss_phase1_ssim_scale}")
                print(f"  Phase Transition Epochs: {self.loss_phase_transition_epochs}")

        print("\n🏆 Best Checkpoint Selection:")
        print(f"  Metric Mode: {self.best_selection_metric}")
        if self.best_selection_metric == 'composite':
            w_psnr = self.best_score_weights.get('psnr', 0.45)
            w_ssim = self.best_score_weights.get('ssim', 0.25)
            w_sam = self.best_score_weights.get('sam', 0.20)
            w_ergas = self.best_score_weights.get('ergas', 0.10)
            print(f"  Weights (PSNR/SSIM/SAM/ERGAS): "
                  f"{w_psnr}/{w_ssim}/{w_sam}/{w_ergas}")
        print(f"  Early Stopping: {self.use_early_stopping}")
        if self.use_early_stopping:
            print(f"  Early Stop Patience: {self.early_stopping_patience}")
            print(f"  Early Stop Min Delta: {self.early_stopping_min_delta}")
            print(f"  Early Stop Start Epoch: {self.early_stopping_start_epoch}")
        
        print("\n📁 Output Settings:")
        print(f"  Experiment: {self.experiment_name}")
        print(f"  Checkpoint Dir: {self.checkpoint_dir}")
        print(f"  Log Dir: {self.log_dir}")
        
        print("\n" + "=" * 70)
    
    def to_dict(self):
        """Execute `to_dict`.

        Args:
            None.

        Returns:
            Any: Output produced by this function.
        """
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and not callable(v)}


# Predefined configurations cho các experiments khác nhau

class ConfigBaseline(Config):
    """Configuration cho ESSA gốc (baseline)"""
    def __init__(self):
        """Initialize the `ConfigBaseline` instance.

        Args:
            None.

        Returns:
            None: This method initializes state and returns no value.
        """
        super().__init__()
        self.model_name = 'ESSA_Original'  # Switch to original ESSA architecture (baseline variant).
        self.feature_dim = 128  # Baseline channel width used in the original setup.
        self.loss_type = 'l1'  # Baseline objective for fair comparison.
        self.refresh_output_paths()


class ConfigProposed(Config):
    """Configuration cho ESSA-SSAM đề xuất"""
    def __init__(self):
        """Initialize the `ConfigProposed` instance.

        Args:
            None.

        Returns:
            None: This method initializes state and returns no value.
        """
        super().__init__()
        self.model_name = 'ESSA_SSAM'  # Use improved ESSA with SSAM blocks.
        self.feature_dim = 128  # Wider features for stronger representation capacity.
        self.fusion_mode = 'sequential'  # Apply SSAM spatial->spectral fusion in sequence.
        self.loss_type = 'combined'  # Optimize with multi-term objective.
        self.lambda_l1 = 1.0  # Pixel fidelity weight.
        self.lambda_sam = 0.1  # Spectral consistency weight.
        self.lambda_ssim = 0.5  # Structural similarity weight.
        self.refresh_output_paths()


class ConfigSpecTrans(Config):
    """Configuration cho ESSA-SSAM-SpecTrans (FINAL PROPOSED) ⭐"""
    def __init__(self):
        """Initialize the `ConfigSpecTrans` instance.

        Args:
            None.

        Returns:
            None: This method initializes state and returns no value.
        """
        super().__init__()
        self.model_name = 'ESSA_SSAM_SpecTrans'  # Use ESSA + SSAM + spectral transformer blocks.
        self.feature_dim = 192  # Feature width for the proposed full model.
        self.fusion_mode = 'sequential'  # SSAM fusion strategy.
        self.use_spectrans = True  # Enable spectral transformer refinement branch.
        self.spectrans_depth = 3  # Number of spectral transformer blocks.
        self.loss_type = 'combined'  # Multi-term objective for reconstruction + spectral structure.
        self.lambda_l1 = 1.0  # Pixel fidelity weight.
        self.lambda_sam = 0.1  # Spectral angle weight.
        self.lambda_ssim = 0.5  # Structural similarity weight.
        self.refresh_output_paths()


class ConfigAblation(Config):
    """Configuration cho ablation study"""
    def __init__(self, ablation_type='parallel'):
        """Initialize the `ConfigAblation` instance.

        Args:
            ablation_type: Input parameter `ablation_type`.

        Returns:
            None: This method initializes state and returns no value.
        """
        super().__init__()
        self.model_name = 'ESSA_SSAM'  # Keep SSAM model while changing fusion behavior for ablation.
        self.feature_dim = 128  # Fixed width for fair ablation comparison.
        self.fusion_mode = ablation_type  # Selected ablation mode: 'sequential', 'parallel', or 'adaptive'.
        self.loss_type = 'combined'  # Keep objective fixed during fusion ablation.
        self.refresh_output_paths()


class ConfigLightweight(Config):
    """Configuration cho lightweight model (faster training)"""
    def __init__(self):
        """Initialize the `ConfigLightweight` instance.

        Args:
            None.

        Returns:
            None: This method initializes state and returns no value.
        """
        super().__init__()
        self.model_name = 'ESSA_SSAM'  # Lightweight preset still uses SSAM backbone.
        self.feature_dim = 128  # Internal channel width for this faster preset.
        self.batch_size = 8  # Larger batch for faster throughput on capable hardware.
        self.patch_size = 64  # Training crop size.
        self.fusion_mode = 'sequential'  # Stable default fusion strategy.
        self.refresh_output_paths()


class ConfigUniversalBest(ConfigSpecTrans):
    """
    Config khuyến nghị để train ổn trên cả dataset nhiều ảnh (Harvard/CAVE)
    và dataset 1 ảnh lớn (Chikusei/Pavia).
    """
    def __init__(self):
        """Initialize the `ConfigUniversalBest` instance.

        Args:
            None.

        Returns:
            None: This method initializes state and returns no value.
        """
        super().__init__()

        # Common robust defaults for cross-dataset training.
        self.optimizer = 'adamw'  # Better generalization and regularization than plain Adam.
        self.weight_decay = 1e-4  # Default regularization strength.
        self.learning_rate = 1e-4  # Safer base LR for stable long training.
        self.lr_min = 1e-7  # Cosine schedule floor.
        self.batch_size = 1  # Memory-safe default for high-dimensional HSI data.
        self.patch_size = 64  # Balanced crop size for context vs. memory usage.
        self.num_epochs = 400  # Initial epoch budget before dataset profile adjustment.
        self.use_augmentation = True  # Improve generalization via random geometric transforms.
        self.train_virtual_samples_per_epoch = 0  # Use real split size by default for multi-scene datasets.
        self.val_virtual_samples_per_epoch = 0  # Use real validation size by default.
        self.cache_in_memory = False  # Default off to avoid large RAM use on multi-scene datasets.
        self.gradient_clip_norm = 1.0  # Stabilize optimization on difficult batches.
        self.use_ema = True  # Use EMA weights for more stable validation/checkpoint metrics.
        self.ema_decay = 0.999  # EMA smoothing factor.
        self.warmup_epochs = 10  # Warmup length for smoother optimization startup.
        self.warmup_start_lr = 1e-6  # LR at the first warmup step.
        self.use_two_phase_loss = True  # Start with conservative loss weights, then ramp spectral/SSIM terms.
        self.loss_phase1_ratio = 0.4  # Portion of training spent in phase-1.
        self.loss_phase1_sam_scale = 0.3  # SAM weight scale in phase-1.
        self.loss_phase1_ssim_scale = 0.25  # SSIM weight scale in phase-1.
        self.loss_phase_transition_epochs = 20  # Number of epochs for smooth phase transition.
        self.best_selection_metric = 'composite'  # Select best checkpoint by composite quality score.
        self.best_score_weights = {
            'psnr': 0.45,
            'ssim': 0.25,
            'sam': 0.20,
            'ergas': 0.10,
        }
        self.use_early_stopping = True  # Enable automatic stop when validation no longer improves.
        self.early_stopping_patience = 40  # Number of non-improving validations allowed.
        self.early_stopping_min_delta = 1e-4  # Minimum improvement threshold for reset.
        self.early_stopping_start_epoch = 20  # Delay early-stopping until model has warmed up.

        self.apply_dataset_profile()
        self.refresh_output_paths()

    def apply_dataset_profile(self):
        """Execute `apply_dataset_profile`.

        Args:
            None.

        Returns:
            None: This function returns no value.
        """
        dataset_key = infer_dataset_name(self.data_root).lower()
        is_single_scene = ('chikusei' in dataset_key) or ('pavia' in dataset_key)

        if is_single_scene:
            # Single-scene datasets need heavier random patch sampling per epoch.
            self.train_virtual_samples_per_epoch = 800  # Increase optimization steps per epoch from one scene.
            self.val_virtual_samples_per_epoch = 128  # Increase validation sampling stability.
            self.cache_in_memory = True  # Avoid reloading the same huge scene from disk every iteration.
            self.num_epochs = 200  # Lower epoch count because each epoch already has many virtual samples.
            self.patch_size = 64  # Keep patch size stable for memory and overlap behavior.
            self.batch_size = 1  # Keep memory-safe default.
            self.num_workers = 0  # Avoid dataloader overhead for single large file workflows.
            self.warmup_epochs = 20  # Longer warmup improves stability on single-scene data.
            self.ema_decay = 0.9995  # Stronger smoothing for noisy validation curves.
            self.loss_phase1_ratio = 0.5  # Spend longer in conservative loss phase.
            self.loss_phase_transition_epochs = 30  # Smoother transition to final loss weights.
            self.early_stopping_patience = 25  # Faster stop if no meaningful gain.
            self.early_stopping_start_epoch = 25  # Start early-stopping after sufficient warmup/training.
        else:
            # Multi-scene datasets usually have enough natural diversity.
            self.train_virtual_samples_per_epoch = 0  # Use true split size each epoch.
            self.val_virtual_samples_per_epoch = 0  # Use true validation size.
            self.cache_in_memory = False  # Prefer lower RAM footprint for many-scene datasets.
            self.num_epochs = 1000  # Longer budget for larger/diverse datasets.
            self.patch_size = 64  # Standard crop size.
            self.batch_size = 1  # Memory-safe default.
            self.num_workers = 4  # Parallel loading for many files.
            self.warmup_epochs = 10  # Standard warmup length.
            self.ema_decay = 0.999  # Standard EMA smoothing.
            self.loss_phase1_ratio = 0.35  # Shorter conservative phase on diverse data.
            self.loss_phase_transition_epochs = 15  # Faster transition to full loss weights.
            self.early_stopping_patience = 60  # More patience due to longer training horizon.
            self.early_stopping_start_epoch = 40  # Start early-stop after stable convergence begins.



class _ConfigDatasetBase(ConfigUniversalBest):
    """Base class cho dataset-specific configs."""

    _DATA_ROOT = './data/dataset'
    _FEATURE_DIM = 128
    _SPECTRANS_DEPTH = 2
    _UPSCALE = 4

    def __init__(self):
        super().__init__()
        self.data_root = self._DATA_ROOT
        self.feature_dim = self._FEATURE_DIM
        self.use_spectrans = True
        self.spectrans_depth = self._SPECTRANS_DEPTH
        self.upscale_factor = self._UPSCALE
        self.apply_dataset_profile()
        self._apply_subclass_profile()
        self.refresh_output_paths()

    def _apply_subclass_profile(self):
        pass

    @classmethod
    def build_x2(cls):
        obj = cls.__new__(cls)
        obj._UPSCALE = 2
        obj.__init__()
        return obj


class ConfigCAVE(_ConfigDatasetBase):
    """CAVE: 31 bands, 32 scenes.

    Fixes cho val instability:
    - val_virtual_samples_per_epoch=50: validate trên 50 patches thay vì 3 scenes
    - use_ema=True: EMA weights cho val curve mượt hơn
    - validate_every=5: bớt noise khi nhìn curve theo epoch
    - best_selection_metric='composite': ít bị ảnh hưởng bởi PSNR spike
    """
    _DATA_ROOT = './dataset/CAVE'
    _FEATURE_DIM = 128
    _SPECTRANS_DEPTH = 2

    def _apply_subclass_profile(self):
        self.learning_rate = 2e-4
        self.lr_min = 1e-7
        self.warmup_epochs = 5
        self.patch_size = 64
        self.num_workers = 0
        self.mixed_precision = False
        self.num_epochs = 1000
        self.use_ema = True
        self.ema_decay = 0.999
        self.validate_every = 5
        # Key fix: sample 50 patches từ val scenes thay vì dùng 3 scenes nguyên
        self.val_virtual_samples_per_epoch = 50
        self.best_selection_metric = 'composite'
        self.loss_phase1_ratio = 0.3
        self.loss_phase_transition_epochs = 10
        self.early_stopping_patience = 80   # patience tính theo val runs, không phải epochs
        self.early_stopping_start_epoch = 20  # = 20 val runs × 5 epochs = epoch 100


class ConfigHarvard(_ConfigDatasetBase):
    """Harvard: 31 bands, ~50 scenes."""
    _DATA_ROOT = './dataset/Harvard'
    _FEATURE_DIM = 128
    _SPECTRANS_DEPTH = 2

    def _apply_subclass_profile(self):
        self.learning_rate = 1e-4
        self.lr_min = 1e-7
        self.warmup_epochs = 10
        self.patch_size = 96
        self.num_workers = 0
        self.mixed_precision = False
        self.num_epochs = 800
        self.use_ema = True
        self.ema_decay = 0.999
        self.validate_every = 5
        self.val_virtual_samples_per_epoch = 50
        self.best_selection_metric = 'composite'
        self.loss_phase1_ratio = 0.35
        self.loss_phase_transition_epochs = 15
        self.early_stopping_patience = 60
        self.early_stopping_start_epoch = 20


class ConfigChikusei(_ConfigDatasetBase):
    """Chikusei: 128 bands, single scene."""
    _DATA_ROOT = './dataset/Chikusei'
    _FEATURE_DIM = 128
    _SPECTRANS_DEPTH = 1

    def _apply_subclass_profile(self):
        self.learning_rate = 1e-4
        self.lr_min = 1e-7
        self.warmup_epochs = 20
        self.patch_size = 64
        self.batch_size = 1
        self.num_workers = 0
        self.mixed_precision = False
        self.train_virtual_samples_per_epoch = 400
        self.val_virtual_samples_per_epoch = 80
        self.cache_in_memory = True
        self.num_epochs = 400
        self.use_ema = True
        self.ema_decay = 0.9995
        self.validate_every = 5
        self.best_selection_metric = 'composite'
        self.loss_phase1_ratio = 0.45
        self.loss_phase_transition_epochs = 25
        self.early_stopping_patience = 40
        self.early_stopping_start_epoch = 40


class ConfigPavia(_ConfigDatasetBase):
    """Pavia: 102 bands, single scene."""
    _DATA_ROOT = './dataset/Pavia'
    _FEATURE_DIM = 128
    _SPECTRANS_DEPTH = 1

    def _apply_subclass_profile(self):
        self.learning_rate = 1e-4
        self.lr_min = 1e-7
        self.warmup_epochs = 20
        self.patch_size = 48
        self.batch_size = 1
        self.num_workers = 0
        self.mixed_precision = False
        self.train_virtual_samples_per_epoch = 500
        self.val_virtual_samples_per_epoch = 80
        self.cache_in_memory = True
        self.num_epochs = 300
        self.use_ema = True
        self.ema_decay = 0.9995
        self.validate_every = 5
        self.best_selection_metric = 'composite'
        self.loss_phase1_ratio = 0.45
        self.loss_phase_transition_epochs = 25
        self.early_stopping_patience = 30
        self.early_stopping_start_epoch = 30



class ConfigBaselineCAVE(ConfigCAVE):
    """Baseline ESSA gốc trên CAVE — cùng feature_dim và training config với proposed."""
    def _apply_subclass_profile(self):
        super()._apply_subclass_profile()
        self.model_name = 'ESSA_Original'
        self.use_spectrans = False


class ConfigBaselineHarvard(ConfigHarvard):
    """Baseline ESSA gốc trên Harvard — cùng feature_dim và training config với proposed."""
    def _apply_subclass_profile(self):
        super()._apply_subclass_profile()
        self.model_name = 'ESSA_Original'
        self.use_spectrans = False


class ConfigBaselineChikusei(ConfigChikusei):
    """Baseline ESSA gốc trên Chikusei — cùng feature_dim=128 và training config với proposed."""
    def _apply_subclass_profile(self):
        super()._apply_subclass_profile()
        self.model_name = 'ESSA_Original'
        self.use_spectrans = False


class ConfigBaselinePavia(ConfigPavia):
    """Baseline ESSA gốc trên Pavia — cùng feature_dim=128 và training config với proposed."""
    def _apply_subclass_profile(self):
        super()._apply_subclass_profile()
        self.model_name = 'ESSA_Original'
        self.use_spectrans = False


CONFIG_PRESETS = {
    'default': Config,
    'baseline': ConfigBaseline,
    'proposed': ConfigProposed,
    'spectrans': ConfigSpecTrans,
    'lightweight': ConfigLightweight,
    'universal_best': ConfigUniversalBest,
    'cave': ConfigCAVE,
    'harvard': ConfigHarvard,
    'chikusei': ConfigChikusei,
    'pavia': ConfigPavia,
    # Baseline — same training config as proposed, only model differs
    'baseline_cave':     ConfigBaselineCAVE,
    'baseline_harvard':  ConfigBaselineHarvard,
    'baseline_chikusei': ConfigBaselineChikusei,
    'baseline_pavia':    ConfigBaselinePavia,
}

CONFIG_PRESET_CHOICES = tuple(CONFIG_PRESETS.keys())
_X2_DATASETS = {'cave', 'harvard', 'chikusei', 'pavia'}


def build_config(preset='default'):
    """Tạo config từ preset. Hỗ trợ suffix _x2/_x4, ví dụ 'cave_x2'."""
    key = str(preset or 'default').lower()

    scale = None
    for suffix, factor in (('_x2', 2), ('_x4', 4)):
        if key.endswith(suffix):
            key = key[:-len(suffix)]
            scale = factor
            break

    if key not in CONFIG_PRESETS:
        supported = ', '.join(
            list(CONFIG_PRESET_CHOICES) +
            [f'{d}_x2' for d in _X2_DATASETS] +
            [f'{d}_x4' for d in _X2_DATASETS]
        )
        raise ValueError(f"Unknown config preset: '{preset}'. Supported:\n  {supported}")

    cls = CONFIG_PRESETS[key]
    if scale == 2 and key in _X2_DATASETS:
        return cls.build_x2()
    return cls()


# Test code
if __name__ == '__main__':
    # Test default config
    print("Testing Default Config:")
    config = Config()
    config.print_config()
    
    print("\n\n" + "="*70)
    print("Testing Proposed Config:")
    config_proposed = ConfigProposed()
    config_proposed.print_config()
    
    print("\n\n" + "="*70)
    print("Testing Lightweight Config:")
    config_light = ConfigLightweight()
    config_light.print_config()
    
    # Save config to dict
    print("\n\n" + "="*70)
    print("Config as dictionary:")
    config_dict = config.to_dict()
    for key, value in list(config_dict.items())[:10]:
        print(f"  {key}: {value}")