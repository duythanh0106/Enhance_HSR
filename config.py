"""
Configuration file for Hyperspectral Super-Resolution Training
Chỉnh sửa file này để thay đổi hyperparameters
"""

import os
from datetime import datetime


def infer_dataset_name(data_root):
    """Infer dataset label from data_root path."""
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
        self.data_root = './data/Harvard'  # Đường dẫn đến dataset
        self.dataset_name = infer_dataset_name(self.data_root)  # Auto from data_root
        self.num_spectral_bands = 31  # Số bands (31 cho CAVE/Harvard)
        # Split settings (cho split.json)
        self.split_seed = 42
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1  # Nên bằng 1 - train_ratio - val_ratio
        self.regenerate_split = False  # True: tạo lại split.json theo seed/ratio ở trên
        
        # ============================================================
        # MODEL SETTINGS
        # ============================================================
        self.model_name = 'ESSA_SSAM'  # 'ESSA_Original' hoặc 'ESSA_SSAM'
        self.feature_dim = 64  # Số feature channels (64, 128, 256)
        self.upscale_factor = 4  # Upscale factor (2, 4, 8)
        
        # SSAM fusion mode (chỉ cho ESSA_SSAM)
        # Options: 'sequential', 'parallel', 'adaptive'
        self.fusion_mode = 'sequential'
        
        # ============================================================
        # TRAINING SETTINGS
        # ============================================================
        self.batch_size = 1  # Batch size (giảm nếu GPU bị OOM)
        self.patch_size = 64  # Kích thước patch để crop (64, 128, 256)
        self.num_epochs = 1000  # Số epochs
        self.num_workers = 4  # Số workers cho DataLoader
        # Virtual samples/epoch (0 = tắt, dùng số ảnh thật của split)
        self.train_virtual_samples_per_epoch = 0
        self.val_virtual_samples_per_epoch = 0
        self.gradient_clip_norm = 1.0  # 0 hoặc <0 để tắt gradient clipping
        
        # Learning rate
        self.learning_rate = 2e-4
        self.lr_scheduler = 'cosine'  # 'cosine', 'step', 'plateau'
        self.lr_min = 1e-6  # Min learning rate (cho cosine scheduler)
        self.warmup_epochs = 0  # 0 = tắt warmup
        self.warmup_start_lr = 1e-6
        
        # Optimizer
        self.optimizer = 'adam'  # 'adam' hoặc 'adamw'
        self.weight_decay = 0.0
        self.betas = (0.9, 0.999)
        
        # ============================================================
        # LOSS SETTINGS
        # ============================================================
        self.loss_type = 'combined'  # 'l1', 'l2', 'combined', 'adaptive'
        
        # Combined loss weights
        self.lambda_l1 = 1.0
        self.lambda_sam = 0.1
        self.lambda_ssim = 0.5
        # Two-phase schedule for CombinedLoss (ổn định đầu train, tăng spectral/structure sau)
        self.use_two_phase_loss = False
        self.loss_phase1_ratio = 0.4  # % đầu training ở phase 1
        self.loss_phase1_sam_scale = 0.35
        self.loss_phase1_ssim_scale = 0.25
        self.loss_phase_transition_epochs = 20  # 0 = đổi phase đột ngột
        
        # ============================================================
        # DATA AUGMENTATION
        # ============================================================
        self.use_augmentation = True
        
        # ============================================================
        # VALIDATION & CHECKPOINT
        # ============================================================
        self.validate_every = 1  # Validate mỗi N epochs
        self.save_checkpoint_every = 10  # Save checkpoint mỗi N epochs
        # Best checkpoint selection
        self.best_selection_metric = 'psnr'  # 'psnr' hoặc 'composite'
        self.best_score_weights = {
            'psnr': 0.45,
            'ssim': 0.25,
            'sam': 0.20,
            'ergas': 0.10,
        }
        self.best_score_refs = {
            'psnr': 50.0,   # normalize PSNR về [0,1]
            'sam': 10.0,    # càng nhỏ càng tốt
            'ergas': 20.0,  # càng nhỏ càng tốt
        }
        
        # Output naming
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.refresh_output_paths()
        
        # ============================================================
        # LOGGING
        # ============================================================
        self.log_interval = 10  # Log mỗi N iterations
        self.save_images = True  # Lưu sample images khi validate
        self.use_ema = False  # Exponential Moving Average for eval/checkpoint
        self.ema_decay = 0.999
        
        # ============================================================
        # RESUME TRAINING
        # ============================================================
        self.resume = False
        self.resume_checkpoint = ''  # Path to checkpoint để resume
        
        # ============================================================
        # EVALUATION
        # ============================================================
        # (Reserved for future evaluation options)
        
        # ============================================================
        # HARDWARE
        # ============================================================
        self.device = 'auto'  # 'auto', 'cuda', 'mps', 'cpu'
        self.mixed_precision = False  # Use AMP (Automatic Mixed Precision)
        self.seed = 42  # Random seed
    
    def create_dirs(self):
        """Tạo các thư mục cần thiết"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'images'), exist_ok=True)

    def refresh_output_paths(self):
        """Refresh dataset label and output directories from current settings."""
        self.dataset_name = infer_dataset_name(self.data_root)
        self.experiment_name = (
            f'{self.model_name}_{self.dataset_name}_x{self.upscale_factor}_{self.timestamp}'
        )
        self.checkpoint_dir = os.path.join('./checkpoints', self.experiment_name)
        self.log_dir = os.path.join('./logs', self.experiment_name)
    
    def print_config(self):
        """In ra configuration"""
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
        
        print("\n📁 Output Settings:")
        print(f"  Experiment: {self.experiment_name}")
        print(f"  Checkpoint Dir: {self.checkpoint_dir}")
        print(f"  Log Dir: {self.log_dir}")
        
        print("\n" + "=" * 70)
    
    def to_dict(self):
        """Convert config to dictionary để save"""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and not callable(v)}


# Predefined configurations cho các experiments khác nhau

class ConfigBaseline(Config):
    """Configuration cho ESSA gốc (baseline)"""
    def __init__(self):
        super().__init__()
        self.model_name = 'ESSA_Original'
        self.feature_dim = 128
        self.loss_type = 'l1'
        self.refresh_output_paths()


class ConfigProposed(Config):
    """Configuration cho ESSA-SSAM đề xuất"""
    def __init__(self):
        super().__init__()
        self.model_name = 'ESSA_SSAM'
        self.feature_dim = 128
        self.fusion_mode = 'sequential'
        self.loss_type = 'combined'
        self.lambda_l1 = 1.0
        self.lambda_sam = 0.1
        self.lambda_ssim = 0.5
        self.refresh_output_paths()


class ConfigSpecTrans(Config):
    """Configuration cho ESSA-SSAM-SpecTrans (FINAL PROPOSED) ⭐"""
    def __init__(self):
        super().__init__()
        self.model_name = 'ESSA_SSAM_SpecTrans'
        self.feature_dim = 128
        self.fusion_mode = 'sequential'
        self.use_spectrans = True
        self.spectrans_depth = 2  # Number of Spectral Transformer blocks
        self.loss_type = 'combined'
        self.lambda_l1 = 1.0
        self.lambda_sam = 0.1
        self.lambda_ssim = 0.5
        self.refresh_output_paths()


class ConfigAblation(Config):
    """Configuration cho ablation study"""
    def __init__(self, ablation_type='parallel'):
        super().__init__()
        self.model_name = 'ESSA_SSAM'
        self.feature_dim = 128
        self.fusion_mode = ablation_type  # 'sequential', 'parallel', 'adaptive'
        self.loss_type = 'combined'
        self.refresh_output_paths()


class ConfigLightweight(Config):
    """Configuration cho lightweight model (faster training)"""
    def __init__(self):
        super().__init__()
        self.model_name = 'ESSA_SSAM'
        self.feature_dim = 128  # Giảm feature dim
        self.batch_size = 8
        self.patch_size = 64
        self.fusion_mode = 'sequential'
        self.refresh_output_paths()


class ConfigUniversalBest(ConfigSpecTrans):
    """
    Config khuyến nghị để train ổn trên cả dataset nhiều ảnh (Harvard/CAVE)
    và dataset 1 ảnh lớn (Chikusei/Pavia).
    """
    def __init__(self):
        super().__init__()

        # Common robust defaults
        self.optimizer = 'adamw'
        self.weight_decay = 1e-4
        self.learning_rate = 1e-4
        self.lr_min = 1e-7
        self.batch_size = 1
        self.patch_size = 64
        self.num_epochs = 400
        self.use_augmentation = True
        self.train_virtual_samples_per_epoch = 0
        self.val_virtual_samples_per_epoch = 0
        self.gradient_clip_norm = 1.0
        self.use_ema = True
        self.ema_decay = 0.999
        self.warmup_epochs = 10
        self.warmup_start_lr = 1e-6
        self.use_two_phase_loss = True
        self.loss_phase1_ratio = 0.4
        self.loss_phase1_sam_scale = 0.3
        self.loss_phase1_ssim_scale = 0.25
        self.loss_phase_transition_epochs = 20
        self.best_selection_metric = 'composite'
        self.best_score_weights = {
            'psnr': 0.45,
            'ssim': 0.25,
            'sam': 0.20,
            'ergas': 0.10,
        }

        self.apply_dataset_profile()
        self.refresh_output_paths()

    def apply_dataset_profile(self):
        dataset_key = infer_dataset_name(self.data_root).lower()
        is_single_scene = ('chikusei' in dataset_key) or ('pavia' in dataset_key)

        if is_single_scene:
            # 1-scene datasets cần nhiều patch/epoch để học ổn định.
            self.train_virtual_samples_per_epoch = 4000
            self.val_virtual_samples_per_epoch = 512
            self.num_epochs = 600
            self.patch_size = 64
            self.batch_size = 1
            self.num_workers = 0
            self.warmup_epochs = 20
            self.ema_decay = 0.9995
            self.loss_phase1_ratio = 0.5
            self.loss_phase_transition_epochs = 30
        else:
            # Multi-scene datasets thường không cần virtual sampling.
            self.train_virtual_samples_per_epoch = 0
            self.val_virtual_samples_per_epoch = 0
            self.num_epochs = 1000
            self.patch_size = 64
            self.batch_size = 1
            self.num_workers = 4
            self.warmup_epochs = 10
            self.ema_decay = 0.999
            self.loss_phase1_ratio = 0.35
            self.loss_phase_transition_epochs = 15


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
