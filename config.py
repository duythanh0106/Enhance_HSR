"""
Configuration file for Hyperspectral Super-Resolution Training
Chỉnh sửa file này để thay đổi hyperparameters
"""

import os
from datetime import datetime


class Config:
    """Base configuration"""
    
    def __init__(self):
        # ============================================================
        # DATASET SETTINGS
        # ============================================================
        self.dataset_type = 'Harvard'  # 'CAVE' hoặc 'Harvard'
        self.data_root = './data/Harvard'  # Đường dẫn đến dataset
        self.num_spectral_bands = 31  # Số bands (31 cho CAVE/Harvard)
        
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
        self.num_epochs = 100  # Số epochs
        self.num_workers = 4  # Số workers cho DataLoader
        
        # Learning rate
        self.learning_rate = 2e-4
        self.lr_scheduler = 'cosine'  # 'cosine', 'step', 'plateau'
        self.lr_min = 1e-6  # Min learning rate (cho cosine scheduler)
        
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
        
        # ============================================================
        # DATA AUGMENTATION
        # ============================================================
        self.use_augmentation = True
        self.horizontal_flip = True
        self.vertical_flip = True
        self.rotation = True  # Random 90, 180, 270 rotation
        
        # ============================================================
        # VALIDATION & CHECKPOINT
        # ============================================================
        self.val_split = 0.2  # 20% for validation
        self.validate_every = 1  # Validate mỗi N epochs
        self.save_checkpoint_every = 10  # Save checkpoint mỗi N epochs
        
        # Checkpoint directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_name = f'{self.model_name}_{self.dataset_type}_x{self.upscale_factor}_{timestamp}'
        self.checkpoint_dir = os.path.join('./checkpoints', self.experiment_name)
        
        # ============================================================
        # LOGGING
        # ============================================================
        self.log_dir = os.path.join('./logs', self.experiment_name)
        self.log_interval = 10  # Log mỗi N iterations
        self.save_images = True  # Lưu sample images khi validate
        self.num_save_images = 4  # Số lượng images để lưu
        
        # ============================================================
        # RESUME TRAINING
        # ============================================================
        self.resume = False
        self.resume_checkpoint = ''  # Path to checkpoint để resume
        
        # ============================================================
        # EVALUATION
        # ============================================================
        self.eval_metrics = ['PSNR', 'SSIM', 'SAM', 'ERGAS']
        
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
    
    def print_config(self):
        """In ra configuration"""
        print("=" * 70)
        print("CONFIGURATION")
        print("=" * 70)
        
        print("\n📊 Dataset Settings:")
        print(f"  Dataset Type: {self.dataset_type}")
        print(f"  Data Root: {self.data_root}")
        print(f"  Spectral Bands: {self.num_spectral_bands}")
        
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
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  LR Scheduler: {self.lr_scheduler}")
        
        print("\n💡 Loss Settings:")
        print(f"  Loss Type: {self.loss_type}")
        if self.loss_type == 'combined':
            print(f"  λ_L1: {self.lambda_l1}")
            print(f"  λ_SAM: {self.lambda_sam}")
            print(f"  λ_SSIM: {self.lambda_ssim}")
        
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


class ConfigAblation(Config):
    """Configuration cho ablation study"""
    def __init__(self, ablation_type='parallel'):
        super().__init__()
        self.model_name = 'ESSA_SSAM'
        self.feature_dim = 128
        self.fusion_mode = ablation_type  # 'sequential', 'parallel', 'adaptive'
        self.loss_type = 'combined'


class ConfigLightweight(Config):
    """Configuration cho lightweight model (faster training)"""
    def __init__(self):
        super().__init__()
        self.model_name = 'ESSA_SSAM'
        self.feature_dim = 64  # Giảm feature dim
        self.batch_size = 8
        self.patch_size = 64
        self.fusion_mode = 'sequential'


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
