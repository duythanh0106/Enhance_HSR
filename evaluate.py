"""
Evaluation script cho Hyperspectral Super-Resolution — tính PSNR/SSIM/SAM/ERGAS trên test set.

Chạy: python evaluate.py --checkpoint ./checkpoints/best.pth --data_root ./data/CAVE
      python evaluate.py --checkpoint ckpt1.pth --compare ckpt2.pth --data_root ./data/CAVE

QUAN TRỌNG:
  - Dùng forward_chop (sliding-window) để xử lý ảnh lớn hơn GPU memory
  - --crop_border (default): bỏ scale pixels viền — chuẩn paper-style evaluation
  - Kết quả lưu vào ./results/<experiment_name>/evaluation_results.json + summary.txt
  - MPS runtime error được catch và tự động fallback sang CPU
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from config import infer_dataset_name
from data.dataset import HyperspectralTestDataset, build_split_kwargs, load_dataset_with_fallback
from models.factory import build_model_from_config, load_state_dict_compat
from utils.inference import forward_chop
from utils.metrics import MetricsCalculator
from utils.device import resolve_device


class Evaluator:
    """Evaluator class để đánh giá model"""
    
    def __init__(self, checkpoint_path, data_root,
                 save_results=True, save_images=True,
                 crop_border=True, chop_patch_size=32, chop_overlap=8):
        """Khởi tạo Evaluator — load checkpoint, build dataset và model.

        Args:
            checkpoint_path: Đường dẫn tới file .pth checkpoint.
            data_root: Thư mục chứa test data và split.json.
            save_results: True để lưu JSON + TXT kết quả (mặc định True).
            save_images: True để lưu SR images dưới dạng .npy + RGB PNG.
            crop_border: True để bỏ scale pixels viền (paper-style evaluation).
            chop_patch_size: Kích thước LR patch cho sliding-window inference (mặc định 32).
            chop_overlap: Overlap giữa các patches (mặc định 8).
        """
        self.checkpoint_path = checkpoint_path
        self.data_root = data_root
        self.dataset_name = infer_dataset_name(data_root)
        self.save_results = save_results
        self.save_images = save_images
        self.crop_border = bool(crop_border)
        self.chop_patch_size = int(chop_patch_size)
        self.chop_overlap = int(chop_overlap)
        
        # Load checkpoint
        print("Loading checkpoint...")
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.config = self.checkpoint.get('config', {})
        if not isinstance(self.config, dict):
            self.config = {
                k: v for k, v in vars(self.config).items()
                if not k.startswith('_') and not callable(v)
            }
        self.checkpoint_num_bands = None
        self.checkpoint_data_root = None
        self.checkpoint_num_bands = self.config.get('num_spectral_bands')
        self.checkpoint_data_root = self.config.get('data_root')
        self.device = resolve_device(self.config.get('device', 'auto'))
        
        # Build dataset
        self.test_loader = self.build_dataloader()

        # Build model (after dataset so num_bands can be auto-detected)
        self.model = self.build_model()
        _, converted_keys = load_state_dict_compat(
            self.model, self.checkpoint['model_state_dict'], strict=True
        )
        if converted_keys:
            print(f"Converted {len(converted_keys)} legacy weight tensors for compatibility.")
        self.model.eval()
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator(data_range=1.0)
        
        # Results directory
        self.results_dir = os.path.join('./results', 
                                       os.path.basename(os.path.dirname(checkpoint_path)))
        
        print(f"Results will be saved to: {self.results_dir}")

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

    def cleanup_empty_results_dir(self):
        """Xóa results_dir nếu rỗng — gọi khi evaluation bị interrupt."""
        self._remove_dir_if_effectively_empty(self.results_dir)
    
    def build_model(self):
        """Build và di chuyển model lên device từ config trong checkpoint."""
        model_name = self.config.get('model_name', 'ESSA_SSAM')
        num_bands = getattr(self, 'num_bands_detected', self.config.get('num_spectral_bands', 31))
        model = build_model_from_config(self.config, num_bands_override=num_bands)
        model = model.to(self.device)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {model_name}")
        print(f"Parameters: {num_params:,}")
        
        return model
    
    def build_dataloader(self):
        """Build test DataLoader từ data_root; tự detect num_bands và validate vs checkpoint."""
        split_kwargs = build_split_kwargs(
            upscale=self.config.get('upscale_factor', 4),
            split_seed=self.config.get('split_seed', 42),
            train_ratio=self.config.get('train_ratio', 0.8),
            val_ratio=self.config.get('val_ratio', 0.1),
            test_ratio=self.config.get('test_ratio', 0.1),
            force_regenerate_split=self.config.get('regenerate_split', False),
        )

        test_dataset, _ = load_dataset_with_fallback(
            dataset_cls=HyperspectralTestDataset,
            primary_split='test',
            fallback_split='train',
            log_fn=print,
            data_root=self.data_root,
            **split_kwargs,
        )

        self.num_bands_detected = test_dataset.num_bands
        if (
            self.checkpoint_num_bands is not None
            and int(self.checkpoint_num_bands) != int(self.num_bands_detected)
        ):
            raise ValueError(
                "Checkpoint spectral bands do not match evaluation dataset bands.\n"
                f"  Checkpoint bands: {self.checkpoint_num_bands}\n"
                f"  Evaluation dataset bands: {self.num_bands_detected}\n"
                f"  Checkpoint data_root: {self.checkpoint_data_root}\n"
                f"  Requested data_root: {self.data_root}\n"
                "Use a checkpoint trained on the same spectral band count as the evaluation dataset."
            )

        config_bands = self.config.get('num_spectral_bands')
        if config_bands != self.num_bands_detected:
            print(
                f"Updating num_spectral_bands: "
                f"{config_bands} -> {self.num_bands_detected} (auto-detected)"
            )
            self.config['num_spectral_bands'] = self.num_bands_detected
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Process one image at a time
            shuffle=False,
            num_workers=0
        )
        
        return test_loader
    
    def evaluate(self):
        """Chạy evaluation trên toàn test set, tính và lưu metrics.

        Returns:
            tuple: (avg_metrics dict, all_metrics list of per-image dicts).
        """
        print("\n" + "="*70)
        print("Starting Evaluation")
        print("="*70)
        
        all_metrics = []
        
        scale = self.config.get('upscale_factor', 4)
        try:
            with torch.no_grad():
                for i, (lr, hr, filepath) in enumerate(tqdm(self.test_loader, desc='Evaluating')):
                    lr = lr.to(self.device)
                    hr = hr.to(self.device)

                    sr = forward_chop(
                        self.model,
                        lr,
                        scale=scale,
                        patch_size=self.chop_patch_size,
                        overlap=self.chop_overlap,
                    )

                    if sr.shape[-2:] != hr.shape[-2:]:
                        min_h = min(sr.shape[-2], hr.shape[-2])
                        min_w = min(sr.shape[-1], hr.shape[-1])
                        sr = sr[:, :, :min_h, :min_w]
                        hr = hr[:, :, :min_h, :min_w]

                    if self.crop_border and scale > 1:
                        sr = sr[:, :, scale:-scale, scale:-scale]
                        hr = hr[:, :, scale:-scale, scale:-scale]

                    sr = sr.clamp(0.0, 1.0)
                    hr = hr.clamp(0.0, 1.0)

                    metrics = self.metrics_calculator.calculate_all(sr, hr, scale=scale)

                    # Add filename
                    metrics['filename'] = str(filepath[0])
                    all_metrics.append(metrics)

                    # Save reconstructed image if requested
                    if self.save_images:
                        self.save_image(sr[0], metrics['filename'], i)

                    # Print metrics for this image
                    if i < 5:  # Print first 5 images
                        print(f"\n{metrics['filename']}:")
                        print(f"  PSNR: {metrics['PSNR']:.2f} dB")
                        print(f"  SSIM: {metrics['SSIM']:.4f}")
                        print(f"  SAM: {metrics['SAM']:.4f}°")
                        print(f"  ERGAS: {metrics['ERGAS']:.4f}")
        except RuntimeError as runtime_err:
            if self.device.type != 'mps':
                raise
            print("⚠️ MPS runtime error detected during evaluation. Falling back to CPU.")
            print(f"   Error: {str(runtime_err).splitlines()[0]}")
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            self.device = torch.device('cpu')
            self.model = self.model.to(self.device)
            return self.evaluate()
        
        # Calculate average metrics
        avg_metrics = self.calculate_average_metrics(all_metrics)
        
        # Print summary
        self.print_summary(avg_metrics, all_metrics)
        
        # Save results
        if self.save_results:
            self.save_results_json(avg_metrics, all_metrics)
            self.save_summary_txt(avg_metrics, all_metrics)
        
        return avg_metrics, all_metrics
    
    def calculate_average_metrics(self, all_metrics):
        """Tính trung bình PSNR/SSIM/SAM/ERGAS từ list per-image metrics.

        Args:
            all_metrics: List dicts với keys PSNR/SSIM/SAM/ERGAS.

        Returns:
            dict: Trung bình PSNR/SSIM/SAM/ERGAS.
        """
        avg_metrics = {
            'PSNR': np.mean([m['PSNR'] for m in all_metrics]),
            'SSIM': np.mean([m['SSIM'] for m in all_metrics]),
            'SAM': np.mean([m['SAM'] for m in all_metrics]),
            'ERGAS': np.mean([m['ERGAS'] for m in all_metrics])
        }
        return avg_metrics
    
    def print_summary(self, avg_metrics, all_metrics):
        """In tóm tắt evaluation ra console — số ảnh và trung bình 4 metrics."""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"Total images: {len(all_metrics)}")
        print(f"\nAverage Metrics:")
        print(f"  PSNR:  {avg_metrics['PSNR']:.2f} dB")
        print(f"  SSIM:  {avg_metrics['SSIM']:.4f}")
        print(f"  SAM:   {avg_metrics['SAM']:.4f}°")
        print(f"  ERGAS: {avg_metrics['ERGAS']:.4f}")
        print("="*70)
    
    def save_results_json(self, avg_metrics, all_metrics):
        """Lưu kết quả evaluation ra file JSON trong results_dir.

        Args:
            avg_metrics: Dict các metrics trung bình (PSNR, SSIM, SAM, ERGAS).
            all_metrics: List dict metrics từng ảnh.

        Returns:
            None: Ghi file evaluation_results.json, in đường dẫn ra console.
        """
        os.makedirs(self.results_dir, exist_ok=True)
        results = {
            'checkpoint': self.checkpoint_path,
            'dataset': self.dataset_name,
            'config': self.config,
            'average_metrics': avg_metrics,
            'per_image_metrics': all_metrics
        }
        
        json_path = os.path.join(self.results_dir, 'evaluation_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {json_path}")

    def save_summary_txt(self, avg_metrics, all_metrics):
        """Lưu tóm tắt evaluation ra file TXT trong results_dir."""
        os.makedirs(self.results_dir, exist_ok=True)
        summary_path = os.path.join(self.results_dir, 'summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("EVALUATION SUMMARY\n")
            f.write("=" * 70 + "\n")
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Data root: {self.data_root}\n")
            f.write(f"Total images: {len(all_metrics)}\n\n")
            f.write("Average Metrics:\n")
            f.write(f"  PSNR : {avg_metrics['PSNR']:.2f} dB\n")
            f.write(f"  SSIM : {avg_metrics['SSIM']:.4f}\n")
            f.write(f"  SAM  : {avg_metrics['SAM']:.4f}°\n")
            f.write(f"  ERGAS: {avg_metrics['ERGAS']:.4f}\n")
            f.write("-" * 70 + "\n")
            f.write("Per-image Metrics:\n")
            for metrics in all_metrics:
                f.write(
                    f"{metrics['filename']:<30} "
                    f"PSNR: {metrics['PSNR']:.2f}  "
                    f"SSIM: {metrics['SSIM']:.4f}  "
                    f"SAM: {metrics['SAM']:.4f}  "
                    f"ERGAS: {metrics['ERGAS']:.4f}\n"
                )
            f.write("=" * 70 + "\n")
        print(f"✅ Summary saved to: {summary_path}")
    
    def save_image(self, sr_image, filename, idx):
        """Lưu SR image dưới dạng .npy và RGB PNG (bands 25/15/5) vào results_dir."""
        # Convert to numpy
        sr_np = sr_image.cpu().numpy()  # [C, H, W]
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Save as .npy file (preserves all spectral bands)
        save_name = f"{os.path.splitext(filename)[0]}_SR.npy"
        save_path = os.path.join(self.results_dir, save_name)
        np.save(save_path, sr_np)
        
        # Also save RGB visualization (using bands for R, G, B)
        # Typical: Band 25 (R), Band 15 (G), Band 5 (B) for CAVE
        if sr_np.shape[0] >= 31:
            r = sr_np[25, :, :]
            g = sr_np[15, :, :]
            b = sr_np[5, :, :]
            
            rgb = np.stack([r, g, b], axis=2)
            rgb = np.clip(rgb, 0, 1)
            
            # Save RGB image
            rgb_name = f"{os.path.splitext(filename)[0]}_SR_RGB.png"
            rgb_path = os.path.join(self.results_dir, rgb_name)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(rgb)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(rgb_path, dpi=150, bbox_inches='tight')
            plt.close()


def compare_models(checkpoint1, checkpoint2, data_root):
    """Evaluate và so sánh hai checkpoint trên cùng test set.

    Args:
        checkpoint1: Đường dẫn checkpoint thứ nhất.
        checkpoint2: Đường dẫn checkpoint thứ hai để so sánh.
        data_root: Thư mục test data chung cho cả hai model.
    """
    print("Comparing two models...")
    print("="*70)
    
    # Evaluate model 1
    print("\nModel 1:")
    evaluator1 = Evaluator(checkpoint1, data_root, save_results=False, save_images=False)
    avg1, _ = evaluator1.evaluate()
    
    # Evaluate model 2
    print("\nModel 2:")
    evaluator2 = Evaluator(checkpoint2, data_root, save_results=False, save_images=False)
    avg2, _ = evaluator2.evaluate()
    
    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"{'Metric':<10} {'Model 1':<15} {'Model 2':<15} {'Improvement':<15}")
    print("-"*70)
    
    for metric in ['PSNR', 'SSIM', 'SAM', 'ERGAS']:
        val1 = avg1[metric]
        val2 = avg2[metric]
        
        # For PSNR and SSIM, higher is better
        # For SAM and ERGAS, lower is better
        if metric in ['PSNR', 'SSIM']:
            improvement = ((val2 - val1) / val1) * 100
            better = "↑" if val2 > val1 else "↓"
        else:
            improvement = ((val1 - val2) / val1) * 100
            better = "↑" if val2 < val1 else "↓"
        
        print(f"{metric:<10} {val1:<15.4f} {val2:<15.4f} {improvement:>+6.2f}% {better}")
    
    print("="*70)


def main():
    """Parse CLI arguments và chạy Evaluator (hoặc compare_models nếu có --compare)."""
    parser = argparse.ArgumentParser(description='Evaluate Hyperspectral SR Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--save_images', action='store_true',
                       help='Save reconstructed images')
    parser.add_argument('--compare', type=str, default=None,
                       help='Path to second checkpoint for comparison')
    parser.add_argument('--chop_patch_size', type=int, default=32,
                       help='LR patch size for sliding-window inference')
    parser.add_argument('--chop_overlap', type=int, default=8,
                       help='Overlap size between LR patches')
    parser.add_argument(
        '--crop_border',
        dest='crop_border',
        action='store_true',
        help='Crop border pixels (paper-style evaluation)'
    )
    parser.add_argument(
        '--no_crop_border',
        dest='crop_border',
        action='store_false',
        help='Do not crop border pixels'
    )
    parser.set_defaults(crop_border=True)
    
    args = parser.parse_args()
    if args.chop_patch_size <= 0:
        raise ValueError(f"--chop_patch_size must be > 0, got {args.chop_patch_size}")
    if args.chop_overlap < 0:
        raise ValueError(f"--chop_overlap must be >= 0, got {args.chop_overlap}")
    if args.chop_overlap >= args.chop_patch_size:
        raise ValueError(
            f"--chop_overlap must be < --chop_patch_size "
            f"(got overlap={args.chop_overlap}, patch_size={args.chop_patch_size})"
        )
    
    if args.compare:
        # Compare two models
        compare_models(args.checkpoint, args.compare, args.data_root)
    else:
        # Single model evaluation
        evaluator = None
        try:
            evaluator = Evaluator(
                checkpoint_path=args.checkpoint,
                data_root=args.data_root,
                save_results=True,
                save_images=args.save_images,
                crop_border=args.crop_border,
                chop_patch_size=args.chop_patch_size,
                chop_overlap=args.chop_overlap,
            )
            
            evaluator.evaluate()
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user.")
            if evaluator is not None:
                evaluator.cleanup_empty_results_dir()
        except Exception:
            if evaluator is not None:
                evaluator.cleanup_empty_results_dir()
            raise


if __name__ == '__main__':
    main()
