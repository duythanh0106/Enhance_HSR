"""
Evaluation Script for Hyperspectral Super-Resolution
Dùng để test model và tính metrics trên test set
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from data.dataset import HyperspectralTestDataset
from utils.metrics import MetricsCalculator
from models.essa_original import ESSA
from models.essa_improved import ESSA_SSAM
from models.essa_ssam_spectrans import ESSA_SSAM_SpecTrans


class Evaluator:
    """Evaluator class để đánh giá model"""
    
    def __init__(self, checkpoint_path, data_root, dataset_type='CAVE', 
                 save_results=True, save_images=True):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            data_root: Path to test data
            dataset_type: 'CAVE' or 'Harvard'
            save_results: Whether to save results to JSON
            save_images: Whether to save reconstructed images
        """
        self.checkpoint_path = checkpoint_path
        self.data_root = data_root
        self.dataset_type = dataset_type
        self.save_results = save_results
        self.save_images = save_images
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        print("Loading checkpoint...")
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = self.checkpoint.get('config', {})
        
        # Build model
        self.model = self.build_model()
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        # Build dataset
        self.test_loader = self.build_dataloader()
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator(data_range=1.0)
        
        # Results directory
        self.results_dir = os.path.join('./results', 
                                       os.path.basename(os.path.dirname(checkpoint_path)))
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"Results will be saved to: {self.results_dir}")
    
    def build_model(self):
        """Build model from checkpoint config"""
        model_name = self.config.get('model_name', 'ESSA_SSAM')
        num_bands = self.config.get('num_spectral_bands', 31)
        feature_dim = self.config.get('feature_dim', 128)
        upscale = self.config.get('upscale_factor', 4)
        
        if model_name == 'ESSA_Original':
            model = ESSA(inch=num_bands, dim=feature_dim, upscale=upscale)
        elif model_name == 'ESSA_SSAM':
            fusion_mode = self.config.get('fusion_mode', 'sequential')
            model = ESSA_SSAM(
                inch=num_bands, 
                dim=feature_dim, 
                upscale=upscale,
                fusion_mode=fusion_mode
            )
        elif model_name == 'ESSA_SSAM_SpecTrans':
            fusion_mode = self.config.get('fusion_mode', 'sequential')
            use_spectrans = self.config.get('use_spectrans', True)
            spectrans_depth = self.config.get('spectrans_depth', 2)
            model = ESSA_SSAM_SpecTrans(
                inch=num_bands,
                dim=feature_dim,
                upscale=upscale,
                fusion_mode=fusion_mode,
                use_spectrans=use_spectrans,
                spectrans_depth=spectrans_depth
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = model.to(self.device)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {model_name}")
        print(f"Parameters: {num_params:,}")
        
        return model
    
    def build_dataloader(self):
        """Build test dataloader"""
        upscale = self.config.get('upscale_factor', 4)
        num_bands = self.config.get('num_spectral_bands', 31)
        
        test_dataset = HyperspectralTestDataset(
            data_root=self.data_root,
            dataset_type=self.dataset_type,
            upscale=upscale,
            num_bands=num_bands
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Process one image at a time
            shuffle=False,
            num_workers=0
        )
        
        return test_loader
    
    def evaluate(self):
        """Run evaluation on test set"""
        print("\n" + "="*70)
        print("Starting Evaluation")
        print("="*70)
        
        all_metrics = []
        
        with torch.no_grad():
            for i, (lr, hr, filepath) in enumerate(tqdm(self.test_loader, desc='Evaluating')):
                lr = lr.to(self.device)
                hr = hr.to(self.device)
                
                # Forward pass
                sr = self.model(lr)
                
                # Calculate metrics
                metrics = self.metrics_calculator.calculate_all(
                    sr, hr, scale=self.config.get('upscale_factor', 4)
                )
                
                # Add filename
                metrics['filename'] = os.path.basename(filepath[0])
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
        
        # Calculate average metrics
        avg_metrics = self.calculate_average_metrics(all_metrics)
        
        # Print summary
        self.print_summary(avg_metrics, all_metrics)
        
        # Save results
        if self.save_results:
            self.save_results_json(avg_metrics, all_metrics)
        
        return avg_metrics, all_metrics
    
    def calculate_average_metrics(self, all_metrics):
        """Calculate average of all metrics"""
        avg_metrics = {
            'PSNR': np.mean([m['PSNR'] for m in all_metrics]),
            'SSIM': np.mean([m['SSIM'] for m in all_metrics]),
            'SAM': np.mean([m['SAM'] for m in all_metrics]),
            'ERGAS': np.mean([m['ERGAS'] for m in all_metrics])
        }
        return avg_metrics
    
    def print_summary(self, avg_metrics, all_metrics):
        """Print evaluation summary"""
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
        """Save results to JSON file"""
        results = {
            'checkpoint': self.checkpoint_path,
            'config': self.config,
            'average_metrics': avg_metrics,
            'per_image_metrics': all_metrics
        }
        
        json_path = os.path.join(self.results_dir, 'evaluation_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {json_path}")
    
    def save_image(self, sr_image, filename, idx):
        """
        Save reconstructed hyperspectral image
        
        Args:
            sr_image: [C, H, W] tensor
            filename: Original filename
            idx: Image index
        """
        # Convert to numpy
        sr_np = sr_image.cpu().numpy()  # [C, H, W]
        
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


def compare_models(checkpoint1, checkpoint2, data_root, dataset_type='CAVE'):
    """
    So sánh 2 models
    Useful cho ablation study hoặc so sánh baseline vs proposed
    """
    print("Comparing two models...")
    print("="*70)
    
    # Evaluate model 1
    print("\nModel 1:")
    evaluator1 = Evaluator(checkpoint1, data_root, dataset_type, 
                           save_results=False, save_images=False)
    avg1, _ = evaluator1.evaluate()
    
    # Evaluate model 2
    print("\nModel 2:")
    evaluator2 = Evaluator(checkpoint2, data_root, dataset_type,
                           save_results=False, save_images=False)
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
    parser = argparse.ArgumentParser(description='Evaluate Hyperspectral SR Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--dataset_type', type=str, default='CAVE',
                       choices=['CAVE', 'Harvard'],
                       help='Dataset type')
    parser.add_argument('--save_images', action='store_true',
                       help='Save reconstructed images')
    parser.add_argument('--compare', type=str, default=None,
                       help='Path to second checkpoint for comparison')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare two models
        compare_models(args.checkpoint, args.compare, args.data_root, args.dataset_type)
    else:
        # Single model evaluation
        evaluator = Evaluator(
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            dataset_type=args.dataset_type,
            save_results=True,
            save_images=args.save_images
        )
        
        evaluator.evaluate()


if __name__ == '__main__':
    main()