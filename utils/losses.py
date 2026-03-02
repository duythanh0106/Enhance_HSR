"""
Loss Functions for Hyperspectral Image Super-Resolution
Bao gồm: L1, L2, SAM Loss, Combined Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class L1Loss(nn.Module):
    """Standard L1 Loss"""
    def __init__(self):
        """Initialize the `L1Loss` instance.

        Args:
            None.

        Returns:
            None: This method initializes state and returns no value.
        """
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, pred, target):
        """Run the forward computation for this module.

        Args:
            pred: Input parameter `pred`.
            target: Input parameter `target`.

        Returns:
            Any: Output produced by this function.
        """
        return self.loss(pred, target)


class L2Loss(nn.Module):
    """Standard L2 (MSE) Loss"""
    def __init__(self):
        """Initialize the `L2Loss` instance.

        Args:
            None.

        Returns:
            None: This method initializes state and returns no value.
        """
        super(L2Loss, self).__init__()
        self.loss = nn.MSELoss()
    
    def forward(self, pred, target):
        """Run the forward computation for this module.

        Args:
            pred: Input parameter `pred`.
            target: Input parameter `target`.

        Returns:
            Any: Output produced by this function.
        """
        return self.loss(pred, target)


class SAMLoss(nn.Module):
    """
    Spectral Angle Mapper (SAM) Loss
    Đặc biệt quan trọng cho hyperspectral images!
    
    SAM đo góc giữa spectral signatures - giữ nguyên hình dạng phổ
    """
    def __init__(self, eps=1e-8):
        """Initialize the `SAMLoss` instance.

        Args:
            eps: Input parameter `eps`.

        Returns:
            None: This method initializes state and returns no value.
        """
        super(SAMLoss, self).__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        """Run the forward computation for this module.

        Args:
            pred: Input parameter `pred`.
            target: Input parameter `target`.

        Returns:
            Any: Output produced by this function.
        """
        # Reshape to [B, C, H*W]
        B, C, H, W = pred.shape
        pred_flat = pred.reshape(B, C, -1)
        target_flat = target.reshape(B, C, -1)
        
        # Calculate dot product
        dot_product = torch.sum(pred_flat * target_flat, dim=1)
        
        # Calculate norms
        norm_pred = torch.sqrt(torch.sum(pred_flat ** 2, dim=1))
        norm_target = torch.sqrt(torch.sum(target_flat ** 2, dim=1))
        
        # Calculate cosine
        cos_theta = dot_product / (norm_pred * norm_target + self.eps)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        
        # Calculate angle (in radians)
        sam = torch.acos(cos_theta)
        
        # Return mean SAM
        return torch.mean(sam)


class SSIMLoss(nn.Module):
    """
    SSIM Loss for better perceptual quality
    """
    def __init__(self, window_size=11, data_range=1.0):
        """Initialize the `SSIMLoss` instance.

        Args:
            window_size: Input parameter `window_size`.
            data_range: Input parameter `data_range`.

        Returns:
            None: This method initializes state and returns no value.
        """
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.data_range = data_range
        
        # Create Gaussian window
        sigma = 1.5
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2 / (2*sigma**2)) 
                              for x in range(window_size)])
        gauss = gauss / gauss.sum()
        
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('window', _2D_window)
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2
    
    def forward(self, pred, target):
        """Run the forward computation for this module.

        Args:
            pred: Input parameter `pred`.
            target: Input parameter `target`.

        Returns:
            Any: Output produced by this function.
        """
        C = pred.size(1)
        window = self.window.expand(C, 1, self.window_size, self.window_size).contiguous()
        window = window.to(pred.device)
        
        # Calculate means
        mu1 = F.conv2d(pred, window, padding=self.window_size//2, groups=C)
        mu2 = F.conv2d(target, window, padding=self.window_size//2, groups=C)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Calculate variances
        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size//2, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size//2, groups=C) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size//2, groups=C) - mu1_mu2
        
        # Calculate SSIM
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        
        # Return 1 - SSIM as loss
        return 1 - ssim_map.mean()


class CombinedLoss(nn.Module):
    """
    Combined Loss = λ1*L1 + λ2*SAM + λ3*SSIM
    
    Đây là loss function được đề xuất trong khóa luận!
    Kết hợp:
    - L1: Pixel-wise accuracy
    - SAM: Spectral fidelity
    - SSIM: Structural similarity
    """
    def __init__(self, lambda_l1=1.0, lambda_sam=0.1, lambda_ssim=0.5):
        """Initialize the `CombinedLoss` instance.

        Args:
            lambda_l1: Input parameter `lambda_l1`.
            lambda_sam: Input parameter `lambda_sam`.
            lambda_ssim: Input parameter `lambda_ssim`.

        Returns:
            None: This method initializes state and returns no value.
        """
        super(CombinedLoss, self).__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_sam = lambda_sam
        self.lambda_ssim = lambda_ssim
        
        self.l1_loss = L1Loss()
        self.sam_loss = SAMLoss()
        self.ssim_loss = SSIMLoss()

    def set_weights(self, lambda_l1=None, lambda_sam=None, lambda_ssim=None):
        """Execute `set_weights`.

        Args:
            lambda_l1: Input parameter `lambda_l1`.
            lambda_sam: Input parameter `lambda_sam`.
            lambda_ssim: Input parameter `lambda_ssim`.

        Returns:
            None: This function returns no value.
        """
        if lambda_l1 is not None:
            self.lambda_l1 = float(lambda_l1)
        if lambda_sam is not None:
            self.lambda_sam = float(lambda_sam)
        if lambda_ssim is not None:
            self.lambda_ssim = float(lambda_ssim)

    def get_weights(self):
        """Execute `get_weights`.

        Args:
            None.

        Returns:
            Any: Output produced by this function.
        """
        return {
            'lambda_l1': float(self.lambda_l1),
            'lambda_sam': float(self.lambda_sam),
            'lambda_ssim': float(self.lambda_ssim),
        }
    
    def forward(self, pred, target):
        """Run the forward computation for this module.

        Args:
            pred: Input parameter `pred`.
            target: Input parameter `target`.

        Returns:
            Any: Output produced by this function.
        """
        # Calculate individual losses
        l1 = self.l1_loss(pred, target)
        sam = self.sam_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        
        # Combined loss
        total_loss = (self.lambda_l1 * l1 + 
                     self.lambda_sam * sam + 
                     self.lambda_ssim * ssim)
        
        # Return both total loss and components (for logging)
        loss_dict = {
            'total': total_loss.item(),
            'l1': l1.item(),
            'sam': sam.item(),
            'ssim': ssim.item()
        }
        
        return total_loss, loss_dict


class AdaptiveCombinedLoss(nn.Module):
    """
    Adaptive Combined Loss với learnable weights
    Tự động học optimal weights cho từng loss component
    
    Đây là một cải tiến nâng cao có thể đề cập trong khóa luận!
    """
    def __init__(self):
        """Initialize the `AdaptiveCombinedLoss` instance.

        Args:
            None.

        Returns:
            None: This method initializes state and returns no value.
        """
        super(AdaptiveCombinedLoss, self).__init__()
        
        # Learnable log variance parameters
        self.log_var_l1 = nn.Parameter(torch.zeros(1))
        self.log_var_sam = nn.Parameter(torch.zeros(1))
        self.log_var_ssim = nn.Parameter(torch.zeros(1))
        
        self.l1_loss = L1Loss()
        self.sam_loss = SAMLoss()
        self.ssim_loss = SSIMLoss()
    
    def forward(self, pred, target):
        """Run the forward computation for this module.

        Args:
            pred: Input parameter `pred`.
            target: Input parameter `target`.

        Returns:
            Any: Output produced by this function.
        """
        # Calculate individual losses
        l1 = self.l1_loss(pred, target)
        sam = self.sam_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        
        # Adaptive weighting
        precision_l1 = torch.exp(-self.log_var_l1)
        precision_sam = torch.exp(-self.log_var_sam)
        precision_ssim = torch.exp(-self.log_var_ssim)
        
        # Combined loss with uncertainty
        total_loss = (precision_l1 * l1 + self.log_var_l1 +
                     precision_sam * sam + self.log_var_sam +
                     precision_ssim * ssim + self.log_var_ssim)
        
        loss_dict = {
            'total': total_loss.item(),
            'l1': l1.item(),
            'sam': sam.item(),
            'ssim': ssim.item(),
            'weight_l1': precision_l1.item(),
            'weight_sam': precision_sam.item(),
            'weight_ssim': precision_ssim.item()
        }
        
        return total_loss, loss_dict


# Test code
if __name__ == '__main__':
    print("Testing Loss Functions...")
    print("=" * 70)
    
    # Create test data
    B, C, H, W = 2, 31, 128, 128
    pred = torch.rand(B, C, H, W)
    target = torch.rand(B, C, H, W)
    
    # Test L1 Loss
    print("\n1. L1 Loss")
    print("-" * 70)
    l1_loss = L1Loss()
    loss = l1_loss(pred, target)
    print(f"L1 Loss: {loss.item():.4f}")
    
    # Test L2 Loss
    print("\n2. L2 Loss")
    print("-" * 70)
    l2_loss = L2Loss()
    loss = l2_loss(pred, target)
    print(f"L2 Loss: {loss.item():.4f}")
    
    # Test SAM Loss
    print("\n3. SAM Loss")
    print("-" * 70)
    sam_loss = SAMLoss()
    loss = sam_loss(pred, target)
    print(f"SAM Loss: {loss.item():.4f} radians")
    print(f"SAM Loss: {loss.item() * 180 / math.pi:.4f} degrees")
    
    # Test SSIM Loss
    print("\n4. SSIM Loss")
    print("-" * 70)
    ssim_loss = SSIMLoss()
    loss = ssim_loss(pred, target)
    print(f"SSIM Loss: {loss.item():.4f}")
    
    # Test Combined Loss
    print("\n5. Combined Loss")
    print("-" * 70)
    combined_loss = CombinedLoss(lambda_l1=1.0, lambda_sam=0.1, lambda_ssim=0.5)
    total_loss, loss_dict = combined_loss(pred, target)
    print(f"Total Loss: {loss_dict['total']:.4f}")
    print(f"  - L1: {loss_dict['l1']:.4f}")
    print(f"  - SAM: {loss_dict['sam']:.4f}")
    print(f"  - SSIM: {loss_dict['ssim']:.4f}")
    
    # Test Adaptive Combined Loss
    print("\n6. Adaptive Combined Loss")
    print("-" * 70)
    adaptive_loss = AdaptiveCombinedLoss()
    
    # Simulate a few training steps
    optimizer = torch.optim.Adam(adaptive_loss.parameters(), lr=0.001)
    
    for step in range(5):
        optimizer.zero_grad()
        total_loss, loss_dict = adaptive_loss(pred, target)
        total_loss.backward()
        optimizer.step()
        
        if step % 2 == 0:
            print(f"\nStep {step}:")
            print(f"  Total Loss: {loss_dict['total']:.4f}")
            print(f"  Weights - L1: {loss_dict['weight_l1']:.4f}, "
                  f"SAM: {loss_dict['weight_sam']:.4f}, "
                  f"SSIM: {loss_dict['weight_ssim']:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ All loss function tests completed!")
    print("\nRecommendation for thesis:")
    print("  - Use CombinedLoss with λ1=1.0, λ2=0.1, λ3=0.5 as baseline")
    print("  - Can try AdaptiveCombinedLoss for advanced experiments")
    print("  - SAM loss is crucial for hyperspectral fidelity!")
