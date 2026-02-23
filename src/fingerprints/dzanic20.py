"""
Dzanic20 fingerprint extraction method implementation.

Based on "Fourier Spectrum Discrepancies in Deep Network Generated Images" by Dzanic et al. (2020).

This method analyzes high-frequency Fourier modes, showing GANs fail to reproduce the steep 
spectral decay of real images. A model based on the high-frequency log-power decay rate 
achieves ~99.2% accuracy.
"""

import torch
import torch.nn as nn
import torch.fft
import numpy as np
from typing import Tuple, Optional, List
from scipy.optimize import fmin
from scipy.ndimage import gaussian_filter1d

from .base import FingerprintExtractor, AnalyticApproximation


def process_image_matlab_style(img: np.ndarray, rthresh: float = 0.50) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Process image to compute radial Fourier coefficients (MATLAB-style implementation).
    
    Args:
        img: Grayscale image as numpy array
        rthresh: Threshold above which to azimuthally average Fourier coefficients
        
    Returns:
        r: Radial distances
        m: Magnitudes of Fourier coefficients
        dc: DC gain at r=0
    """
    # Compute 2D FFT and shift to center
    F = np.fft.fftshift(np.fft.fft2(img))
    
    nx = int(F.shape[1] / 2)
    ny = int(F.shape[0] / 2)
    r_max = np.sqrt(nx**2 + ny**2)
    
    rlist = []
    maglist = []
    dc = 0.0
    
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            r_ij = np.sqrt((i - ny)**2 + (j - nx)**2)
            
            if r_ij > rthresh * r_max:
                rlist.append(r_ij)
                if not np.isnan(np.abs(F[i, j])):
                    maglist.append(np.abs(F[i, j]))
                else:
                    maglist.append(0.0)
            
            if r_ij == 0:
                dc = np.abs(F[i, j])  # DC gain at r = 0
    
    return np.array(rlist), np.array(maglist), dc


def bin_spectrum(x: np.ndarray, y: np.ndarray, nbins: int = 200, rthresh: float = 0.50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin the spectrum along radial direction.
    
    Args:
        x: Radial distances
        y: Magnitudes
        nbins: Number of bins
        rthresh: Radial threshold
        
    Returns:
        xh: Binned radial distances
        yh: Binned magnitudes
    """
    xmin, xmax = np.min(x), np.max(x)
    
    # Create bin edges
    xh_edges = np.linspace(xmin, xmax, nbins + 1)
    yh = np.zeros(nbins)
    nh = np.zeros(nbins)
    
    # Compute bins by rolling average
    for i in range(len(y)):
        xcurr = x[i]
        for j in range(nbins):
            if xcurr >= xh_edges[j] and xcurr <= xh_edges[j + 1]:
                yh[j] += y[i]
                nh[j] += 1
                break
    
    # Average within each bin
    yh = yh / np.maximum(nh, 1)  # Avoid division by zero
    
    # Set binned radius to average of adjacent points
    xh = []
    for i in range(nbins):
        xh.append((xh_edges[i] + xh_edges[i + 1]) / 2.0)
    
    # If binned coeff == 0, set to average of adjacent points
    for i in range(1, nbins - 1):
        if yh[i] == 0:
            yh[i] = 0.5 * (yh[i - 1] + yh[i + 1])
    
    return np.array(xh), yh


def fit_power_law(x: np.ndarray, y: np.ndarray, yi: float, pnorm: int = 2) -> float:
    """
    Fit power law decay to binned spectrum.
    
    Args:
        x: Radial distances (normalized)
        y: Magnitudes
        yi: Initial magnitude
        pnorm: Norm for fitting
        
    Returns:
        Decay exponent
    """
    def objective_function(c, x, y, yi):
        return yi * ((x / x[0]) ** c[0])
    
    def loss_function(c):
        return np.linalg.norm(y - objective_function(c, x, y, yi), pnorm)
    
    # Initial guess for decay exponent
    c0 = [-2.0]
    
    # Optimize
    result = fmin(loss_function, c0, maxfun=10000, disp=False)
    return result[0]


def get_fit_coeffs(x: np.ndarray, y: np.ndarray, dc: float, nsmooth: int = 5, 
                   rthresh_fit: float = 0.75, pnorm: int = 2) -> List[float]:
    """
    Get power law decay coefficients.
    
    Args:
        x: Radial distances
        y: Magnitudes
        dc: DC gain
        nsmooth: Smoothing window
        rthresh_fit: Threshold for fitting
        pnorm: Norm for fitting
        
    Returns:
        [decay_exponent, initial_magnitude, final_magnitude]
    """
    # Offset from threshold = 0.5 to fitting threshold (using 200 bins)
    nstart = int(200 * (rthresh_fit - 0.5) / 0.5) + 1
    ystart = y[0]
    
    # Smooth the spectrum
    y_smooth = gaussian_filter1d(y, nsmooth)
    
    # Normalize and select fitting range
    x_fit = x[nstart:] / np.max(x)
    y_fit = y_smooth[nstart:]
    
    yi = y_fit[0]
    yf = y_fit[-1]
    
    # Fit power law
    decay_exp = fit_power_law(x_fit, y_fit, yi, pnorm)
    
    return [decay_exp, yi, yf]


@FingerprintExtractor.register("dzanic20")
class Dzanic20(FingerprintExtractor):
    """
    Dzanic20 fingerprint extractor using high-frequency Fourier spectrum analysis.
    
    Extracts power law decay parameters from the high-frequency Fourier spectrum
    to distinguish real from generated images.
    """
    
    def __init__(self, rthresh: float = 0.50, rthresh_fit: float = 0.75, 
                 nbins: int = 200, nsmooth: int = 5, pnorm: int = 2):
        """
        Initialize Dzanic20 extractor.
        
        Args:
            rthresh: Threshold above which to azimuthally average Fourier coefficients
            rthresh_fit: Threshold above which to fit coefficients
            nbins: Number of bins for binning Fourier coefficients
            nsmooth: Smoothing window for binned coefficients
            pnorm: Norm for decay fitting
        """
        super().__init__(
            method_name="dzanic20",
            is_differentiable=False,
            has_analytic_approx=True,
            feature_dim=3  # [decay_exponent, initial_magnitude, final_magnitude]
        )
        
        self.rthresh = rthresh
        self.rthresh_fit = rthresh_fit
        self.nbins = nbins
        self.nsmooth = nsmooth
        self.pnorm = pnorm
    
    def extract_fingerprint(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract high-frequency Fourier spectrum features from images.
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Fingerprint features tensor of shape (N, 3)
        """
        batch_size = images.shape[0]
        features = []
        
        # Convert to numpy for processing
        images_np = images.detach().cpu().numpy()
        
        for i in range(batch_size):
            # Convert to grayscale
            if images_np[i].shape[0] == 3:  # RGB
                img_gray = np.dot(images_np[i].transpose(1, 2, 0), [0.2989, 0.5870, 0.1140])
            else:  # Already grayscale
                img_gray = images_np[i][0]
            
            # Process image to get radial Fourier coefficients
            r, m, dc = process_image_matlab_style(img_gray, self.rthresh)
            
            # Bin the spectrum
            xh, yh = bin_spectrum(r, m, self.nbins, self.rthresh)
            
            # Get power law fit coefficients
            coeffs = get_fit_coeffs(xh, yh, dc, self.nsmooth, self.rthresh_fit, self.pnorm)
            
            features.append(coeffs)
        
        return torch.tensor(features, dtype=torch.float32, device=images.device)


class Dzanic20_Approx(AnalyticApproximation):
    """
    Differentiable approximation of Dzanic20 fingerprint extractor.
    
    Uses PyTorch operations to approximate the power law fitting process
    in a differentiable manner.
    """
    
    def __init__(self, original_extractor: Dzanic20 = None, rthresh: float = 0.50, 
                 rthresh_fit: float = 0.75, nbins: int = 100, nsmooth: int = 5,
                 use_fast_approx: bool = True):
        """
        Initialize differentiable Dzanic20 extractor.
        
        Args:
            original_extractor: Original Dzanic20 extractor (optional)
            rthresh: Threshold above which to azimuthally average Fourier coefficients
            rthresh_fit: Threshold above which to fit coefficients
            nbins: Number of bins for binning Fourier coefficients (reduced from 200 for speed)
            nsmooth: Smoothing window for binned coefficients
            use_fast_approx: Whether to use fast approximation mode (trades accuracy for speed)
        """
        if original_extractor is None:
            # Create a default original extractor if none provided
            original_extractor = Dzanic20(rthresh, rthresh_fit, nbins, nsmooth)
        
        super().__init__(original_extractor)
        
        self.rthresh = rthresh
        self.rthresh_fit = rthresh_fit
        self.nbins = nbins
        self.nsmooth = nsmooth
        self.use_fast_approx = use_fast_approx
    
    def azimuthal_average_torch(self, fft_mag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute azimuthally averaged radial profile using PyTorch (optimized).
        
        Args:
            fft_mag: Magnitude of 2D FFT
            
        Returns:
            r: Radial distances
            mag: Averaged magnitudes
        """
        h, w = fft_mag.shape[-2:]
        center_y, center_x = h // 2, w // 2
        
        # Create coordinate grids more efficiently
        y_coords = torch.arange(h, device=fft_mag.device, dtype=torch.float32).view(-1, 1)
        x_coords = torch.arange(w, device=fft_mag.device, dtype=torch.float32).view(1, -1)
        
        # Compute radial distances (vectorized)
        r = torch.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        r_max = torch.sqrt(torch.tensor(center_x**2 + center_y**2, device=fft_mag.device, dtype=torch.float32))
        
        # Create mask for high-frequency components
        mask = r > self.rthresh * r_max
        
        # Extract values above threshold
        r_vals = r[mask]
        mag_vals = fft_mag[mask]
        
        return r_vals, mag_vals
    
    def bin_spectrum_torch(self, r: torch.Tensor, mag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bin the spectrum along radial direction using PyTorch (optimized version).
        
        Args:
            r: Radial distances
            mag: Magnitudes
            
        Returns:
            r_binned: Binned radial distances
            mag_binned: Binned magnitudes
        """
        r_min, r_max = r.min(), r.max()
        
        # Create bins
        bin_edges = torch.linspace(r_min, r_max, self.nbins + 1, device=r.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Use torch.bucketize for efficient binning
        # bucketize returns the index of the rightmost bin edge that is <= r[i]
        bin_indices = torch.bucketize(r, bin_edges, right=True) - 1
        
        # Ensure indices are within valid range
        bin_indices = torch.clamp(bin_indices, 0, self.nbins - 1)
        
        # Use scatter_add for efficient accumulation
        mag_binned = torch.zeros(self.nbins, device=r.device)
        counts = torch.zeros(self.nbins, device=r.device)
        
        # Scatter add magnitudes and counts
        mag_binned.scatter_add_(0, bin_indices, mag)
        counts.scatter_add_(0, bin_indices, torch.ones_like(mag))
        
        # Average within each bin
        mag_binned = mag_binned / torch.clamp(counts, min=1)
        
        return bin_centers, mag_binned
    
    def fit_power_law_torch(self, x: torch.Tensor, y: torch.Tensor, yi: torch.Tensor) -> torch.Tensor:
        """
        Fit power law decay using PyTorch operations (optimized differentiable approximation).
        
        Args:
            x: Radial distances (normalized)
            y: Magnitudes
            yi: Initial magnitude
            
        Returns:
            Decay exponent
        """
        # Normalize x
        x_norm = x / x[0]
        
        # Take log for linear fitting
        log_x = torch.log(x_norm + 1e-8)
        log_y = torch.log(y + 1e-8)
        log_yi = torch.log(yi + 1e-8)
        
        # Vectorized slope computation
        # Compute all slopes at once
        log_x_diff = log_x[1:] - log_x[:-1]
        log_y_diff = log_y[1:] - log_yi
        
        # Only use valid slopes (where log_x_diff > 0)
        valid_mask = log_x_diff > 1e-8
        if valid_mask.sum() > 0:
            slopes = log_y_diff[valid_mask] / log_x_diff[valid_mask]
            # Use median for robustness
            decay_exp = torch.median(slopes)
        else:
            decay_exp = torch.tensor(-2.0, device=x.device)
        
        return decay_exp
    
    def fit_power_law_torch_fast(self, x: torch.Tensor, y: torch.Tensor, yi: torch.Tensor) -> torch.Tensor:
        """
        Fast power law fitting using simple linear regression (less accurate but much faster).
        
        Args:
            x: Radial distances (normalized)
            y: Magnitudes
            yi: Initial magnitude
            
        Returns:
            Decay exponent
        """
        # Normalize x
        x_norm = x / x[0]
        
        # Take log for linear fitting
        log_x = torch.log(x_norm + 1e-8)
        log_y = torch.log(y + 1e-8)
        
        # Simple linear regression: log(y) = a + b * log(x)
        # Use only a subset of points for speed
        n_points = len(log_x)
        if n_points > 20:
            # Sample every few points
            step = max(1, n_points // 20)
            log_x_sub = log_x[::step]
            log_y_sub = log_y[::step]
        else:
            log_x_sub = log_x
            log_y_sub = log_y
        
        # Compute linear regression coefficients
        n_sub = len(log_x_sub)
        if n_sub < 2:
            return torch.tensor(-2.0, device=x.device)
        
        # Simple least squares: b = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
        sum_x = log_x_sub.sum()
        sum_y = log_y_sub.sum()
        sum_xy = (log_x_sub * log_y_sub).sum()
        sum_x2 = (log_x_sub * log_x_sub).sum()
        
        denominator = n_sub * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-8:
            return torch.tensor(-2.0, device=x.device)
        
        decay_exp = (n_sub * sum_xy - sum_x * sum_y) / denominator
        
        return decay_exp
    
    def extract_fingerprint_approx(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract high-frequency Fourier spectrum features from images (optimized differentiable).
        
        Args:
            images: Input images tensor of shape (N, C, H, W)
            
        Returns:
            Fingerprint features tensor of shape (N, 3)
        """
        batch_size = images.shape[0]
        
        # RGB to grayscale weights
        if images.shape[1] == 3:
            rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=images.device).view(1, 3, 1, 1)
            images_gray = (images * rgb_weights).sum(dim=1, keepdim=True)
        else:
            images_gray = images
        
        # Process all images in batch
        features = []
        
        # Process each image (FFT needs to be per-image due to channel handling)
        for i in range(batch_size):
            img = images_gray[i, 0]  # Single grayscale image
            
            # Compute 2D FFT and shift to center
            fft_2d = torch.fft.fft2(img)
            fft_shifted = torch.fft.fftshift(fft_2d)
            fft_mag = torch.abs(fft_shifted)
            
            # Get azimuthally averaged spectrum
            r, mag = self.azimuthal_average_torch(fft_mag)
            
            # Bin the spectrum
            r_binned, mag_binned = self.bin_spectrum_torch(r, mag)
            
            # Apply smoothing
            if self.nsmooth > 1:
                mag_smooth = torch.nn.functional.avg_pool1d(
                    mag_binned.unsqueeze(0).unsqueeze(0), 
                    kernel_size=self.nsmooth, 
                    stride=1, 
                    padding=self.nsmooth//2
                ).squeeze()
            else:
                mag_smooth = mag_binned
            
            # Select fitting range
            nstart = int(self.nbins * (self.rthresh_fit - 0.5) / 0.5) + 1
            nstart = max(0, min(nstart, len(r_binned) - 1))
            
            x_fit = r_binned[nstart:] / r_binned.max()
            y_fit = mag_smooth[nstart:]
            
            if len(x_fit) < 2:
                # Fallback values if not enough data
                features.append(torch.tensor([-2.0, 1.0, 0.1], device=images.device))
                continue
            
            yi = y_fit[0]
            yf = y_fit[-1]
            
            # Fit power law
            if self.use_fast_approx:
                decay_exp = self.fit_power_law_torch_fast(x_fit, y_fit, yi)
            else:
                decay_exp = self.fit_power_law_torch(x_fit, y_fit, yi)
            
            features.append(torch.stack([decay_exp, yi, yf]))
        
        return torch.stack(features)
