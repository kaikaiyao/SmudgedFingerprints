"""
Factory for creating fingerprint extractors with proper parameter handling.
"""

from typing import Optional
from src.fingerprints import FingerprintExtractor


class FingerprintExtractorFactory:
    """Factory for creating fingerprint extractors with proper configuration."""
    
    # List of fingerprint methods that accept device parameter
    DEVICE_ACCEPTING_METHODS = {
        'wang20', 'nataraj19', 'song24', 'giudice21', 'corvi23r', 'corvi23s'
    }
    
    @classmethod
    def create(
        cls,
        fingerprint_method: str,
        device: str,
        real_data_path: Optional[str] = None,
        cache_dir: str = "cache",
        data_dir: Optional[str] = None
    ) -> FingerprintExtractor:
        """
        Create fingerprint extractor with proper parameter handling for different methods.
        
        Args:
            fingerprint_method: Name of the fingerprint method
            device: Device to use ('cpu', 'cuda', 'mps')
            real_data_path: Path to real data (for Song24 methods)
            cache_dir: Cache directory (for Song24 methods)
            data_dir: Data directory (for Song24 methods)
            
        Returns:
            Configured fingerprint extractor
        """
        
        # Handle Song24 methods that require real_data_path
        if fingerprint_method.startswith('song24'):
            # For Song24 methods, real_data_path is optional (will auto-download if not provided)
            return FingerprintExtractor.create(
                fingerprint_method, 
                device=device,
                real_data_path=real_data_path,
                cache_dir=cache_dir,
                data_dir=data_dir
            )
        elif fingerprint_method in cls.DEVICE_ACCEPTING_METHODS:
            # For methods that accept device parameter
            return FingerprintExtractor.create(fingerprint_method, device=device)
        else:
            # For methods that don't accept device parameter (like durall20, nowroozi22, etc.)
            return FingerprintExtractor.create(fingerprint_method)
    
    @classmethod
    def is_implicit_method(
        cls,
        fingerprint_method: str,
        device: str,
        real_data_path: Optional[str] = None,
        cache_dir: str = "cache",
        data_dir: Optional[str] = None
    ) -> bool:
        """
        Check if a fingerprint method is implicit (computes features on-the-fly).
        
        Args:
            fingerprint_method: Name of the fingerprint method
            device: Device to use
            
        Returns:
            True if the method is implicit, False otherwise
        """
        try:
            extractor = cls.create(
                fingerprint_method,
                device,
                real_data_path=real_data_path,
                cache_dir=cache_dir,
                data_dir=data_dir
            )
            return extractor.is_implicit_fingerprint
        except Exception:
            # Fallback: assume it's not implicit if we can't check
            return False
