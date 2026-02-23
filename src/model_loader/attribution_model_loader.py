"""
Attribution model loading functionality.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


class AttributionModelLoader:
    """Handles loading of attribution models with proper configuration."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def load_true_attribution_model(
        self,
        model_path: str,
        fingerprint_method: str,
        num_models: int,
        feature_dim: int,
        is_implicit: bool,
        real_data_path: Optional[str] = None,
        cache_dir: str = "cache",
        data_dir: Optional[str] = None
    ) -> torch.nn.Module:
        """
        Load the true attribution model (h) or implicit fingerprint model.
        
        Args:
            model_path: Path to the saved model
            fingerprint_method: Fingerprint method name
            num_models: Number of models/classes
            feature_dim: Feature dimension (for explicit methods)
            is_implicit: Whether this is an implicit fingerprint method
            real_data_path: Path to real data (for some methods)
            cache_dir: Cache directory
            data_dir: Data directory
            
        Returns:
            Loaded attribution model
        """
        
        if is_implicit:
            return self._load_implicit_fingerprint_model(
                model_path, fingerprint_method, num_models,
                real_data_path, cache_dir, data_dir
            )
        else:
            return self._load_explicit_attribution_model(
                model_path, fingerprint_method, num_models, feature_dim
            )
    
    def _load_implicit_fingerprint_model(
        self,
        model_path: str,
        fingerprint_method: str,
        num_models: int,
        real_data_path: Optional[str] = None,
        cache_dir: str = "cache",
        data_dir: Optional[str] = None
    ) -> torch.nn.Module:
        """Load implicit fingerprint model."""
        
        print("Loading implicit fingerprint model...")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Implicit fingerprint model file not found at {model_path}")
        
        # Load model data
        model_data = self._load_model_data(model_path)
        
        # Create model with correct parameters
        implicit_model = self._create_implicit_model(fingerprint_method, num_models, model_data)
        
        # Load state dict
        try:
            implicit_model.load_state_dict(model_data['model_state_dict'])
            print("✅ Model state dict loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model state dict: {e}")
            print("This might indicate a mismatch between saved model architecture and current model")
            raise
        
        # Setup model
        implicit_model.to(self.device)
        implicit_model.eval()
        
        print(f"✅ Implicit fingerprint model loaded successfully (num_classes={num_models})")
        
        # Print model architecture info
        self._print_model_info(implicit_model)
        
        # Test model
        self._test_model_forward_pass(implicit_model)
        
        return implicit_model
    
    def _load_explicit_attribution_model(
        self,
        model_path: str,
        fingerprint_method: str,
        num_models: int,
        feature_dim: int
    ) -> torch.nn.Module:
        """Load explicit attribution model."""
        
        print("Loading true attribution model (h)...")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load model data
        model_data = self._load_model_data(model_path)
        
        print(f"   Debug: Saved model data keys: {list(model_data.keys())}")
        print(f"   Debug: Model data type: {type(model_data)}")
        
        # Create trainer
        try:
            from src.trainers.true_attribution import TrueAttributionModelTrainer
            true_trainer = TrueAttributionModelTrainer(
                fingerprint_method=fingerprint_method,
                dataset="ffhq",
                image_size=256,
                device=self.device
            )
        except ImportError as e:
            raise ImportError(f"Failed to import TrueAttributionModelTrainer: {e}")
        
        # Load model based on saved data structure
        if 'model_architecture' in model_data:
            # Use saved architecture
            model = model_data['model_architecture']
            model.load_state_dict(model_data['model_state_dict'])
            model.to(self.device)
            model.eval()
            print("✅ True attribution model loaded successfully (using saved architecture)")
        elif 'input_dim' in model_data and 'num_classes' in model_data:
            # Use saved dimensions
            saved_input_dim = model_data['input_dim']
            saved_output_dim = model_data['num_classes']
            print(f"✅ Found saved model dimensions: input_dim={saved_input_dim}, output_dim={saved_output_dim}")
            
            model = true_trainer.build_model(saved_input_dim, saved_output_dim)
            model.load_state_dict(model_data['model_state_dict'])
            model.to(self.device)
            model.eval()
            print("✅ True attribution model loaded successfully (using saved dimensions)")
        else:
            # Fallback: rebuild with calculated dimensions
            print("⚠️  Warning: No saved model architecture or dimensions found, rebuilding model...")
            print(f"  Using calculated feature dimension: {feature_dim}")
            
            saved_input_dim = model_data.get('input_dim')
            saved_output_dim = model_data.get('output_dim') or model_data.get('num_classes')
            
            if saved_input_dim is not None and saved_output_dim is not None:
                print(f"  Using saved dimensions: input_dim={saved_input_dim}, output_dim={saved_output_dim}")
                model = true_trainer.build_model(saved_input_dim, saved_output_dim)
            else:
                print(f"  No saved dimensions found, using calculated: feature_dim={feature_dim}, num_models={num_models}")
                model = true_trainer.build_model(feature_dim, num_models)
            
            model.load_state_dict(model_data['model_state_dict'])
            model.to(self.device)
            model.eval()
            print("✅ True attribution model loaded successfully (rebuilt)")
        
        # Print model info
        self._print_model_info(model)
        
        # Check class count
        if hasattr(model, 'num_classes') and model.num_classes != num_models:
            print(f"   ⚠️  WARNING: Model was trained with {model.num_classes} classes but attack expects {num_models} classes!")
        
        # Test model (prefer saved input_dim if available)
        test_input_dim = None
        try:
            # Reload model data to fetch saved dims safely
            model_data_for_test = self._load_model_data(model_path)
            test_input_dim = model_data_for_test.get('input_dim', feature_dim)
        except Exception:
            test_input_dim = feature_dim
        self._test_model_forward_pass(model, test_input_dim)
        
        return model
    
    def load_surrogate_attribution_model(
        self,
        model_path: str,
        fingerprint_method: str,
        num_models: int
    ) -> torch.nn.Module:
        """Load surrogate attribution model (h_s)."""
        
        print("Loading surrogate attribution model (h_s)...")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load model data
        model_data = self._load_model_data(model_path)
        
        print(f"   Debug: Saved surrogate model data keys: {list(model_data.keys())}")
        
        # Create trainer
        try:
            from src.trainers.surrogate_attribution import SurrogateAttributionModelTrainer
            surrogate_attr_trainer = SurrogateAttributionModelTrainer(
                fingerprint_method=fingerprint_method,
                dataset="ffhq",
                image_size=256,
                device=self.device
            )
        except ImportError as e:
            raise ImportError(f"Failed to import SurrogateAttributionModelTrainer: {e}")
        
        # Load model based on saved data structure
        if 'model_architecture' in model_data:
            # Use saved architecture
            model = model_data['model_architecture']
            model.load_state_dict(model_data['model_state_dict'])
            model.to(self.device)
            model.eval()
            print("✅ Surrogate attribution model loaded successfully (using saved architecture)")
        elif 'num_classes' in model_data:
            # Use saved output dimension and standard image input
            saved_output_dim = model_data['num_classes']
            saved_input_dim = 3 * 256 * 256  # Standard RGB image flattened
            print(f"✅ Found saved surrogate model output dimension: output_dim={saved_output_dim}")
            print(f"  Using standard image input dimension: input_dim={saved_input_dim}")
            
            model = surrogate_attr_trainer.build_model(saved_input_dim, saved_output_dim)
            model.load_state_dict(model_data['model_state_dict'])
            model.to(self.device)
            model.eval()
            print("✅ Surrogate attribution model loaded successfully (using saved dimensions)")
        else:
            # Fallback: rebuild with standard parameters
            print("⚠️  Warning: No saved surrogate model architecture or dimensions found, rebuilding model...")
            
            saved_input_dim = model_data.get('input_dim', 3 * 256 * 256)
            saved_output_dim = model_data.get('output_dim') or model_data.get('num_classes', num_models)
            
            print(f"  Using dimensions: input_dim={saved_input_dim}, output_dim={saved_output_dim}")
            model = surrogate_attr_trainer.build_model(saved_input_dim, saved_output_dim)
            
            model.load_state_dict(model_data['model_state_dict'])
            model.to(self.device)
            model.eval()
            print("✅ Surrogate attribution model loaded successfully (rebuilt)")
        
        # Print model info
        self._print_model_info(model)
        
        return model
    
    def load_surrogate_extractor(
        self,
        model_path: str,
        fingerprint_method: str,
        is_implicit: bool
    ) -> Optional[torch.nn.Module]:
        """Load surrogate extractor (φ_s)."""
        
        if is_implicit:
            print("Skipping surrogate extractor (φ_s) - not needed for implicit fingerprint methods")
            return None
        
        print("Loading surrogate extractor (φ_s)...")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load model data
        model_data = self._load_model_data(model_path)
        
        print(f"   Debug: Saved surrogate extractor data keys: {list(model_data.keys())}")
        
        # Create trainer
        try:
            from src.trainers.surrogate_extractor import SurrogateExtractorTrainer
            surrogate_extractor_trainer = SurrogateExtractorTrainer(
                fingerprint_method=fingerprint_method,
                dataset="ffhq",
                image_size=256,
                device=self.device
            )
        except ImportError as e:
            raise ImportError(f"Failed to import SurrogateExtractorTrainer: {e}")
        
        # Load model based on saved data structure
        if 'model_architecture' in model_data:
            # Use saved architecture
            model = model_data['model_architecture']
            model.load_state_dict(model_data['model_state_dict'])
            model.to(self.device)
            model.eval()
            print("✅ Surrogate extractor loaded successfully (using saved architecture)")
        elif 'input_channels' in model_data and 'output_features' in model_data:
            # Use saved dimensions
            saved_input_channels = model_data['input_channels']
            saved_output_features = model_data['output_features']
            print(f"✅ Found saved surrogate extractor dimensions: input_channels={saved_input_channels}, output_features={saved_output_features}")
            
            model = surrogate_extractor_trainer.build_model(saved_input_channels, saved_output_features)
            model.load_state_dict(model_data['model_state_dict'])
            model.to(self.device)
            model.eval()
            print("✅ Surrogate extractor loaded successfully (using saved dimensions)")
        else:
            # Fallback: rebuild with standard parameters
            print("⚠️  Warning: No saved surrogate extractor architecture or dimensions found, rebuilding model...")
            
            saved_input_channels = model_data.get('input_channels', 3)  # Default RGB
            saved_output_features = model_data.get('output_features') or model_data.get('output_dim')
            
            if saved_output_features is None:
                raise ValueError("Cannot determine output feature dimension for surrogate extractor")
            
            print(f"  Using dimensions: input_channels={saved_input_channels}, output_features={saved_output_features}")
            model = surrogate_extractor_trainer.build_model(saved_input_channels, saved_output_features)
            
            model.load_state_dict(model_data['model_state_dict'])
            model.to(self.device)
            model.eval()
            print("✅ Surrogate extractor loaded successfully (rebuilt)")
        
        # Print model info
        self._print_model_info(model)
        
        return model
    
    def _load_model_data(self, model_path: str) -> Dict[str, Any]:
        """Load model data from file with proper error handling."""
        
        try:
            # Try with weights_only=True first (PyTorch 2.6+ default)
            model_data = torch.load(model_path, map_location=self.device, weights_only=True)
        except Exception:
            # Fallback to weights_only=False for compatibility
            import torch.serialization
            with torch.serialization.safe_globals(['numpy._core.multiarray.scalar']):
                model_data = torch.load(model_path, map_location=self.device, weights_only=False)
        
        return model_data
    
    def _create_implicit_model(
        self,
        fingerprint_method: str,
        num_models: int,
        model_data: Dict[str, Any]
    ) -> torch.nn.Module:
        """Create implicit model with correct parameters."""
        
        model_config = model_data.get('model_config', {})
        
        if fingerprint_method == "qian20":
            from src.fingerprints.qian20 import Qian20
            
            img_size = model_config.get('img_size', 256)
            LFS_window_size = model_config.get('LFS_window_size', 10)
            LFS_M = model_config.get('LFS_M', 6)
            
            print(f"Creating qian20 model with parameters:")
            print(f"  - img_size: {img_size}")
            print(f"  - LFS_window_size: {LFS_window_size}")
            print(f"  - LFS_M: {LFS_M}")
            print(f"  - num_classes: {num_models}")
            
            return Qian20(
                img_size=img_size,
                LFS_window_size=LFS_window_size,
                LFS_M=LFS_M,
                num_classes=num_models
            )
        elif fingerprint_method == "wang20":
            from src.fingerprints.wang20 import Wang20
            
            img_size = model_config.get('img_size', 256)
            pretrained = model_config.get('pretrained', True)
            hidden_dims = model_config.get('hidden_dims', [512, 256])
            
            print(f"Creating wang20 model with parameters:")
            print(f"  - img_size: {img_size}")
            print(f"  - pretrained: {pretrained}")
            print(f"  - hidden_dims: {hidden_dims}")
            print(f"  - num_classes: {num_models}")
            
            return Wang20(
                img_size=img_size,
                num_classes=num_models,
                pretrained=pretrained,
                hidden_dims=hidden_dims
            )
        else:
            # For other implicit methods, use the default approach
            from src.data_loader.fingerprint_extractor_factory import FingerprintExtractorFactory
            fingerprint_extractor = FingerprintExtractorFactory.create(
                fingerprint_method,
                self.device,
                real_data_path=real_data_path,
                cache_dir=cache_dir,
                data_dir=data_dir
            )
            implicit_model = fingerprint_extractor.get_attribution_model()
            if implicit_model is None:
                raise ValueError(f"Method {fingerprint_method} does not provide an attribution model")
            return implicit_model
    
    def _print_model_info(self, model: torch.nn.Module) -> None:
        """Print model architecture information."""
        
        print(f"Model architecture info:")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Device: {next(model.parameters()).device}")
        print(f"  - Training mode: {model.training}")
        
        # Print dimension info if available
        if hasattr(model, 'input_dim'):
            print(f"  - Input dimension: {model.input_dim}")
        if hasattr(model, 'num_classes'):
            print(f"  - Output dimension: {model.num_classes}")
        if hasattr(model, 'output_dim'):
            print(f"  - Output dimension: {model.output_dim}")
    
    def _test_model_forward_pass(self, model: torch.nn.Module, input_dim: Optional[int] = None) -> None:
        """Test model with dummy input to verify it works."""
        
        try:
            with torch.no_grad():
                if hasattr(model, 'img_size'):
                    # For implicit models, test with image input
                    dummy_input = torch.randn(1, 3, model.img_size, model.img_size).to(self.device)
                    dummy_output = model(dummy_input)
                    print(f"✅ Model forward pass test successful")
                    print(f"  - Input shape: {dummy_input.shape}")
                    print(f"  - Output shape: {dummy_output.shape}")
                    print(f"  - Output classes: {dummy_output.shape[1]}")
                    
                    # Test with [0, 1] range input (like real images)
                    dummy_input_01 = torch.rand(1, 3, model.img_size, model.img_size).to(self.device)
                    dummy_output_01 = model(dummy_input_01)
                    print(f"  - [0,1] input test successful, output shape: {dummy_output_01.shape}")
                elif input_dim is not None:
                    # For explicit models, test with feature input
                    dummy_input = torch.randn(1, input_dim).to(self.device)
                    dummy_output = model(dummy_input)
                    print(f"✅ Model forward pass test successful")
                    print(f"  - Input shape: {dummy_input.shape}")
                    print(f"  - Output shape: {dummy_output.shape}")
                    print(f"  - Output classes: {dummy_output.shape[1]}")
                else:
                    print(f"⚠️  Could not test model forward pass - no input dimension available")
        except Exception as e:
            print(f"❌ Model forward pass test failed: {e}")
            raise
