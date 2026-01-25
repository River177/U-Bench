"""
Compute model metrics (Parameters, FLOPs, FPS) for multiple models
Usage: python scripts/compute_model_metrics.py
"""

import torch
import torch.nn as nn
import time
import sys
import os
import numpy as np
from typing import Dict, Tuple, List
import traceback

# GPU Configuration - Set which GPU to use
GPU_ID = 1  # Change this to select different GPU (0, 1, 2, etc.)
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Setup logging
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# Redirect output to log file
log_file = os.path.join(project_root, 'log.txt')
sys.stdout = Logger(log_file)
sys.stderr = sys.stdout

# Try to import FLOPs computation libraries
try:
    from thop import profile, clever_format
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("Warning: thop not installed. FLOPs computation will be skipped.")
    print("Install with: pip install thop")

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False


# Model configurations with their required input sizes
MODEL_CONFIGS = {
    'CFM_UNet': (1, 3, 224, 224),           # Fixed: added adaptive interpolation in Fusion module
    # 'CSCAUNet': (1, 3, 224, 224),           # Works with 224
    # 'CSWin_UNet': (1, 3, 256, 256),         # Requires 256 (hardcoded)
    # 'ERDUnet': (1, 3, 256, 256),            # Requires 256 (default in_h, in_w)
    # 'GH_UNet': (1, 3, 224, 224),            # Works with 224
    # 'MDSA_UNet': (1, 3, 256, 256),          # Requires size divisible by n_win=8, 256 is safer
    # 'Perspective_Unet': (1, 3, 256, 256),   # Requires specific size, 256 works
    # 'UNetV2': (1, 3, 224, 224),             # Works with 224
    # 'UTANetMamba': (1, 3, 224, 224),        # Works with 224
}

MODELS = list(MODEL_CONFIGS.keys())


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters.
    
    Returns:
        Tuple[int, int]: (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_flops_thop(model: nn.Module, input_size: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """
    Compute FLOPs using thop library.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
        
    Returns:
        Tuple[float, float]: (flops, params)
    """
    if not HAS_THOP:
        return None, None
    
    try:
        device = next(model.parameters()).device
        input_tensor = torch.randn(input_size).to(device)
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        return flops, params
    except Exception as e:
        print(f"    thop error: {e}")
        return None, None


def compute_flops_fvcore(model: nn.Module, input_size: Tuple[int, int, int, int]) -> float:
    """
    Compute FLOPs using fvcore library.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
        
    Returns:
        float: FLOPs count
    """
    if not HAS_FVCORE:
        return None
    
    try:
        device = next(model.parameters()).device
        input_tensor = torch.randn(input_size).to(device)
        flop_counter = FlopCountAnalysis(model, input_tensor)
        return flop_counter.total()
    except Exception as e:
        print(f"    fvcore error: {e}")
        return None


def measure_fps(model: nn.Module, 
                input_size: Tuple[int, int, int, int],
                num_warmup: int = 10,
                num_iterations: int = 100) -> Dict[str, float]:
    """
    Measure inference speed (FPS).
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
        num_warmup: Number of warmup iterations
        num_iterations: Number of timing iterations
        
    Returns:
        Dict with FPS metrics
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            input_tensor = torch.randn(input_size).to(device)
            _ = model(input_tensor)
    
    # Synchronize
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            input_tensor = torch.randn(input_size).to(device)
            
            start_time = time.time()
            _ = model(input_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            times.append(time.time() - start_time)
    
    times = np.array(times)
    batch_size = input_size[0]
    
    return {
        'mean_time': times.mean(),
        'std_time': times.std(),
        'fps': batch_size / times.mean(),
        'throughput': batch_size * num_iterations / times.sum()
    }


def load_model(model_name: str, device: str = 'cuda'):
    """
    Load a model by name.
    
    Args:
        model_name: Name of the model
        device: Device to load model on
        
    Returns:
        Model instance or None if failed
    """
    try:
        # Import the model
        if model_name == 'CFM_UNet':
            from models.Mamba.CFM_UNet.CFM_UNet import cfm_unet
            model = cfm_unet(input_channel=3, num_classes=1)
        elif model_name == 'CSCAUNet':
            from models.CNN.CSCAUNet.CSCAUNet import CSCAUNet
            model = CSCAUNet(input_channel=3, num_classes=1)
        elif model_name == 'CSWin_UNet':
            from models.Hybrid.CSWin_UNet.CSWin_UNet import cswin_unet
            model = cswin_unet(input_channel=3, num_classes=1)
        elif model_name == 'ERDUnet':
            from models.CNN.ERDUnet.ERDUnet import ERDUnet
            model = ERDUnet(input_channel=3, num_classes=1)
        elif model_name == 'GH_UNet':
            from models.CNN.GH_UNet.GH_UNet import gh_unet
            model = gh_unet(input_channel=3, num_classes=1)
        elif model_name == 'MDSA_UNet':
            from models.CNN.MDSA_UNet.MDSA_UNet import mdsa_unet
            model = mdsa_unet(input_channel=3, num_classes=1)
        elif model_name == 'Perspective_Unet':
            from models.CNN.Perspective_Unet.Perspective_Unet import perspective_unet
            model = perspective_unet(input_channel=3, num_classes=1)
        elif model_name == 'UNetV2':
            from models.CNN.UNet_v2.UNet_v2 import UNetV2
            model = UNetV2(input_channel=3, num_classes=1)
        elif model_name == 'UTANetMamba':
            from models.Exp.UTANetMamba.UTANetMamba import utanet_mamba
            model = utanet_mamba(input_channel=3, num_classes=1, pretrained=True)
        else:
            print(f"  Unknown model: {model_name}")
            return None
        
        model = model.to(device)
        model.eval()
        return model
        
    except Exception as e:
        print(f"  Failed to load {model_name}: {e}")
        print("  Full traceback:")
        traceback.print_exc()
        return None


def compute_model_metrics(model_name: str, 
                         input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
                         device: str = 'cuda') -> Dict:
    """
    Compute all metrics for a single model.
    
    Args:
        model_name: Name of the model
        input_size: Input tensor size
        device: Device to use
        
    Returns:
        Dict with all metrics
    """
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")
    
    # Load model
    print("Loading model...")
    model = load_model(model_name, device)
    if model is None:
        return None
    
    print("✓ Model loaded successfully")
    
    metrics = {
        'model_name': model_name,
        'input_size': input_size,
        'device': device
    }
    
    # 1. Parameters
    print("\n1. Computing Parameters...")
    try:
        total_params, trainable_params = count_parameters(model)
        metrics['total_params'] = total_params
        metrics['trainable_params'] = trainable_params
        print(f"   Total:      {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"   Trainable:  {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    except Exception as e:
        print(f"   Error: {e}")
        metrics['total_params'] = None
    
    # 2. FLOPs
    print("\n2. Computing FLOPs...")
    flops_computed = False
    try:
        flops, _ = compute_flops_thop(model, input_size)
        if flops is not None:
            metrics['flops'] = flops
            print(f"   FLOPs: {flops:,} ({flops/1e9:.2f}G)")
            flops_computed = True
        else:
            metrics['flops'] = None
            print("   FLOPs: N/A (thop not available)")
    except Exception as e:
        print(f"   Error computing FLOPs: {e}")
        print("   Full traceback:")
        traceback.print_exc()
        metrics['flops'] = None
    
    # Clean up and reload model if FLOPs was computed
    # This avoids conflicts with thop's modifications to the model
    if flops_computed:
        print("   Reloading model for FPS measurement...")
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
        model = load_model(model_name, device)
        if model is None:
            print("   Failed to reload model, skipping FPS")
            return metrics
    
    # 3. FPS
    print("\n3. Measuring FPS...")
    try:
        fps_metrics = measure_fps(model, input_size, num_warmup=10, num_iterations=100)
        metrics.update(fps_metrics)
        print(f"   Mean Time: {fps_metrics['mean_time']*1000:.2f} ms (±{fps_metrics['std_time']*1000:.2f} ms)")
        print(f"   FPS:       {fps_metrics['fps']:.2f}")
    except Exception as e:
        print(f"   Error measuring FPS: {e}")
        print("   Full traceback:")
        traceback.print_exc()
        metrics['fps'] = None
        metrics['mean_time'] = None
    
    # Clean up
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return metrics


def format_number(num):
    """Format number with appropriate unit."""
    if num is None:
        return "N/A"
    if num >= 1e9:
        return f"{num/1e9:.2f}G"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"


def print_summary_table(all_metrics: List[Dict]):
    """Print a summary table of all results."""
    print("\n" + "="*110)
    print("SUMMARY TABLE")
    print("="*110)
    
    # Header
    print(f"{'Model':<20} {'Input':<10} {'Params':<15} {'FLOPs':<15} {'FPS':<12} {'Time (ms)':<12}")
    print("-"*110)
    
    # Data rows
    for metrics in all_metrics:
        if metrics is None:
            continue
        
        model_name = metrics['model_name']
        input_size = metrics.get('input_size', '')
        if isinstance(input_size, tuple):
            input_str = f"{input_size[2]}x{input_size[3]}"
        else:
            input_str = "N/A"
        params = format_number(metrics.get('total_params'))
        flops = format_number(metrics.get('flops'))
        fps = f"{metrics.get('fps', 0):.2f}" if metrics.get('fps') else "N/A"
        time_ms = f"{metrics.get('mean_time', 0)*1000:.2f}" if metrics.get('mean_time') else "N/A"
        
        print(f"{model_name:<20} {input_str:<10} {params:<15} {flops:<15} {fps:<12} {time_ms:<12}")
    
    print("="*110)


def save_results_to_csv(all_metrics: List[Dict], output_file: str = "model_metrics.csv"):
    """Save results to CSV file."""
    import csv
    
    output_path = os.path.join(project_root, output_file)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Model', 'Input_Size', 'Total_Params', 'Trainable_Params', 'FLOPs', 
            'FPS', 'Mean_Time_ms', 'Std_Time_ms', 'Throughput'
        ])
        
        # Data
        for metrics in all_metrics:
            if metrics is None:
                continue
            
            # Handle None values safely
            mean_time = metrics.get('mean_time')
            std_time = metrics.get('std_time')
            input_size = metrics.get('input_size', '')
            
            # Format input size as string
            if isinstance(input_size, tuple):
                input_size_str = f"{input_size[2]}x{input_size[3]}"
            else:
                input_size_str = str(input_size)
            
            writer.writerow([
                metrics['model_name'],
                input_size_str,
                metrics.get('total_params', ''),
                metrics.get('trainable_params', ''),
                metrics.get('flops', ''),
                metrics.get('fps', ''),
                mean_time * 1000 if mean_time is not None else '',
                std_time * 1000 if std_time is not None else '',
                metrics.get('throughput', '')
            ])
    
    print(f"\n✓ Results saved to: {output_path}")


def main():
    print("="*100)
    print("MODEL METRICS COMPUTATION")
    print("="*100)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nConfiguration:")
    print(f"  Device:     {device}")
    print(f"  Models:     {len(MODELS)}")
    
    if torch.cuda.is_available():
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
    
    print(f"\nModel-specific input sizes:")
    for model_name, input_size in MODEL_CONFIGS.items():
        print(f"  {model_name:<20} {input_size}")
    
    # Compute metrics for all models
    all_metrics = []
    
    for i, model_name in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] Processing {model_name}...")
        
        # Get model-specific input size
        input_size = MODEL_CONFIGS[model_name]
        print(f"  Input size: {input_size}")
        
        try:
            metrics = compute_model_metrics(model_name, input_size, device)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Failed to process {model_name}: {e}")
            print("Full traceback:")
            traceback.print_exc()
            all_metrics.append(None)
    
    # Print summary
    print_summary_table(all_metrics)
    
    # Save to CSV
    try:
        save_results_to_csv(all_metrics)
    except Exception as e:
        print(f"Failed to save CSV: {e}")
    
    print("\n" + "="*100)
    print("✓ All computations complete!")
    print("="*100)
    print(f"\nLog saved to: {log_file}")
    
    # Close log file
    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
    finally:
        # Ensure log file is closed
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()
