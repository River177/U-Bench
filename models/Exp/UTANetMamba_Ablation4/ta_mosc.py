
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, List, Set, Optional

# Try to import Mamba, handle if not available
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

class MambaExpert(nn.Module):
    """
    Mamba Expert: Based on UltraLight VM-UNet PVMLayer
    LayerNorm -> Mamba SSM -> Linear Projection
    Provides global receptive field and long-range dependency modeling.
    """
    def __init__(self, emb_size: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError("mamba_ssm is not installed. Cannot use MambaExpert.")
            
        self.input_dim = emb_size
        self.norm = nn.LayerNorm(emb_size)
        
        # Mamba block
        self.mamba = Mamba(
            d_model=emb_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # Output projection
        self.proj = nn.Linear(emb_size, emb_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Flatten for Mamba: (B, C, H, W) -> (B, H*W, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        x_norm = self.norm(x_flat)
        
        # Mamba forward
        x_mamba = self.mamba(x_norm)
        
        # Projection and reshape back
        x_out = self.proj(x_mamba)
        out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        return out

class MoE(nn.Module):
    """
    Sparse Mixture of Experts Module with Mamba Experts
    Implements TRUE sparse Top-K computation - only selected experts are computed.
    """
    def __init__(self, 
                 num_experts: int = 4, 
                 top: int = 2, 
                 emb_size: int = 128, 
                 expert_type: str = 'mamba',
                 H: int = 224, 
                 W: int = 224):
        super().__init__()
        self.top = top
        self.num_experts = num_experts
        
        # Initialize Mamba Experts
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            if HAS_MAMBA:
                self.experts.append(MambaExpert(emb_size))
            else:
                raise ImportError("mamba_ssm is required for MambaExpert")
        
        # Gating Networks (4 gates for different scales)
        self.gate1 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self.gate2 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self.gate3 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self.gate4 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        
        self._initialize_weights()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
    def _initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.gate1)
        nn.init.xavier_uniform_(self.gate2)
        nn.init.xavier_uniform_(self.gate3)
        nn.init.xavier_uniform_(self.gate4)
        
    def cv_squared(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-10
        mean = x.mean()
        var = x.var()
        return var / (mean ** 2 + eps)
    
    def _process_gate(self, x: torch.Tensor, gate_weights: nn.Parameter) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input through a single gating mechanism with TRUE sparse computation.
        Only computes forward pass for selected Top-K experts.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            gate_weights: Gate weights for this gating mechanism
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and load balancing loss
        """
        B, C, H, W = x.shape
        
        # 1. Compute gating probabilities
        x_gap = self.gap(x).view(B, C)  # (B, C)
        gate_logits = torch.matmul(x_gap, gate_weights)  # (B, num_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)  # (B, num_experts)
        
        # 2. Select Top-K experts
        topk = min(self.top, self.num_experts)
        topk_probs, topk_indices = torch.topk(gate_probs, topk, dim=-1)  # (B, topk)
        
        # Normalize topk probabilities
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        # 3. Collect all unique experts that need to be computed across the batch
        selected_experts = torch.unique(topk_indices).tolist()
        
        # 4. SPARSE COMPUTATION: Only compute selected experts
        expert_outputs = {}
        for expert_idx in selected_experts:
            expert_outputs[expert_idx] = self.experts[expert_idx](x)  # (B, C, H, W)
        
        # 5. Combine expert outputs based on gating weights
        output = torch.zeros_like(x)
        for i in range(B):
            for j in range(topk):
                expert_idx = topk_indices[i, j].item()
                weight = topk_probs[i, j]
                output[i] += weight * expert_outputs[expert_idx][i]
        
        # 6. Load balancing loss
        expert_usage = gate_probs.mean(dim=0)
        loss = self.cv_squared(expert_usage)
        
        return output, loss
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through all gating mechanisms with sparse computation.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Tuple of 4 output tensors and combined load balancing loss
        """
        o1, loss1 = self._process_gate(x, self.gate1)
        o2, loss2 = self._process_gate(x, self.gate2)
        o3, loss3 = self._process_gate(x, self.gate3)
        o4, loss4 = self._process_gate(x, self.gate4)
        
        # Combine losses
        loss = loss1 + loss2 + loss3 + loss4
        
        return o1, o2, o3, o4, loss

def count_parameters(model: nn.Module) -> str:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        str: Formatted string with parameter count
    """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if params >= 1e6:
        return f"{params / 1e6:.2f}M parameters"
    elif params >= 1e3:
        return f"{params / 1e3:.2f}K parameters"
    else:
        return f"{params} parameters"

if __name__ == '__main__':
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        
        # Test Mamba MoE
        print("\nTesting Mamba MoE with Sparse Top-K Computation...")
        model = MoE(num_experts=4, top=2, emb_size=64, expert_type='mamba').to(device)
        x = torch.randn(2, 64, 64, 64).to(device)
        
        model.train()
        o1, o2, o3, o4, loss = model(x)
        
        print(f"Input Shape: {x.shape}")
        print(f"Output Shapes: {o1.shape}, {o2.shape}, {o3.shape}, {o4.shape}")
        print(f"Load Balancing Loss: {loss.item():.4f}")
        print(f"Model Parameters: {count_parameters(model)}")
        
        # Verify outputs
        assert o1.shape == x.shape, f"Output shape mismatch: {o1.shape} vs {x.shape}"
        assert o2.shape == x.shape, f"Output shape mismatch: {o2.shape} vs {x.shape}"
        assert o3.shape == x.shape, f"Output shape mismatch: {o3.shape} vs {x.shape}"
        assert o4.shape == x.shape, f"Output shape mismatch: {o4.shape} vs {x.shape}"
        
        print("\n=== Sparse Mamba MoE Test Passed ===")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
