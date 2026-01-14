
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

class StandardExpert(nn.Module):
    """
    Standard Expert: Based on bottleneck CNN (Original UTANet style)
    1x1 Conv -> 1x1 Conv (Expand) -> 1x1 Conv (Squeeze)
    """
    def __init__(self, emb_size: int, hidden_rate: int = 2):
        super().__init__()
        hidden_emb = hidden_rate * emb_size
        self.seq = nn.Sequential(
            nn.Conv2d(emb_size, hidden_emb, 1, 1, 0, bias=True),
            nn.Conv2d(hidden_emb, hidden_emb, 1, 1, 0, bias=True),
            nn.BatchNorm2d(hidden_emb),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_emb, emb_size, 1, 1, 0, bias=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)

class LargeKernelExpert(nn.Module):
    """
    Large Kernel Expert: Based on CMUNeXt Block
    Depth-wise Large Kernel Conv (7x7) -> Inverted Bottleneck (1x1 expansion)
    Provides strong local connectivity.
    """
    def __init__(self, emb_size: int, kernel_size: int = 7, expand_ratio: int = 4):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(emb_size, emb_size, kernel_size=kernel_size, 
                      padding=kernel_size//2, groups=emb_size, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(emb_size)
        )
        
        hidden_dim = emb_size * expand_ratio
        self.pw = nn.Sequential(
            nn.Conv2d(emb_size, hidden_dim, 1, 1, 0, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, emb_size, 1, 1, 0, bias=False),
            nn.BatchNorm2d(emb_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection for DW part
        x = x + self.dw(x)
        # Feed Forward part
        x = self.pw(x)
        return x

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
            d_model=emb_size, # Input dimension
            d_state=d_state,  # SSM state expansion
            d_conv=d_conv,    # Local convolution width
            expand=expand,    # Block expansion factor
        )
        
        # Output projection (optional, as Mamba outputs d_model)
        # But to match Expert interface we ensure output shape
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
    Improved Mixture of Experts Module
    Supports different expert types: 'cnn' (standard), 'large_kernel', 'mamba', 'hybrid'
    """
    def __init__(self, 
                 num_experts: int = 4, 
                 top: int = 2, 
                 emb_size: int = 128, 
                 expert_type: str = 'hybrid',
                 H: int = 224, 
                 W: int = 224):
        super().__init__()
        self.top = top
        self.expert_type = expert_type.lower()
        self.experts = nn.ModuleList()
        
        # Initialize Experts
        for i in range(num_experts):
            if self.expert_type == 'cnn':
                self.experts.append(StandardExpert(emb_size))
            elif self.expert_type == 'large_kernel':
                self.experts.append(LargeKernelExpert(emb_size))
            elif self.expert_type == 'mamba':
                if HAS_MAMBA:
                    self.experts.append(MambaExpert(emb_size))
                else:
                    print("Warning: Mamba not found, falling back to LargeKernelExpert")
                    self.experts.append(LargeKernelExpert(emb_size))
            elif self.expert_type == 'hybrid':
                # Hybrid: Mix of experts for balanced performance
                # Strategy for 4 experts:
                # 0: Mamba (Global)
                # 1: LargeKernel (Local Connectivity)
                # 2: Standard (Channel Mixing)
                # 3: LargeKernel (Local Connectivity) - emphasis on spatial
                
                # Adjust based on availability
                if i == 0: 
                    if HAS_MAMBA:
                        self.experts.append(MambaExpert(emb_size))
                    else:
                         self.experts.append(LargeKernelExpert(emb_size))
                elif i == 1:
                    self.experts.append(LargeKernelExpert(emb_size))
                elif i == 2:
                    self.experts.append(StandardExpert(emb_size))
                else: # i == 3 or others
                    self.experts.append(LargeKernelExpert(emb_size))
            else:
                raise ValueError(f"Unknown expert type: {expert_type}")

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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        
        # 1. Global Average Pooling for Gating
        x_gap = self.gap(x).view(B, C)  # (B, C)
        
        # 2. Calculate Gate Logits
        gate1_logits = torch.matmul(x_gap, self.gate1)
        gate2_logits = torch.matmul(x_gap, self.gate2)
        gate3_logits = torch.matmul(x_gap, self.gate3)
        gate4_logits = torch.matmul(x_gap, self.gate4)
        
        # 3. Softmax Probabilities
        gate1_probs = F.softmax(gate1_logits, dim=-1)
        gate2_probs = F.softmax(gate2_logits, dim=-1)
        gate3_probs = F.softmax(gate3_logits, dim=-1)
        gate4_probs = F.softmax(gate4_logits, dim=-1)
        
        # 4. Top-k Selection
        topk = min(self.top, len(self.experts))
        
        def get_topk(probs):
            val, idx = torch.topk(probs, topk, dim=-1)
            # Normalize topk probabilities
            norm_val = val / (val.sum(dim=-1, keepdim=True) + 1e-10)
            return norm_val, idx

        gate1_topk_probs, gate1_topk_indices = get_topk(gate1_probs)
        gate2_topk_probs, gate2_topk_indices = get_topk(gate2_probs)
        gate3_topk_probs, gate3_topk_indices = get_topk(gate3_probs)
        gate4_topk_probs, gate4_topk_indices = get_topk(gate4_probs)
        
        # 步骤4：通过专家网络处理输入，并加权组合
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))  # 每个输出: (B, C, H, W)
        
        # 将专家输出堆叠
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, num_experts, C, H, W)
        
        # 为每个门控计算加权输出
        o1 = torch.zeros_like(x)
        o2 = torch.zeros_like(x)
        o3 = torch.zeros_like(x)
        o4 = torch.zeros_like(x)
        
        for i in range(B):
            # Gate 1
            for j, idx in enumerate(gate1_topk_indices[i]):
                o1[i] += gate1_topk_probs[i, j] * expert_outputs[i, idx]
            
            # Gate 2
            for j, idx in enumerate(gate2_topk_indices[i]):
                o2[i] += gate2_topk_probs[i, j] * expert_outputs[i, idx]
            
            # Gate 3
            for j, idx in enumerate(gate3_topk_indices[i]):
                o3[i] += gate3_topk_probs[i, j] * expert_outputs[i, idx]
            
            # Gate 4
            for j, idx in enumerate(gate4_topk_indices[i]):
                o4[i] += gate4_topk_probs[i, j] * expert_outputs[i, idx]
        
        # 7. Load Balancing Loss
        expert_usage = (gate1_probs + gate2_probs + gate3_probs + gate4_probs) / 4.0
        expert_usage_mean = expert_usage.mean(dim=0)
        loss = self.cv_squared(expert_usage_mean)
        
        return o1, o2, o3, o4, loss

def count_parameters(model: nn.Module) -> str:
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
        
        # Test Standard
        print("\nTesting Standard CNN MoE...")
        model = MoE(num_experts=4, top=2, emb_size=64, expert_type='cnn').to(device)
        x = torch.randn(2, 64, 64, 64).to(device)
        o1, o2, o3, o4, loss = model(x)
        print(f"Standard Output: {o1.shape}, Loss: {loss.item()}")
        
        # Test Large Kernel
        print("\nTesting Large Kernel MoE...")
        model = MoE(num_experts=4, top=2, emb_size=64, expert_type='large_kernel').to(device)
        o1, o2, o3, o4, loss = model(x)
        print(f"LargeKernel Output: {o1.shape}, Loss: {loss.item()}")
        
        # Test Hybrid
        print("\nTesting Hybrid MoE...")
        model = MoE(num_experts=4, top=2, emb_size=64, expert_type='hybrid').to(device)
        o1, o2, o3, o4, loss = model(x)
        print(f"Hybrid Output: {o1.shape}, Loss: {loss.item()}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
