"""
Debug CFM_UNet to find the shape mismatch issue
"""
import torch
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.Mamba.CFM_UNet.CFM_UNet import CFMUNet

def test_cfm_unet_shapes(input_size=(1, 3, 224, 224)):
    """Test CFM_UNet with different input sizes to find which works"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = CFMUNet(input_channel=3, num_classes=1)
    model = model.to(device)
    model.eval()
    
    print(f"Testing CFM_UNet with input size: {input_size}")
    print("="*80)
    
    x = torch.randn(input_size).to(device)
    
    # Trace through the forward pass and print shapes
    with torch.no_grad():
        try:
            # VSSM path
            x1_0 = model.vssm.patch_embed(x)
            x1_0 = model.vssm.pos_drop(x1_0)
            print(f"x1_0 (VSSM patch_embed): {x1_0.shape}")
            
            # ResNet path
            x1_1 = model.resnet.conv1(x)
            x1_1 = model.resnet.bn1(x1_1)
            x1_1 = model.resnet.relu(x1_1)
            x1_1 = model.resnet.maxpool(x1_1)
            print(f"x1_1 (ResNet maxpool):   {x1_1.shape}")
            
            # Fusion1
            x1 = model.Fusion1(x1_0.permute(0, 3, 1, 2), x1_1)
            print(f"x1 (Fusion1):            {x1.shape}")
            
            # Layer 1
            x2_0 = x1.permute(0, 2, 3, 1) + x1_0
            x2_0 = model.vssm.layers['X1'](torch.sigmoid(x2_0))
            print(f"x2_0 (VSSM X1):          {x2_0.shape}")
            
            x2_1 = x1 + x1_1
            x2_1 = model.resnet.layer1(torch.sigmoid(x2_1))
            print(f"x2_1 (ResNet layer1):    {x2_1.shape}")
            
            # Fusion2
            x2 = model.Fusion2(x2_0.permute(0, 3, 1, 2), x2_1)
            print(f"x2 (Fusion2):            {x2.shape}")
            
            # Layer 2
            x3_0 = x2.permute(0, 2, 3, 1) + x2_0
            x3_0 = model.vssm.layers['X2'](torch.sigmoid(x3_0))
            print(f"x3_0 (VSSM X2):          {x3_0.shape}")
            
            x3_1 = x2 + x2_1
            x3_1 = model.resnet.layer2(torch.sigmoid(x3_1))
            print(f"x3_1 (ResNet layer2):    {x3_1.shape}")
            
            # Fusion3
            x3 = model.Fusion3(x3_0.permute(0, 3, 1, 2), x3_1)
            print(f"x3 (Fusion3):            {x3.shape}")
            
            # Layer 3
            x4_0 = x3.permute(0, 2, 3, 1) + x3_0
            x4_0 = model.vssm.layers['X3'](torch.sigmoid(x4_0))
            print(f"x4_0 (VSSM X3):          {x4_0.shape}")
            
            x4_1 = x3 + x3_1
            x4_1 = model.resnet.layer3(torch.sigmoid(x4_1))
            print(f"x4_1 (ResNet layer3):    {x4_1.shape}")
            
            # Fusion4
            x4 = model.Fusion4(x4_0.permute(0, 3, 1, 2), x4_1)
            print(f"x4 (Fusion4):            {x4.shape}")
            
            # Layer 4
            x5_0 = x4.permute(0, 2, 3, 1) + x4_0
            x5_0 = model.vssm.layers['X4'](torch.sigmoid(x5_0))
            print(f"x5_0 (VSSM X4):          {x5_0.shape}")
            print(f"x5_0 permuted:           {x5_0.permute(0, 3, 1, 2).shape}")
            
            x5_1 = x4 + x4_1
            x5_1 = model.resnet.layer4(torch.sigmoid(x5_1))
            print(f"x5_1 (ResNet layer4):    {x5_1.shape}")
            
            print("\n" + "="*80)
            print("PROBLEM: Trying to fuse x5_0.permute(0,3,1,2) and x5_1")
            print(f"  x5_0.permute(0,3,1,2) shape: {x5_0.permute(0, 3, 1, 2).shape}")
            print(f"  x5_1 shape:                   {x5_1.shape}")
            print("="*80)
            
            # Try Fusion5
            x5 = model.Fusion5(x5_0.permute(0, 3, 1, 2), x5_1)
            print(f"✓ Fusion5 succeeded!")
            print(f"x5 (Fusion5):            {x5.shape}")
            
            # Full forward
            output = model(x)
            print(f"\n✓ Full forward pass succeeded!")
            print(f"Output shape: {output.shape}")
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


if __name__ == "__main__":
    print("Testing different input sizes for CFM_UNet\n")
    
    test_sizes = [
        (1, 3, 224, 224),
        (1, 3, 256, 256),
        (1, 3, 352, 352),
        (1, 3, 384, 384),
        (1, 3, 512, 512),
    ]
    
    for size in test_sizes:
        print(f"\n{'='*80}")
        success = test_cfm_unet_shapes(size)
        if success:
            print(f"✓ {size} works!")
            break
        print(f"✗ {size} failed")
