import os
import sys
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
from pytorch_nndct.apis import torch_quantizer

# Add lib directory to Python path (current directory has lib subfolder)
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')
sys.path.insert(0, lib_path)

# Import your SINet-V2 models
from Res2Net_v1b import res2net50_v1b_26w_4s
from Network_Res2Net_GRA_NCD import Network

# Configuration - Using container paths
CALIB_DATA_PATH = './calib_data'  # Relative to /workspace/SINet-V2/SINet-V2
WEIGHTS_PATH = './Net_epoch_best.pth'  # Relative to /workspace/SINet-V2/SINet-V2
QUANTIZE_OUTPUT_DIR = './quantize_result'
DEVICE = 'cpu'  # Vitis AI 3.5 CPU container
BATCH_SIZE = 1
NUM_CALIB_IMAGES = 50  # Use 100-200 for robust quantization

class SINetV2Wrapper(nn.Module):
    """Wrapper to ensure proper mask output for quantization"""
    def __init__(self, base_model):
        super(SINetV2Wrapper, self).__init__()
        self.model = base_model
    
    def forward(self, x):
        # SINet-V2 returns (Sg_pred, S5_pred, S4_pred, S3_pred)
        # Sg_pred is [1, 1, 352, 352]: main segmentation mask (not feature map)
        output = self.model(x)
        if isinstance(output, tuple):
            Sg_pred = output[0]
            # Force correct shape, fail early if wrong
            assert Sg_pred.shape[2:] == (352,352), f"ERROR: Wrong output shape! Got {Sg_pred.shape}"
            return Sg_pred
        # If model returns just the mask
        assert output.shape[2:] == (352,352), f"ERROR: Wrong output shape! Got {output.shape}"
        return output

class CalibDataLoader:
    """Custom dataloader for calibration dataset"""
    def __init__(self, calib_dir, batch_size=1, input_size=352):
        self.calib_dir = calib_dir
        self.batch_size = batch_size
        self.input_size = input_size
        
        # Same preprocessing as your VSCode training pipeline
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Get image files
        self.image_files = [
            os.path.join(calib_dir, f) 
            for f in os.listdir(calib_dir) 
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ][:NUM_CALIB_IMAGES]
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {calib_dir}")
        
        print(f"✓ Loaded {len(self.image_files)} calibration images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __iter__(self):
        for img_path in self.image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0)  # Add batch dimension
                yield img_tensor, 0  # Return dummy label
            except Exception as e:
                print(f"⚠ Warning: Failed to load {img_path}: {e}")
                continue

def load_model(weights_path, device='cpu'):
    """Load pre-trained SINet-V2 model"""
    print(f"📦 Loading SINet-V2 model from {weights_path}")
    
    # Initialize model (same as your VSCode setup)
    model = Network(imagenet_pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    
    # Wrap model for proper output handling
    wrapped_model = SINetV2Wrapper(model)
    wrapped_model.eval()
    
    print("✓ Model loaded successfully")
    return wrapped_model

def quantize_model(model, calib_loader, output_dir, quant_mode='calib'):
    """
    Quantize model using Vitis AI PyTorch Quantizer
    """
    os.makedirs(output_dir, exist_ok=True)
    
    input_shape = (1, 3, 352, 352)
    dummy_input = torch.randn(input_shape)
    
    print(f"\n{'='*60}")
    print(f"🔧 Starting quantization in '{quant_mode}' mode")
    print(f"{'='*60}\n")
    
    quantizer = torch_quantizer(
        quant_mode=quant_mode,
        module=model,
        input_args=dummy_input,
        output_dir=output_dir,
        device=torch.device('cpu'),
        bitwidth=8,
    )
    
    quantized_model = quantizer.quant_model
    
    if quant_mode == 'calib':
        print("📊 Running calibration pass...")
        print("   This analyzes activation ranges for optimal quantization")
        print("   Unsupported ops will be automatically identified for CPU partitioning\n")
        
        with torch.no_grad():
            for idx, (images, _) in enumerate(calib_loader):
                if idx >= NUM_CALIB_IMAGES:
                    break
                
                images = images.to(DEVICE)
                _ = quantized_model(images)
                
                if (idx + 1) % 10 == 0:
                    print(f"   Processed {idx + 1}/{NUM_CALIB_IMAGES} calibration images")
        
        print("\n✓ Calibration completed")
        print(f"   Configuration saved to: {output_dir}")
        
    elif quant_mode == 'test':
        print("🧪 Running quantization evaluation...")
        print("   Generating deployable quantized model\n")
        
        with torch.no_grad():
            test_images = 0
            for images, _ in calib_loader:
                images = images.to(DEVICE)
                _ = quantized_model(images)
                test_images += 1
                
                if test_images >= 20:
                    break
        
        print("\n✓ Quantization evaluation completed")
        print(f"   Quantized model saved to: {output_dir}")
    
    # ========== PASTE DEBUGGING CODE HERE ==========
    print("\n" + "="*60)
    print("🔍 DEBUG: Checking quantized model output shape...")
    print("="*60)
    
    dummy_input_debug = torch.randn(1, 3, 352, 352)
    quantized_model.eval()
    with torch.no_grad():
        out = quantized_model(dummy_input_debug)
        print(f"✓ Quantized model returns output shape: {out.shape}")
        if out.shape == torch.Size([1, 1, 352, 352]):
            print("✅ SUCCESS: Output shape is correct [1, 1, 352, 352]!")
        else:
            print(f"❌ ERROR: Output shape is WRONG! Expected [1, 1, 352, 352], got {out.shape}")
    print("="*60 + "\n")
    # ========== END DEBUGGING CODE ==========
    
    # Export quantized model
    quantizer.export_quant_config()
    
    if quant_mode == 'test':
        quantizer.export_xmodel(output_dir=output_dir, deploy_check=True)
        print(f"\n✅ INT8 .xmodel generated: {output_dir}/Network_int.xmodel")
        print("   Ready for compilation with vai_c_xir")
    
    return quantized_model


def main():
    parser = argparse.ArgumentParser(description='Quantize SINet-V2 for DPUCZDX8G')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['calib', 'test'],
                       help='Quantization mode: calib or test')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎯 SINet-V2 Quantization for Vitis AI 3.5")
    print("="*60 + "\n")
    
    # Verify paths exist
    if not os.path.exists(CALIB_DATA_PATH):
        raise FileNotFoundError(f"Calibration data not found: {CALIB_DATA_PATH}")
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Weights file not found: {WEIGHTS_PATH}")
    
    print(f"✓ Calibration data: {os.path.abspath(CALIB_DATA_PATH)}")
    print(f"✓ Weights file: {os.path.abspath(WEIGHTS_PATH)}")
    print(f"✓ Lib path: {lib_path}\n")
    
    # Load model
    model = load_model(WEIGHTS_PATH, DEVICE)
    
    # Create calibration dataloader
    calib_loader = CalibDataLoader(
        calib_dir=CALIB_DATA_PATH,
        batch_size=BATCH_SIZE,
        input_size=352
    )
    
    # Quantize model
    quantized_model = quantize_model(
        model=model,
        calib_loader=calib_loader,
        output_dir=QUANTIZE_OUTPUT_DIR,
        quant_mode=args.mode
    )
    
    print("\n" + "="*60)
    print("✅ Quantization pipeline completed successfully!")
    print("="*60 + "\n")
    
    if args.mode == 'calib':
        print("📝 Next step:")
        print("   Run: python quantize_sinet_v2_fixed.py --mode test")
    else:
        print("📝 Next steps:")
        print("   1. Compile .xmodel: ./compile_kv260.sh")
        print("   2. Deploy to KV260")
        print("   3. Run inference with VART runtime\n")

if __name__ == '__main__':
    main()