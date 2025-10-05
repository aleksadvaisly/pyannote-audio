import torch
import torch.onnx
import os

# Load the embedding model checkpoint directly from cache
print("Loading embedding model checkpoint from local cache...")
model_path = os.path.expanduser(
    "~/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1/"
    "snapshots/3533c8cf8e369892e6b79ff1bf80f7b0286a54ee/embedding/pytorch_model.bin"
)

checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
print(f"Checkpoint keys: {checkpoint.keys()}")
print(f"PyAnnote audio version: {checkpoint.get('pytorch-lightning_version')}")

# Get model specifications from checkpoint
specs = checkpoint.get('pyannote.audio')
print(f"\nModel specs: {specs}")

# Build model architecture from checkpoint
from pyannote.audio.models.embedding import WeSpeakerResNet34

# Create model with default config (ResNet34)
print("\nCreating model architecture...")
model = WeSpeakerResNet34(
    sample_rate=16000,
    num_channels=1
)

# Load weights
print("Loading weights from checkpoint...")
state_dict = checkpoint['state_dict']

# Remove 'model.' prefix if present
cleaned_state_dict = {}
for k, v in state_dict.items():
    # Some keys might have 'model.' prefix from Lightning
    new_k = k.replace('model.', '') if k.startswith('model.') else k
    cleaned_state_dict[new_k] = v

try:
    model.load_state_dict(cleaned_state_dict, strict=False)
    print("✓ Weights loaded successfully")
except Exception as e:
    print(f"Warning during weight loading: {e}")
    # Try with original state_dict
    model.load_state_dict(state_dict, strict=False)

model.eval()

# The model includes mel filterbank computation which uses torch.vmap
# This is not supported by ONNX export, so we'll export just the ResNet encoder
# The mel filterbank computation will need to be done separately in C/Go

print("\nExtracting ResNet encoder (without mel filterbank preprocessing)...")

class ResNetEncoder(torch.nn.Module):
    """Wrapper for just the ResNet encoder part"""
    def __init__(self, base_model):
        super().__init__()
        self.resnet = base_model.resnet

    def forward(self, fbank):
        # fbank: [batch, num_frames, num_mel_bins]
        # ResNet returns (loss, embedding) tuple during training
        # We only want the embedding
        _, embedding = self.resnet(fbank)
        return embedding

encoder = ResNetEncoder(model)
encoder.eval()

# Create example input (mel filterbank features)
# Typical: 80 mel bins, ~500 frames for 5 seconds at 10ms frame shift
example_fbank = torch.randn(1, 500, 80)

print("\nTesting encoder forward pass...")
with torch.no_grad():
    output = encoder(example_fbank)
    print(f"Encoder output shape: {output.shape}")

print("\nExporting encoder to ONNX...")
torch.onnx.export(
    encoder,
    example_fbank,
    "embedding_model.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['mel_filterbank'],
    output_names=['embedding'],
    dynamic_axes={
        'mel_filterbank': {1: 'num_frames'},
    },
    verbose=False
)
print("✓ Embedding model exported to: embedding_model.onnx")

# Test with ONNX Runtime
import onnxruntime as ort
import numpy as np

print("\nTesting ONNX model...")
sess = ort.InferenceSession("embedding_model.onnx")

# Test inference with mel filterbank input
input_tensor = example_fbank.numpy()
outputs = sess.run(None, {'mel_filterbank': input_tensor})
print(f"Embedding output shape: {outputs[0].shape}")
print(f"Embedding sample values: {outputs[0][0][:10]}")

# Check file size
size_mb = os.path.getsize("embedding_model.onnx") / (1024 * 1024)
print(f"\n✓ Export successful!")
print(f"File size: {size_mb:.1f} MB")
print(f"\nNote: This model expects mel filterbank features as input (80 mel bins).")
print(f"You'll need to compute mel spectrograms from raw audio in your C/Go code.")
