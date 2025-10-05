import torch
import torch.onnx
import os

os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN', '')

# Load the embedding model from the cached pipeline
print("Loading embedding model from cache...")
model_path = os.path.expanduser(
    "~/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1/"
    "snapshots/3533c8cf8e369892e6b79ff1bf80f7b0286a54ee/embedding/pytorch_model.bin"
)

checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
print(f"Checkpoint keys: {checkpoint.keys()}")

# Extract state dict
state_dict = checkpoint['state_dict']

# Try to load the model architecture
from pyannote.audio import Model

print("Loading embedding model architecture...")
emb_model = Model.from_pretrained(
    "pyannote/embedding",
    use_auth_token=os.environ.get('HF_TOKEN')
)
emb_model.eval()

class EmbeddingWrapper(torch.nn.Module):
    """Wrapper to handle embedding model input"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform):
        # Model expects dict with 'waveform' key
        return self.model({'waveform': waveform})

wrapped_model = EmbeddingWrapper(emb_model)
wrapped_model.eval()

# Create example input (3 seconds of audio at 16kHz)
# Embedding model typically needs longer chunks
example_input = torch.randn(1, 1, 48000)  # 3 seconds

print("Tracing embedding model...")
with torch.no_grad():
    traced_model = torch.jit.trace(wrapped_model, example_input)

print("Exporting embedding to ONNX...")
torch.onnx.export(
    traced_model,
    example_input,
    "embedding_model.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['waveform'],
    output_names=['embedding'],
    dynamic_axes={
        'waveform': {2: 'audio_length'},
    }
)
print("✓ Embedding model exported to: embedding_model.onnx")

# Test with ONNX Runtime
import onnxruntime as ort
import numpy as np

print("\nTesting ONNX embedding model...")
sess = ort.InferenceSession("embedding_model.onnx")
print(f"Available execution providers: {ort.get_available_providers()}")

# Test inference
input_tensor = example_input.numpy()
outputs = sess.run(None, {'waveform': input_tensor})
print(f"Embedding output shape: {outputs[0].shape}")
print(f"Embedding sample values: {outputs[0][0][:10]}")
print("✓ ONNX embedding export successful!")

# Check file size
import os
size_mb = os.path.getsize("embedding_model.onnx") / (1024 * 1024)
print(f"\nFile size: {size_mb:.1f} MB")
