import torch
import torch.onnx
from pyannote.audio import Model
import os

os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN', '')

class ModelWrapper(torch.nn.Module):
    """Wrapper to avoid Lightning-specific code during export"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

# Load and export segmentation model
print("Loading segmentation model...")
seg_model = Model.from_pretrained(
    "pyannote/segmentation-3.0",
    use_auth_token=os.environ.get('HF_TOKEN')
)
seg_model.eval()

wrapped_seg = ModelWrapper(seg_model)
wrapped_seg.eval()

# Create example input (1 second of audio at 16kHz)
example_input = torch.randn(1, 1, 16000)

print("Exporting segmentation to ONNX...")
torch.onnx.export(
    wrapped_seg,
    example_input,
    "segmentation_model.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['audio'],
    output_names=['segmentation'],
    dynamic_axes={
        'audio': {2: 'audio_length'},
        'segmentation': {1: 'time_frames'}
    }
)
print("Segmentation model exported to: segmentation_model.onnx")

# Test with ONNX Runtime
import onnxruntime as ort
print("\nTesting ONNX model...")
sess = ort.InferenceSession("segmentation_model.onnx")
print(f"Available execution providers: {ort.get_available_providers()}")
print(f"Using providers: {sess.get_providers()}")

# Test inference
input_tensor = example_input.numpy()
outputs = sess.run(None, {'audio': input_tensor})
print(f"Output shape: {outputs[0].shape}")
print("✓ ONNX export successful!")
