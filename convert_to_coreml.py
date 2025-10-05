import torch
import coremltools as ct
from pyannote.audio import Model
import os

os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN', '')

# Load segmentation model
print("Loading segmentation model...")
seg_model = Model.from_pretrained(
    "pyannote/segmentation-3.0",
    use_auth_token=os.environ.get('HF_TOKEN')
)
seg_model.eval()

# Create example input (1 second of audio at 16kHz)
example_input = torch.randn(1, 1, 16000)

print("Tracing segmentation model...")
traced_seg = torch.jit.trace(seg_model, example_input)

print("Converting to CoreML...")
coreml_seg = ct.convert(
    traced_seg,
    inputs=[ct.TensorType(name="audio", shape=(1, 1, 16000))],
    outputs=[ct.TensorType(name="segmentation")],
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.ALL
)

output_path = "segmentation_model.mlpackage"
coreml_seg.save(output_path)
print(f"Segmentation model saved to: {output_path}")

# Load embedding model
print("\nLoading embedding model...")
emb_model = Model.from_pretrained(
    "pyannote/embedding",
    use_auth_token=os.environ.get('HF_TOKEN')
)
emb_model.eval()

# Example input for embedding (typically mel spectrogram)
example_emb_input = torch.randn(1, 1, 80, 300)  # batch, channels, freq_bins, time_frames

print("Tracing embedding model...")
traced_emb = torch.jit.trace(emb_model, example_emb_input)

print("Converting embedding to CoreML...")
coreml_emb = ct.convert(
    traced_emb,
    inputs=[ct.TensorType(name="spectrogram", shape=(1, 1, 80, 300))],
    outputs=[ct.TensorType(name="embedding")],
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.ALL
)

output_path_emb = "embedding_model.mlpackage"
coreml_emb.save(output_path_emb)
print(f"Embedding model saved to: {output_path_emb}")
