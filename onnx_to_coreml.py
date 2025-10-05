from onnx_coreml import convert
import onnx

print("Loading ONNX model...")
onnx_model = onnx.load("segmentation_model.onnx")

print("Converting ONNX to CoreML...")
coreml_model = convert(
    onnx_model,
    minimum_ios_deployment_target='13',
    target_ios='16'
)

# Save as mlpackage
output_path = "segmentation_model.mlpackage"
coreml_model.save(output_path)
print(f"✓ CoreML model saved to: {output_path}")

# Print model info
print("\nModel info:")
print(f"  Inputs: {coreml_model.input_description}")
print(f"  Outputs: {coreml_model.output_description}")

# Test the model
import numpy as np
print("\nTesting CoreML model...")
test_input = {'audio': np.random.randn(1, 1, 16000).astype(np.float32)}
predictions = coreml_model.predict(test_input)
print(f"  Output shape: {predictions['segmentation'].shape}")
print("✓ CoreML model works!")
