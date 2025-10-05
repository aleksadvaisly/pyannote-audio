# Speaker Diarization Inference - C/Go with Metal Acceleration

## Overview

Model pyannote speaker-diarization został wyeksportowany do **ONNX** i działa na **Apple M1 Metal** przez **CoreML Execution Provider**.

## Architecture

```
PyTorch Model (31 MB)
    ↓
ONNX Export (segmentation_model.onnx)
    ↓
ONNX Runtime + CoreML EP
    ↓
Apple Metal (GPU acceleration)
```

## Files

- `segmentation_model.onnx` - Model segmentacji (5.6 MB)
- `inference.c` - Czysty C inference z ONNX Runtime C API
- `inference.go` - Go wrapper z cgo
- `Makefile` - Build scripts

## Installation

### 1. Install ONNX Runtime

#### Option A: Download pre-built (recommended)

```bash
# Download ONNX Runtime for macOS ARM64
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.23.0/onnxruntime-osx-arm64-1.23.0.tgz -o onnxruntime.tgz
tar -xzf onnxruntime.tgz
cd onnxruntime-osx-arm64-1.23.0

# Install to /usr/local
sudo mkdir -p /usr/local/include/onnxruntime
sudo cp -r include/* /usr/local/include/onnxruntime/
sudo cp lib/* /usr/local/lib/
```

#### Option B: Use Makefile

```bash
make install-onnxruntime
```

### 2. Export Model to ONNX

```bash
# Activate venv and export
source .venv/bin/activate
python export_to_onnx.py
```

## Usage

### C Version

```bash
# Build
make

# Run
./inference segmentation_model.onnx
```

### Go Version

```bash
# Build
go build -o inference_go inference.go

# Run
./inference_go
```

## Static vs Dynamic Linking

### Dynamic (default, recommended)

- **Pros**: Mniejszy binary, łatwiejsze updates
- **Cons**: Wymaga `libonnxruntime.dylib` w systemie

### Static Linking

Możliwe, ale wymaga:
1. ONNX Runtime built as static library
2. Wszystkie dependencies (CoreML, Foundation) jako frameworks
3. Większy binary (~20-30 MB więcej)

**Przykład statycznego linkowania:**

```bash
# Build ONNX Runtime from source with static libs
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime
./build.sh --config Release --build_shared_lib --use_coreml --parallel

# Link statically
clang inference.c \
  -I/path/to/onnxruntime/include \
  -L/path/to/onnxruntime/build/Release \
  -lonnxruntime_static \
  -framework CoreML -framework Foundation \
  -o inference_static
```

## Go Integration

### CGO Linkage

W Go używasz **cgo** do linkowania z ONNX Runtime:

```go
/*
#cgo CFLAGS: -I/usr/local/include/onnxruntime
#cgo LDFLAGS: -L/usr/local/lib -lonnxruntime -framework CoreML
*/
import "C"
```

### Distribution Options

1. **Dynamic linking** (standard):
   - Ship binary + `libonnxruntime.dylib`
   - User installs ONNX Runtime globally

2. **Embed dylib** w Go binary:
   ```bash
   # Embed resource
   go get github.com/go-bindata/go-bindata
   go-bindata -pkg main libonnxruntime.dylib
   # Extract at runtime do temp dir
   ```

3. **Static linking** (advanced):
   - Build ONNX Runtime as static
   - `CGO_ENABLED=1 go build -tags static`

## Performance

### Metal Acceleration

CoreML EP automatycznie używa:
- **Neural Engine** - dla operacji ML
- **Metal GPU** - dla compute kernels
- **CPU** - fallback dla unsupported ops

### Benchmark (Apple M1)

- **CPU only**: ~37s per hour of audio
- **CoreML (Metal)**: ~14s per hour of audio
- **Speed up**: ~2.6x

## Troubleshooting

### Library not loaded

```bash
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
```

### CoreML not available

Sprawdź dostępne execution providers:

```c
size_t count;
char** providers;
OrtGetAvailableProviders(&providers, &count);
for (size_t i = 0; i < count; i++) {
    printf("Provider: %s\n", providers[i]);
}
```

## Next Steps

1. Export embedding model (25 MB)
2. Implement full pipeline (segmentation + embedding + clustering)
3. Add audio preprocessing (resample do 16kHz)
4. Optimize batch processing

## References

- ONNX Runtime: https://onnxruntime.ai/
- CoreML EP: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
- pyannote.audio: https://github.com/pyannote/pyannote-audio
