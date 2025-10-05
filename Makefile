# Makefile for ONNX Runtime C inference with CoreML/Metal

# ONNX Runtime paths (adjust if needed)
ONNX_RT_VERSION = 1.23.0
ONNX_RT_DIR = /usr/local
ONNX_RT_INCLUDE = $(ONNX_RT_DIR)/include/onnxruntime
ONNX_RT_LIB = $(ONNX_RT_DIR)/lib

# Compiler settings
CC = clang
CFLAGS = -I$(ONNX_RT_INCLUDE) -Wall -O2
LDFLAGS = -L$(ONNX_RT_LIB) -lonnxruntime
FRAMEWORKS = -framework CoreML -framework Foundation

# Target
TARGET = inference
MODEL = segmentation_model.onnx

all: $(TARGET)

$(TARGET): inference.c
	$(CC) $(CFLAGS) inference.c -o $(TARGET) $(LDFLAGS) $(FRAMEWORKS)

run: $(TARGET)
	./$(TARGET) $(MODEL)

clean:
	rm -f $(TARGET)

# Download ONNX Runtime if not installed
install-onnxruntime:
	@echo "Downloading ONNX Runtime for macOS ARM64..."
	curl -L https://github.com/microsoft/onnxruntime/releases/download/v$(ONNX_RT_VERSION)/onnxruntime-osx-arm64-$(ONNX_RT_VERSION).tgz -o onnxruntime.tgz
	tar -xzf onnxruntime.tgz
	sudo cp -r onnxruntime-osx-arm64-$(ONNX_RT_VERSION)/include/* $(ONNX_RT_INCLUDE)/
	sudo cp onnxruntime-osx-arm64-$(ONNX_RT_VERSION)/lib/* $(ONNX_RT_LIB)/
	rm -rf onnxruntime.tgz onnxruntime-osx-arm64-$(ONNX_RT_VERSION)
	@echo "✓ ONNX Runtime installed to $(ONNX_RT_DIR)"

.PHONY: all run clean install-onnxruntime
