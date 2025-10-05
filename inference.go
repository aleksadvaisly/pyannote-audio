package main

/*
#cgo CFLAGS: -I/usr/local/include/onnxruntime
#cgo LDFLAGS: -L/usr/local/lib -lonnxruntime -framework CoreML -framework Foundation
#include <onnxruntime_c_api.h>
#include <stdlib.h>

// Helper function to avoid Go pointer issues
static OrtStatus* create_session(const OrtApi* api, OrtEnv* env,
                                  const char* model_path,
                                  OrtSessionOptions* options,
                                  OrtSession** session) {
    return api->CreateSession(env, model_path, options, session);
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

type ONNXModel struct {
	api     *C.OrtApi
	env     *C.OrtEnv
	session *C.OrtSession
}

func check(status *C.OrtStatus, api *C.OrtApi) {
	if status != nil {
		msg := C.GoString(api.GetErrorMessage(status))
		api.ReleaseStatus(status)
		panic(fmt.Sprintf("ONNX Runtime error: %s", msg))
	}
}

func NewONNXModel(modelPath string) (*ONNXModel, error) {
	m := &ONNXModel{}

	// Get API
	base := C.OrtGetApiBase()
	m.api = base.GetApi(C.ORT_API_VERSION)
	if m.api == nil {
		return nil, fmt.Errorf("failed to get ONNX Runtime API")
	}

	// Create environment
	envName := C.CString("pyannote-go")
	defer C.free(unsafe.Pointer(envName))

	status := m.api.CreateEnv(C.ORT_LOGGING_LEVEL_WARNING, envName, &m.env)
	check(status, m.api)

	// Create session options with CoreML
	var options *C.OrtSessionOptions
	status = m.api.CreateSessionOptions(&options)
	check(status, m.api)
	defer m.api.ReleaseSessionOptions(options)

	// Enable CoreML Execution Provider (Metal)
	fmt.Println("Enabling CoreML Execution Provider (Metal)...")
	status = C.OrtSessionOptionsAppendExecutionProvider_CoreML(options, 0)
	check(status, m.api)

	// Load model
	fmt.Printf("Loading model: %s\n", modelPath)
	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	status = C.create_session(m.api, m.env, cModelPath, options, &m.session)
	check(status, m.api)

	return m, nil
}

func (m *ONNXModel) Predict(audioData []float32) ([]float32, []int64, error) {
	// Prepare input shape: [1, 1, audio_length]
	inputShape := []C.int64_t{1, 1, C.int64_t(len(audioData))}
	inputSize := len(audioData)

	// Create memory info
	var memInfo *C.OrtMemoryInfo
	status := m.api.CreateCpuMemoryInfo(C.OrtArenaAllocator, C.OrtMemTypeDefault, &memInfo)
	check(status, m.api)
	defer m.api.ReleaseMemoryInfo(memInfo)

	// Create input tensor
	var inputTensor *C.OrtValue
	status = m.api.CreateTensorWithDataAsOrtValue(
		memInfo,
		unsafe.Pointer(&audioData[0]),
		C.size_t(inputSize*4), // 4 bytes per float32
		&inputShape[0],
		3,
		C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
		&inputTensor,
	)
	check(status, m.api)
	defer m.api.ReleaseValue(inputTensor)

	// Prepare input/output names
	inputName := C.CString("audio")
	defer C.free(unsafe.Pointer(inputName))
	outputName := C.CString("segmentation")
	defer C.free(unsafe.Pointer(outputName))

	inputNames := []*C.char{inputName}
	outputNames := []*C.char{outputName}

	// Run inference
	fmt.Println("Running inference with Metal acceleration...")
	var outputTensor *C.OrtValue
	status = m.api.Run(
		m.session,
		nil,
		&inputNames[0],
		(**C.OrtValue)(unsafe.Pointer(&inputTensor)),
		1,
		&outputNames[0],
		1,
		&outputTensor,
	)
	check(status, m.api)
	defer m.api.ReleaseValue(outputTensor)

	// Get output shape
	var outputInfo *C.OrtTensorTypeAndShapeInfo
	status = m.api.GetTensorTypeAndShape(outputTensor, &outputInfo)
	check(status, m.api)
	defer m.api.ReleaseTensorTypeAndShapeInfo(outputInfo)

	var dimCount C.size_t
	status = m.api.GetDimensionsCount(outputInfo, &dimCount)
	check(status, m.api)

	dims := make([]C.int64_t, dimCount)
	status = m.api.GetDimensions(outputInfo, &dims[0], dimCount)
	check(status, m.api)

	outputShape := make([]int64, dimCount)
	outputSize := int64(1)
	for i, d := range dims {
		outputShape[i] = int64(d)
		outputSize *= int64(d)
	}

	// Get output data
	var outputData *C.float
	status = m.api.GetTensorMutableData(outputTensor, (*unsafe.Pointer)(unsafe.Pointer(&outputData)))
	check(status, m.api)

	// Copy to Go slice
	result := make([]float32, outputSize)
	for i := int64(0); i < outputSize; i++ {
		result[i] = float32(*(*C.float)(unsafe.Pointer(uintptr(unsafe.Pointer(outputData)) + uintptr(i)*4)))
	}

	return result, outputShape, nil
}

func (m *ONNXModel) Close() {
	if m.session != nil {
		m.api.ReleaseSession(m.session)
	}
	if m.env != nil {
		m.api.ReleaseEnv(m.env)
	}
}

func main() {
	// Load model
	model, err := NewONNXModel("segmentation_model.onnx")
	if err != nil {
		panic(err)
	}
	defer model.Close()

	// Prepare dummy audio data (1 second at 16kHz)
	audioData := make([]float32, 16000)
	for i := range audioData {
		audioData[i] = 0.01 // dummy data
	}

	// Run inference
	output, shape, err := model.Predict(audioData)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Output shape: %v\n", shape)
	fmt.Printf("First 10 values: %v\n", output[:10])
	fmt.Println("✓ Inference completed successfully!")
}
