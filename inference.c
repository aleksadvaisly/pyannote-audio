#include <onnxruntime_c_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_STATUS(status)                                           \
  do {                                                                 \
    if (status != NULL) {                                              \
      const char* msg = g_ort->GetErrorMessage(status);                \
      fprintf(stderr, "Error: %s\n", msg);                             \
      g_ort->ReleaseStatus(status);                                    \
      exit(1);                                                         \
    }                                                                  \
  } while (0)

const OrtApi* g_ort = NULL;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.onnx>\n", argv[0]);
        return 1;
    }

    printf("Initializing ONNX Runtime...\n");
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        fprintf(stderr, "Failed to get ONNX Runtime API\n");
        return 1;
    }

    OrtEnv* env;
    CHECK_STATUS(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "pyannote", &env));

    OrtSessionOptions* session_options;
    CHECK_STATUS(g_ort->CreateSessionOptions(&session_options));

    // Enable CoreML Execution Provider for Metal acceleration
    printf("Enabling CoreML Execution Provider (Metal)...\n");
    uint32_t coreml_flags = 0;
    CHECK_STATUS(OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, coreml_flags));

    // Load model
    printf("Loading model: %s\n", argv[1]);
    OrtSession* session;
    CHECK_STATUS(g_ort->CreateSession(env, argv[1], session_options, &session));

    // Get input/output info
    size_t num_input_nodes;
    CHECK_STATUS(g_ort->SessionGetInputCount(session, &num_input_nodes));
    printf("Number of inputs: %zu\n", num_input_nodes);

    size_t num_output_nodes;
    CHECK_STATUS(g_ort->SessionGetOutputCount(session, &num_output_nodes));
    printf("Number of outputs: %zu\n", num_output_nodes);

    // Prepare input (1 second of audio at 16kHz)
    const int64_t input_shape[] = {1, 1, 16000};
    const size_t input_size = 1 * 1 * 16000;
    float* input_data = (float*)malloc(input_size * sizeof(float));

    // Fill with dummy data (in real use case, load actual audio)
    for (size_t i = 0; i < input_size; i++) {
        input_data[i] = (float)rand() / RAND_MAX * 0.1f - 0.05f;
    }

    // Create input tensor
    OrtMemoryInfo* memory_info;
    CHECK_STATUS(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    OrtValue* input_tensor = NULL;
    CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(
        memory_info, input_data, input_size * sizeof(float),
        input_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

    // Run inference
    printf("Running inference with Metal acceleration...\n");
    const char* input_names[] = {"audio"};
    const char* output_names[] = {"segmentation"};

    OrtValue* output_tensor = NULL;
    CHECK_STATUS(g_ort->Run(session, NULL, input_names,
                            (const OrtValue* const*)&input_tensor, 1,
                            output_names, 1, &output_tensor));

    // Get output shape
    OrtTensorTypeAndShapeInfo* output_info;
    CHECK_STATUS(g_ort->GetTensorTypeAndShape(output_tensor, &output_info));

    size_t output_dim_count;
    CHECK_STATUS(g_ort->GetDimensionsCount(output_info, &output_dim_count));

    int64_t* output_dims = (int64_t*)malloc(output_dim_count * sizeof(int64_t));
    CHECK_STATUS(g_ort->GetDimensions(output_info, output_dims, output_dim_count));

    printf("Output shape: [");
    for (size_t i = 0; i < output_dim_count; i++) {
        printf("%lld%s", output_dims[i], i < output_dim_count - 1 ? ", " : "");
    }
    printf("]\n");

    // Get output data
    float* output_data;
    CHECK_STATUS(g_ort->GetTensorMutableData(output_tensor, (void**)&output_data));

    // Print first few values
    printf("First output values: ");
    size_t output_size = 1;
    for (size_t i = 0; i < output_dim_count; i++) {
        output_size *= output_dims[i];
    }
    for (size_t i = 0; i < (output_size < 10 ? output_size : 10); i++) {
        printf("%.4f ", output_data[i]);
    }
    printf("\n");

    printf("✓ Inference completed successfully!\n");

    // Cleanup
    free(output_dims);
    g_ort->ReleaseTensorTypeAndShapeInfo(output_info);
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    free(input_data);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);

    return 0;
}
