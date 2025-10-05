#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <onnxruntime_c_api.h>

#define SAMPLE_RATE 16000
#define CHUNK_DURATION_SEC 5.0
#define CHUNK_SIZE (int)(SAMPLE_RATE * CHUNK_DURATION_SEC)

const OrtApi* g_ort = NULL;

void check_status(OrtStatus* status) {
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error: %s\n", msg);
        g_ort->ReleaseStatus(status);
        exit(1);
    }
}

// Load audio using FFmpeg and convert to 16kHz mono PCM
float* load_audio_ffmpeg(const char* filepath, size_t* out_num_samples) {
    char cmd[2048];

    // Use FFmpeg to convert to raw PCM: 16kHz, mono, float32
    snprintf(cmd, sizeof(cmd),
        "ffmpeg -i \"%s\" -ar 16000 -ac 1 -f f32le -acodec pcm_f32le - 2>/dev/null",
        filepath);

    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        fprintf(stderr, "Failed to run ffmpeg\n");
        return NULL;
    }

    // Read all PCM data
    size_t capacity = SAMPLE_RATE * 60 * 10; // 10 minutes max
    float* samples = malloc(capacity * sizeof(float));
    size_t num_samples = 0;

    size_t chunk_size = 4096;
    while (num_samples < capacity) {
        size_t read = fread(samples + num_samples, sizeof(float), chunk_size, pipe);
        if (read == 0) break;
        num_samples += read;
    }

    pclose(pipe);

    *out_num_samples = num_samples;
    printf("Loaded %zu samples (%.1f seconds) at %d Hz\n",
           num_samples, (float)num_samples / SAMPLE_RATE, SAMPLE_RATE);

    return samples;
}

// Run segmentation model on audio chunk
float** run_segmentation(OrtSession* session, float* audio, size_t num_samples,
                         int64_t* out_num_frames, int64_t* out_num_classes) {
    // Prepare input tensor [1, 1, num_samples]
    int64_t input_shape[] = {1, 1, num_samples};

    OrtMemoryInfo* memory_info;
    check_status(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    OrtValue* input_tensor = NULL;
    check_status(g_ort->CreateTensorWithDataAsOrtValue(
        memory_info, audio, num_samples * sizeof(float),
        input_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

    // Run inference
    const char* input_names[] = {"audio"};
    const char* output_names[] = {"segmentation"};

    OrtValue* output_tensor = NULL;
    check_status(g_ort->Run(session, NULL, input_names,
                            (const OrtValue* const*)&input_tensor, 1,
                            output_names, 1, &output_tensor));

    // Get output shape
    OrtTensorTypeAndShapeInfo* output_info;
    check_status(g_ort->GetTensorTypeAndShape(output_tensor, &output_info));

    size_t output_dim_count;
    check_status(g_ort->GetDimensionsCount(output_info, &output_dim_count));

    int64_t* output_dims = malloc(output_dim_count * sizeof(int64_t));
    check_status(g_ort->GetDimensions(output_info, output_dims, output_dim_count));

    // Output shape: [batch, num_frames, num_classes]
    *out_num_frames = output_dims[1];
    *out_num_classes = output_dims[2];

    // Get output data
    float* output_data;
    check_status(g_ort->GetTensorMutableData(output_tensor, (void**)&output_data));

    // Copy to 2D array
    float** result = malloc(*out_num_frames * sizeof(float*));
    for (int64_t i = 0; i < *out_num_frames; i++) {
        result[i] = malloc(*out_num_classes * sizeof(float));
        memcpy(result[i], output_data + i * (*out_num_classes),
               *out_num_classes * sizeof(float));
    }

    // Cleanup
    free(output_dims);
    g_ort->ReleaseTensorTypeAndShapeInfo(output_info);
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);

    return result;
}

// Apply sigmoid activation
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Simple speaker diarization using segmentation output
void process_segmentation(float** seg_output, int64_t num_frames, int64_t num_classes,
                          float frame_duration) {
    printf("\n=== SPEAKER DIARIZATION RESULTS ===\n");
    printf("Frame duration: %.3f seconds\n", frame_duration);
    printf("Number of frames: %lld\n", (long long)num_frames);
    printf("Number of speaker classes: %lld\n\n", (long long)num_classes);

    // Apply sigmoid to convert logits to probabilities
    for (int64_t frame = 0; frame < num_frames; frame++) {
        for (int64_t spk = 0; spk < num_classes; spk++) {
            seg_output[frame][spk] = sigmoid(seg_output[frame][spk]);
        }
    }

    // Debug: print first frame probabilities
    printf("First frame probabilities (after sigmoid):\n");
    for (int64_t spk = 0; spk < num_classes; spk++) {
        printf("  Speaker %lld: %.4f\n", (long long)spk, seg_output[0][spk]);
    }
    printf("\n");

    // Find max probability across all frames/speakers for statistics
    float max_overall = 0.0;
    for (int64_t frame = 0; frame < num_frames; frame++) {
        for (int64_t spk = 0; spk < num_classes; spk++) {
            if (seg_output[frame][spk] > max_overall) {
                max_overall = seg_output[frame][spk];
            }
        }
    }
    printf("Max probability across all frames: %.4f\n\n", max_overall);

    // For each frame, find dominant speaker (simple argmax)
    int current_speaker = -1;
    float segment_start = 0.0;
    const float threshold = 0.3; // Speech detection threshold (lowered)

    for (int64_t frame = 0; frame < num_frames; frame++) {
        float time = frame * frame_duration;

        // Find speaker with max probability
        int max_speaker = 0;
        float max_prob = seg_output[frame][0];

        for (int64_t spk = 1; spk < num_classes; spk++) {
            if (seg_output[frame][spk] > max_prob) {
                max_prob = seg_output[frame][spk];
                max_speaker = spk;
            }
        }

        // Only consider as speech if probability > threshold
        if (max_prob < threshold) {
            max_speaker = -1; // silence
        }

        // Detect speaker change
        if (max_speaker != current_speaker) {
            if (current_speaker >= 0) {
                printf("%.1fs - %.1fs: SPEAKER_%d\n",
                       segment_start, time, current_speaker);
            }
            current_speaker = max_speaker;
            segment_start = time;
        }
    }

    // Print final segment
    if (current_speaker >= 0) {
        printf("%.1fs - %.1fs: SPEAKER_%d\n",
               segment_start, num_frames * frame_duration, current_speaker);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <audio_file>\n", argv[0]);
        return 1;
    }

    const char* audio_file = argv[1];

    printf("=== PyAnnote Audio Diarization POC ===\n\n");

    // Initialize ONNX Runtime
    printf("Initializing ONNX Runtime...\n");
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        fprintf(stderr, "Failed to get ONNX Runtime API\n");
        return 1;
    }

    OrtEnv* env;
    check_status(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "diarization", &env));

    OrtSessionOptions* session_options;
    check_status(g_ort->CreateSessionOptions(&session_options));

    // Load segmentation model
    printf("Loading segmentation model...\n");
    OrtSession* session;
    check_status(g_ort->CreateSession(env, "segmentation_model.onnx",
                                      session_options, &session));

    // Load audio
    printf("Loading audio: %s\n", audio_file);
    size_t num_samples;
    float* audio = load_audio_ffmpeg(audio_file, &num_samples);
    if (!audio) {
        fprintf(stderr, "Failed to load audio\n");
        return 1;
    }

    // Process audio in chunks
    printf("\nProcessing audio...\n");

    // For simplicity, process first chunk only (5 seconds)
    size_t chunk_samples = num_samples < CHUNK_SIZE ? num_samples : CHUNK_SIZE;

    int64_t num_frames, num_classes;
    float** segmentation = run_segmentation(session, audio, chunk_samples,
                                           &num_frames, &num_classes);

    // Calculate frame duration
    float frame_duration = (float)chunk_samples / SAMPLE_RATE / num_frames;

    // Process and display results
    process_segmentation(segmentation, num_frames, num_classes, frame_duration);

    // Cleanup
    for (int64_t i = 0; i < num_frames; i++) {
        free(segmentation[i]);
    }
    free(segmentation);
    free(audio);

    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);

    printf("\n✓ Processing complete!\n");

    return 0;
}
