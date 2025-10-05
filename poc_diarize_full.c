#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Accelerate/Accelerate.h>
#include <onnxruntime_c_api.h>

#define SAMPLE_RATE 16000
#define CHUNK_DURATION_SEC 5.0
#define CHUNK_SIZE (int)(SAMPLE_RATE * CHUNK_DURATION_SEC)

// Mel spectrogram parameters (for embedding model)
#define MEL_BINS 80
#define FRAME_LENGTH_MS 25
#define FRAME_SHIFT_MS 10
#define FRAME_LENGTH ((int)(SAMPLE_RATE * FRAME_LENGTH_MS / 1000.0))
#define FRAME_SHIFT ((int)(SAMPLE_RATE * FRAME_SHIFT_MS / 1000.0))
#define FFT_SIZE 512

const OrtApi* g_ort = NULL;

typedef struct {
    float** data;  // [num_frames][num_bins]
    int64_t num_frames;
    int64_t num_bins;
} MelSpectrogram;

typedef struct {
    float* data;
    int64_t size;
} Embedding;

void check_status(OrtStatus* status) {
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error: %s\n", msg);
        g_ort->ReleaseStatus(status);
        exit(1);
    }
}

// Load audio using FFmpeg
float* load_audio_ffmpeg(const char* filepath, size_t* out_num_samples) {
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
        "ffmpeg -i \"%s\" -ar 16000 -ac 1 -f f32le -acodec pcm_f32le - 2>/dev/null",
        filepath);

    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        fprintf(stderr, "Failed to run ffmpeg\n");
        return NULL;
    }

    size_t capacity = SAMPLE_RATE * 60 * 10;
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

// Compute mel filterbank features using Accelerate framework
MelSpectrogram* compute_mel_spectrogram(float* audio, size_t num_samples) {
    printf("Computing mel spectrogram...\n");

    int64_t num_frames = (num_samples - FRAME_LENGTH) / FRAME_SHIFT + 1;

    MelSpectrogram* mel = malloc(sizeof(MelSpectrogram));
    mel->num_frames = num_frames;
    mel->num_bins = MEL_BINS;
    mel->data = malloc(num_frames * sizeof(float*));

    // Setup FFT
    FFTSetup fft_setup = vDSP_create_fftsetup(9, kFFTRadix2); // 2^9 = 512

    // Create Hamming window
    float* window = malloc(FRAME_LENGTH * sizeof(float));
    vDSP_hamm_window(window, FRAME_LENGTH, 0);

    // Mel filterbank (simplified - triangular filters)
    float mel_low = 2595.0 * log10(1.0 + 0.0 / 700.0);
    float mel_high = 2595.0 * log10(1.0 + (SAMPLE_RATE / 2.0) / 700.0);

    for (int64_t frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        mel->data[frame_idx] = malloc(MEL_BINS * sizeof(float));

        // Extract frame
        float* frame = malloc(FRAME_LENGTH * sizeof(float));
        memcpy(frame, audio + frame_idx * FRAME_SHIFT, FRAME_LENGTH * sizeof(float));

        // Apply window
        vDSP_vmul(frame, 1, window, 1, frame, 1, FRAME_LENGTH);

        // Prepare for FFT (split complex)
        float* real = calloc(FFT_SIZE / 2, sizeof(float));
        float* imag = calloc(FFT_SIZE / 2, sizeof(float));

        // Copy frame data (zero pad if needed)
        for (int i = 0; i < FRAME_LENGTH && i < FFT_SIZE; i++) {
            if (i % 2 == 0) {
                real[i / 2] = frame[i];
            } else {
                imag[i / 2] = frame[i];
            }
        }

        DSPSplitComplex split_complex = {real, imag};

        // Perform FFT
        vDSP_fft_zrip(fft_setup, &split_complex, 1, 9, FFT_FORWARD);

        // Compute power spectrum
        float* power_spectrum = malloc((FFT_SIZE / 2) * sizeof(float));
        vDSP_zvabs(&split_complex, 1, power_spectrum, 1, FFT_SIZE / 2);

        // Square for power
        vDSP_vsq(power_spectrum, 1, power_spectrum, 1, FFT_SIZE / 2);

        // Apply mel filterbank (simplified - just downsample for POC)
        // Proper implementation would use triangular mel filters
        for (int mel_idx = 0; mel_idx < MEL_BINS; mel_idx++) {
            int start_bin = mel_idx * (FFT_SIZE / 2) / MEL_BINS;
            int end_bin = (mel_idx + 1) * (FFT_SIZE / 2) / MEL_BINS;

            float sum = 0.0;
            for (int bin = start_bin; bin < end_bin; bin++) {
                sum += power_spectrum[bin];
            }

            // Log mel spectrogram
            mel->data[frame_idx][mel_idx] = log(sum / (end_bin - start_bin) + 1e-10);
        }

        free(frame);
        free(real);
        free(imag);
        free(power_spectrum);
    }

    vDSP_destroy_fftsetup(fft_setup);
    free(window);

    printf("Computed mel spectrogram: %lld frames x %lld bins\n",
           (long long)mel->num_frames, (long long)mel->num_bins);

    return mel;
}

// Run segmentation model
float** run_segmentation(OrtSession* session, float* audio, size_t num_samples,
                         int64_t* out_num_frames, int64_t* out_num_classes) {
    int64_t input_shape[] = {1, 1, num_samples};

    OrtMemoryInfo* memory_info;
    check_status(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    OrtValue* input_tensor = NULL;
    check_status(g_ort->CreateTensorWithDataAsOrtValue(
        memory_info, audio, num_samples * sizeof(float),
        input_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

    const char* input_names[] = {"audio"};
    const char* output_names[] = {"segmentation"};

    OrtValue* output_tensor = NULL;
    check_status(g_ort->Run(session, NULL, input_names,
                            (const OrtValue* const*)&input_tensor, 1,
                            output_names, 1, &output_tensor));

    OrtTensorTypeAndShapeInfo* output_info;
    check_status(g_ort->GetTensorTypeAndShape(output_tensor, &output_info));

    size_t output_dim_count;
    check_status(g_ort->GetDimensionsCount(output_info, &output_dim_count));

    int64_t* output_dims = malloc(output_dim_count * sizeof(int64_t));
    check_status(g_ort->GetDimensions(output_info, output_dims, output_dim_count));

    *out_num_frames = output_dims[1];
    *out_num_classes = output_dims[2];

    float* output_data;
    check_status(g_ort->GetTensorMutableData(output_tensor, (void**)&output_data));

    float** result = malloc(*out_num_frames * sizeof(float*));
    for (int64_t i = 0; i < *out_num_frames; i++) {
        result[i] = malloc(*out_num_classes * sizeof(float));
        memcpy(result[i], output_data + i * (*out_num_classes),
               *out_num_classes * sizeof(float));
    }

    free(output_dims);
    g_ort->ReleaseTensorTypeAndShapeInfo(output_info);
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);

    return result;
}

// Run embedding model on mel spectrogram
Embedding* run_embedding(OrtSession* session, MelSpectrogram* mel) {
    // Prepare input: flatten mel spectrogram to 1D array
    size_t total_size = mel->num_frames * mel->num_bins;
    float* mel_flat = malloc(total_size * sizeof(float));

    for (int64_t i = 0; i < mel->num_frames; i++) {
        memcpy(mel_flat + i * mel->num_bins, mel->data[i], mel->num_bins * sizeof(float));
    }

    // Input shape: [batch=1, num_frames, num_bins]
    int64_t input_shape[] = {1, mel->num_frames, mel->num_bins};

    OrtMemoryInfo* memory_info;
    check_status(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    OrtValue* input_tensor = NULL;
    check_status(g_ort->CreateTensorWithDataAsOrtValue(
        memory_info, mel_flat, total_size * sizeof(float),
        input_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

    const char* input_names[] = {"mel_filterbank"};
    const char* output_names[] = {"embedding"};

    OrtValue* output_tensor = NULL;
    check_status(g_ort->Run(session, NULL, input_names,
                            (const OrtValue* const*)&input_tensor, 1,
                            output_names, 1, &output_tensor));

    // Get output
    OrtTensorTypeAndShapeInfo* output_info;
    check_status(g_ort->GetTensorTypeAndShape(output_tensor, &output_info));

    size_t output_dim_count;
    check_status(g_ort->GetDimensionsCount(output_info, &output_dim_count));

    int64_t* output_dims = malloc(output_dim_count * sizeof(int64_t));
    check_status(g_ort->GetDimensions(output_info, output_dims, output_dim_count));

    int64_t embedding_size = output_dims[1];

    float* output_data;
    check_status(g_ort->GetTensorMutableData(output_tensor, (void**)&output_data));

    Embedding* emb = malloc(sizeof(Embedding));
    emb->size = embedding_size;
    emb->data = malloc(embedding_size * sizeof(float));
    memcpy(emb->data, output_data, embedding_size * sizeof(float));

    free(output_dims);
    free(mel_flat);
    g_ort->ReleaseTensorTypeAndShapeInfo(output_info);
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);

    return emb;
}

// Compute cosine similarity between two embeddings
float cosine_similarity(Embedding* e1, Embedding* e2) {
    if (e1->size != e2->size) return 0.0;

    float dot = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (int64_t i = 0; i < e1->size; i++) {
        dot += e1->data[i] * e2->data[i];
        norm1 += e1->data[i] * e1->data[i];
        norm2 += e2->data[i] * e2->data[i];
    }

    return dot / (sqrt(norm1) * sqrt(norm2) + 1e-10);
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Process segmentation with embeddings
void process_with_embeddings(float** seg_output, int64_t num_frames, int64_t num_classes,
                             float frame_duration, float* audio, size_t num_samples,
                             OrtSession* embedding_session) {
    printf("\n=== SPEAKER DIARIZATION WITH EMBEDDINGS ===\n");

    // Apply sigmoid to segmentation output
    for (int64_t frame = 0; frame < num_frames; frame++) {
        for (int64_t spk = 0; spk < num_classes; spk++) {
            seg_output[frame][spk] = sigmoid(seg_output[frame][spk]);
        }
    }

    // Detect speech segments
    typedef struct {
        float start_time;
        float end_time;
        int speaker_id;
        Embedding* embedding;
    } Segment;

    Segment* segments = malloc(num_frames * sizeof(Segment));
    int num_segments = 0;

    int current_speaker = -1;
    float segment_start = 0.0;
    const float threshold = 0.3;

    printf("Detecting speech segments...\n");

    for (int64_t frame = 0; frame < num_frames; frame++) {
        float time = frame * frame_duration;

        int max_speaker = 0;
        float max_prob = seg_output[frame][0];

        for (int64_t spk = 1; spk < num_classes; spk++) {
            if (seg_output[frame][spk] > max_prob) {
                max_prob = seg_output[frame][spk];
                max_speaker = spk;
            }
        }

        if (max_prob < threshold) {
            max_speaker = -1;
        }

        if (max_speaker != current_speaker) {
            if (current_speaker >= 0 && time - segment_start > 0.5) {
                // Save segment
                segments[num_segments].start_time = segment_start;
                segments[num_segments].end_time = time;
                segments[num_segments].speaker_id = current_speaker;
                segments[num_segments].embedding = NULL;
                num_segments++;
            }
            current_speaker = max_speaker;
            segment_start = time;
        }
    }

    // Add final segment
    if (current_speaker >= 0) {
        segments[num_segments].start_time = segment_start;
        segments[num_segments].end_time = num_frames * frame_duration;
        segments[num_segments].speaker_id = current_speaker;
        segments[num_segments].embedding = NULL;
        num_segments++;
    }

    printf("Found %d speech segments\n", num_segments);

    // Compute embeddings for each segment
    printf("Computing embeddings for segments...\n");
    for (int i = 0; i < num_segments; i++) {
        size_t start_sample = (size_t)(segments[i].start_time * SAMPLE_RATE);
        size_t end_sample = (size_t)(segments[i].end_time * SAMPLE_RATE);

        if (end_sample > num_samples) end_sample = num_samples;
        size_t segment_samples = end_sample - start_sample;

        if (segment_samples < SAMPLE_RATE) continue; // Skip segments < 1 second

        // Compute mel spectrogram for this segment
        MelSpectrogram* mel = compute_mel_spectrogram(audio + start_sample, segment_samples);

        // Get embedding
        segments[i].embedding = run_embedding(embedding_session, mel);

        // Cleanup mel
        for (int64_t j = 0; j < mel->num_frames; j++) {
            free(mel->data[j]);
        }
        free(mel->data);
        free(mel);

        printf("  Segment %d (%.1fs - %.1fs): embedding size %lld\n",
               i, segments[i].start_time, segments[i].end_time,
               (long long)segments[i].embedding->size);
    }

    // Cluster segments based on embedding similarity
    printf("\nClustering speakers based on embeddings...\n");

    int* cluster_ids = malloc(num_segments * sizeof(int));
    for (int i = 0; i < num_segments; i++) {
        cluster_ids[i] = -1;
    }

    int next_cluster_id = 0;
    const float similarity_threshold = 0.7; // Cosine similarity threshold

    for (int i = 0; i < num_segments; i++) {
        if (!segments[i].embedding) continue;

        if (cluster_ids[i] == -1) {
            // Start new cluster
            cluster_ids[i] = next_cluster_id++;
        }

        // Find similar segments
        for (int j = i + 1; j < num_segments; j++) {
            if (!segments[j].embedding) continue;
            if (cluster_ids[j] != -1) continue;

            float sim = cosine_similarity(segments[i].embedding, segments[j].embedding);

            if (sim > similarity_threshold) {
                cluster_ids[j] = cluster_ids[i];
            }
        }
    }

    // Print results
    printf("\n=== FINAL DIARIZATION RESULTS ===\n");
    for (int i = 0; i < num_segments; i++) {
        if (cluster_ids[i] >= 0) {
            printf("%.1fs - %.1fs: SPEAKER_%d\n",
                   segments[i].start_time, segments[i].end_time, cluster_ids[i]);
        }
    }

    // Cleanup
    for (int i = 0; i < num_segments; i++) {
        if (segments[i].embedding) {
            free(segments[i].embedding->data);
            free(segments[i].embedding);
        }
    }
    free(segments);
    free(cluster_ids);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <audio_file>\n", argv[0]);
        return 1;
    }

    const char* audio_file = argv[1];

    printf("=== PyAnnote Full Diarization with Embeddings ===\n\n");

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

    // Load models
    printf("Loading segmentation model...\n");
    OrtSession* seg_session;
    check_status(g_ort->CreateSession(env, "segmentation_model.onnx",
                                      session_options, &seg_session));

    printf("Loading embedding model...\n");
    OrtSession* emb_session;
    check_status(g_ort->CreateSession(env, "embedding_model.onnx",
                                      session_options, &emb_session));

    // Load audio
    printf("Loading audio: %s\n", audio_file);
    size_t num_samples;
    float* audio = load_audio_ffmpeg(audio_file, &num_samples);
    if (!audio) {
        fprintf(stderr, "Failed to load audio\n");
        return 1;
    }

    // Process in chunks
    printf("\nProcessing audio in chunks...\n");

    size_t offset = 0;
    while (offset < num_samples) {
        size_t chunk_samples = (num_samples - offset) < CHUNK_SIZE ?
                              (num_samples - offset) : CHUNK_SIZE;

        printf("\nChunk: %.1fs - %.1fs\n",
               (float)offset / SAMPLE_RATE,
               (float)(offset + chunk_samples) / SAMPLE_RATE);

        // Run segmentation
        int64_t num_frames, num_classes;
        float** segmentation = run_segmentation(seg_session, audio + offset,
                                               chunk_samples, &num_frames, &num_classes);

        float frame_duration = (float)chunk_samples / SAMPLE_RATE / num_frames;

        // Process with embeddings
        process_with_embeddings(segmentation, num_frames, num_classes, frame_duration,
                               audio + offset, chunk_samples, emb_session);

        // Cleanup
        for (int64_t i = 0; i < num_frames; i++) {
            free(segmentation[i]);
        }
        free(segmentation);

        offset += chunk_samples;
    }

    // Cleanup
    free(audio);
    g_ort->ReleaseSession(emb_session);
    g_ort->ReleaseSession(seg_session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);

    printf("\n✓ Processing complete!\n");

    return 0;
}
