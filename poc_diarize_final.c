#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Accelerate/Accelerate.h>
#include <onnxruntime_c_api.h>

#define SAMPLE_RATE 16000
#define CHUNK_DURATION_SEC 5.0
#define CHUNK_SIZE (int)(SAMPLE_RATE * CHUNK_DURATION_SEC)

#define MEL_BINS 80
#define FRAME_LENGTH_MS 25
#define FRAME_SHIFT_MS 10
#define FRAME_LENGTH ((int)(SAMPLE_RATE * FRAME_LENGTH_MS / 1000.0))
#define FRAME_SHIFT ((int)(SAMPLE_RATE * FRAME_SHIFT_MS / 1000.0))
#define FFT_SIZE 512

const OrtApi* g_ort = NULL;
FILE* output_file = NULL;

typedef struct {
    float** data;
    int64_t num_frames;
    int64_t num_bins;
} MelSpectrogram;

typedef struct {
    float* data;
    int64_t size;
} Embedding;

typedef struct {
    float start_time;
    float end_time;
    int speaker_id;
    Embedding* embedding;
} Segment;

void check_status(OrtStatus* status) {
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error: %s\n", msg);
        g_ort->ReleaseStatus(status);
        exit(1);
    }
}

float* load_audio_ffmpeg(const char* filepath, size_t* out_num_samples) {
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
        "ffmpeg -i \"%s\" -ar 16000 -ac 1 -f f32le -acodec pcm_f32le - 2>/dev/null",
        filepath);

    FILE* pipe = popen(cmd, "r");
    if (!pipe) return NULL;

    size_t capacity = SAMPLE_RATE * 60 * 10;
    float* samples = malloc(capacity * sizeof(float));
    size_t num_samples = 0;

    while (num_samples < capacity) {
        size_t read = fread(samples + num_samples, sizeof(float), 4096, pipe);
        if (read == 0) break;
        num_samples += read;
    }

    pclose(pipe);
    *out_num_samples = num_samples;

    fprintf(output_file, "Loaded %zu samples (%.1f seconds) at %d Hz\n",
           num_samples, (float)num_samples / SAMPLE_RATE, SAMPLE_RATE);
    printf("Loaded audio: %.1f seconds\n", (float)num_samples / SAMPLE_RATE);

    return samples;
}

MelSpectrogram* compute_mel_spectrogram(float* audio, size_t num_samples) {
    int64_t num_frames = (num_samples - FRAME_LENGTH) / FRAME_SHIFT + 1;

    MelSpectrogram* mel = malloc(sizeof(MelSpectrogram));
    mel->num_frames = num_frames;
    mel->num_bins = MEL_BINS;
    mel->data = malloc(num_frames * sizeof(float*));

    FFTSetup fft_setup = vDSP_create_fftsetup(9, kFFTRadix2);
    float* window = malloc(FRAME_LENGTH * sizeof(float));
    vDSP_hamm_window(window, FRAME_LENGTH, 0);

    for (int64_t frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        mel->data[frame_idx] = malloc(MEL_BINS * sizeof(float));

        float* frame = malloc(FRAME_LENGTH * sizeof(float));
        memcpy(frame, audio + frame_idx * FRAME_SHIFT, FRAME_LENGTH * sizeof(float));
        vDSP_vmul(frame, 1, window, 1, frame, 1, FRAME_LENGTH);

        float* real = calloc(FFT_SIZE / 2, sizeof(float));
        float* imag = calloc(FFT_SIZE / 2, sizeof(float));

        for (int i = 0; i < FRAME_LENGTH && i < FFT_SIZE; i++) {
            if (i % 2 == 0) real[i / 2] = frame[i];
            else imag[i / 2] = frame[i];
        }

        DSPSplitComplex split_complex = {real, imag};
        vDSP_fft_zrip(fft_setup, &split_complex, 1, 9, FFT_FORWARD);

        float* power_spectrum = malloc((FFT_SIZE / 2) * sizeof(float));
        vDSP_zvabs(&split_complex, 1, power_spectrum, 1, FFT_SIZE / 2);
        vDSP_vsq(power_spectrum, 1, power_spectrum, 1, FFT_SIZE / 2);

        for (int mel_idx = 0; mel_idx < MEL_BINS; mel_idx++) {
            int start_bin = mel_idx * (FFT_SIZE / 2) / MEL_BINS;
            int end_bin = (mel_idx + 1) * (FFT_SIZE / 2) / MEL_BINS;

            float sum = 0.0;
            for (int bin = start_bin; bin < end_bin; bin++) {
                sum += power_spectrum[bin];
            }

            mel->data[frame_idx][mel_idx] = log(sum / (end_bin - start_bin) + 1e-10);
        }

        free(frame);
        free(real);
        free(imag);
        free(power_spectrum);
    }

    vDSP_destroy_fftsetup(fft_setup);
    free(window);

    return mel;
}

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

Embedding* run_embedding(OrtSession* session, MelSpectrogram* mel) {
    size_t total_size = mel->num_frames * mel->num_bins;
    float* mel_flat = malloc(total_size * sizeof(float));

    for (int64_t i = 0; i < mel->num_frames; i++) {
        memcpy(mel_flat + i * mel->num_bins, mel->data[i], mel->num_bins * sizeof(float));
    }

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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <audio_file>\n", argv[0]);
        return 1;
    }

    const char* audio_file = argv[1];

    // Open output file
    output_file = fopen("poc_output_2.txt", "w");
    if (!output_file) {
        fprintf(stderr, "Failed to open poc_output_2.txt\n");
        return 1;
    }

    fprintf(output_file, "=== PyAnnote Speaker Diarization Results ===\n\n");
    printf("=== PyAnnote Speaker Diarization ===\n");
    printf("Output will be saved to: poc_output_2.txt\n\n");

    // Initialize ONNX Runtime
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        fprintf(stderr, "Failed to get ONNX Runtime API\n");
        return 1;
    }

    OrtEnv* env;
    check_status(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "diarization", &env));

    OrtSessionOptions* session_options;
    check_status(g_ort->CreateSessionOptions(&session_options));

    printf("Loading models...\n");
    OrtSession* seg_session;
    check_status(g_ort->CreateSession(env, "segmentation_model.onnx",
                                      session_options, &seg_session));

    OrtSession* emb_session;
    check_status(g_ort->CreateSession(env, "embedding_model.onnx",
                                      session_options, &emb_session));

    // Load audio
    printf("Loading audio...\n");
    size_t num_samples;
    float* audio = load_audio_ffmpeg(audio_file, &num_samples);
    if (!audio) {
        fprintf(stderr, "Failed to load audio\n");
        return 1;
    }

    // Collect all segments globally
    Segment* all_segments = malloc(1000 * sizeof(Segment));
    int total_segments = 0;

    // Process in chunks
    printf("Processing audio...\n");
    fprintf(output_file, "\n=== Processing ===\n");

    size_t offset = 0;
    int chunk_num = 0;

    while (offset < num_samples) {
        size_t chunk_samples = (num_samples - offset) < CHUNK_SIZE ?
                              (num_samples - offset) : CHUNK_SIZE;

        printf("  Chunk %d: %.1fs - %.1fs\n", chunk_num,
               (float)offset / SAMPLE_RATE,
               (float)(offset + chunk_samples) / SAMPLE_RATE);

        // Run segmentation
        int64_t num_frames, num_classes;
        float** segmentation = run_segmentation(seg_session, audio + offset,
                                               chunk_samples, &num_frames, &num_classes);

        float frame_duration = (float)chunk_samples / SAMPLE_RATE / num_frames;

        // Apply sigmoid
        for (int64_t frame = 0; frame < num_frames; frame++) {
            for (int64_t spk = 0; spk < num_classes; spk++) {
                segmentation[frame][spk] = sigmoid(segmentation[frame][spk]);
            }
        }

        // Detect segments
        int current_speaker = -1;
        float segment_start = 0.0;
        const float threshold = 0.3;

        for (int64_t frame = 0; frame < num_frames; frame++) {
            float time = frame * frame_duration;

            int max_speaker = 0;
            float max_prob = segmentation[frame][0];

            for (int64_t spk = 1; spk < num_classes; spk++) {
                if (segmentation[frame][spk] > max_prob) {
                    max_prob = segmentation[frame][spk];
                    max_speaker = spk;
                }
            }

            if (max_prob < threshold) max_speaker = -1;

            if (max_speaker != current_speaker) {
                if (current_speaker >= 0 && time - segment_start > 0.5) {
                    float abs_start = (float)offset / SAMPLE_RATE + segment_start;
                    float abs_end = (float)offset / SAMPLE_RATE + time;

                    all_segments[total_segments].start_time = abs_start;
                    all_segments[total_segments].end_time = abs_end;
                    all_segments[total_segments].speaker_id = -1; // Will be assigned by clustering
                    all_segments[total_segments].embedding = NULL;

                    // Compute embedding for this segment
                    size_t seg_start = (size_t)(segment_start * SAMPLE_RATE);
                    size_t seg_end = (size_t)(time * SAMPLE_RATE);
                    if (seg_end > chunk_samples) seg_end = chunk_samples;
                    size_t seg_len = seg_end - seg_start;

                    if (seg_len >= SAMPLE_RATE) {
                        MelSpectrogram* mel = compute_mel_spectrogram(audio + offset + seg_start, seg_len);
                        all_segments[total_segments].embedding = run_embedding(emb_session, mel);

                        for (int64_t j = 0; j < mel->num_frames; j++) free(mel->data[j]);
                        free(mel->data);
                        free(mel);
                    }

                    total_segments++;
                }
                current_speaker = max_speaker;
                segment_start = time;
            }
        }

        // Add final segment of chunk
        if (current_speaker >= 0) {
            float abs_start = (float)offset / SAMPLE_RATE + segment_start;
            float abs_end = (float)(offset + chunk_samples) / SAMPLE_RATE;

            all_segments[total_segments].start_time = abs_start;
            all_segments[total_segments].end_time = abs_end;
            all_segments[total_segments].speaker_id = -1;
            all_segments[total_segments].embedding = NULL;

            size_t seg_start = (size_t)(segment_start * SAMPLE_RATE);
            size_t seg_len = chunk_samples - seg_start;

            if (seg_len >= SAMPLE_RATE) {
                MelSpectrogram* mel = compute_mel_spectrogram(audio + offset + seg_start, seg_len);
                all_segments[total_segments].embedding = run_embedding(emb_session, mel);

                for (int64_t j = 0; j < mel->num_frames; j++) free(mel->data[j]);
                free(mel->data);
                free(mel);
            }

            total_segments++;
        }

        for (int64_t i = 0; i < num_frames; i++) free(segmentation[i]);
        free(segmentation);

        offset += chunk_samples;
        chunk_num++;
    }

    // Global clustering with improved algorithm
    printf("\nClustering %d segments globally...\n", total_segments);
    fprintf(output_file, "\nFound %d speech segments\n", total_segments);
    fprintf(output_file, "Clustering speakers...\n\n");

    // Compute similarity matrix and find average similarities
    fprintf(output_file, "=== CLUSTERING DEBUG ===\n");

    // Count segments with embeddings
    int valid_segments = 0;
    for (int i = 0; i < total_segments; i++) {
        if (all_segments[i].embedding) valid_segments++;
    }

    fprintf(output_file, "Segments with embeddings: %d\n", valid_segments);

    // Compute pairwise similarities
    float max_sim = 0.0, min_sim = 1.0, avg_sim = 0.0;
    int pair_count = 0;

    for (int i = 0; i < total_segments; i++) {
        if (!all_segments[i].embedding) continue;
        for (int j = i + 1; j < total_segments; j++) {
            if (!all_segments[j].embedding) continue;

            float sim = cosine_similarity(all_segments[i].embedding, all_segments[j].embedding);

            if (sim > max_sim) max_sim = sim;
            if (sim < min_sim) min_sim = sim;
            avg_sim += sim;
            pair_count++;

            // Show some high similarity pairs
            if (sim > 0.6) {
                fprintf(output_file, "  Seg %d (%.1fs-%.1fs) <-> Seg %d (%.1fs-%.1fs): similarity=%.3f\n",
                       i, all_segments[i].start_time, all_segments[i].end_time,
                       j, all_segments[j].start_time, all_segments[j].end_time, sim);
            }
        }
    }

    if (pair_count > 0) avg_sim /= pair_count;

    fprintf(output_file, "\nSimilarity statistics:\n");
    fprintf(output_file, "  Min: %.3f, Max: %.3f, Avg: %.3f\n", min_sim, max_sim, avg_sim);

    // Use higher threshold to separate distinct speakers
    // If max similarity is very high (>0.9), there are clear clusters
    float similarity_threshold;
    if (max_sim > 0.9) {
        // Strong clusters exist - use threshold between avg and max
        similarity_threshold = (avg_sim + max_sim) / 2.0;
        if (similarity_threshold > 0.77) similarity_threshold = 0.77;
        if (similarity_threshold < 0.70) similarity_threshold = 0.70;
    } else {
        // Moderate clusters - use avg + margin
        similarity_threshold = avg_sim + 0.05;
        if (similarity_threshold > 0.70) similarity_threshold = 0.70;
        if (similarity_threshold < 0.60) similarity_threshold = 0.60;
    }

    fprintf(output_file, "  Using threshold: %.3f\n\n", similarity_threshold);

    // Agglomerative clustering
    int next_cluster_id = 0;

    for (int i = 0; i < total_segments; i++) {
        if (!all_segments[i].embedding) continue;

        if (all_segments[i].speaker_id == -1) {
            all_segments[i].speaker_id = next_cluster_id++;
        }

        // Find all similar segments and assign to same cluster
        for (int j = i + 1; j < total_segments; j++) {
            if (!all_segments[j].embedding) continue;

            float sim = cosine_similarity(all_segments[i].embedding, all_segments[j].embedding);

            if (sim > similarity_threshold) {
                // If j already has a cluster, merge clusters
                if (all_segments[j].speaker_id != -1) {
                    int old_cluster = all_segments[j].speaker_id;
                    int new_cluster = all_segments[i].speaker_id;

                    // Merge: change all old_cluster to new_cluster
                    for (int k = 0; k < total_segments; k++) {
                        if (all_segments[k].speaker_id == old_cluster) {
                            all_segments[k].speaker_id = new_cluster;
                        }
                    }
                } else {
                    all_segments[j].speaker_id = all_segments[i].speaker_id;
                }
            }
        }
    }

    // Renumber clusters to be consecutive
    int cluster_map[100] = {0};
    int final_cluster_count = 0;

    for (int i = 0; i < total_segments; i++) {
        if (all_segments[i].speaker_id >= 0) {
            int old_id = all_segments[i].speaker_id;
            if (cluster_map[old_id] == 0 && (i == 0 || all_segments[i].speaker_id != all_segments[i-1].speaker_id || i == 0)) {
                // Check if this cluster ID was already mapped
                int already_mapped = 0;
                for (int j = 0; j < i; j++) {
                    if (all_segments[j].speaker_id == old_id) {
                        already_mapped = 1;
                        break;
                    }
                }
                if (!already_mapped) {
                    cluster_map[old_id] = final_cluster_count++;
                }
            }
        }
    }

    for (int i = 0; i < total_segments; i++) {
        if (all_segments[i].speaker_id >= 0) {
            all_segments[i].speaker_id = cluster_map[all_segments[i].speaker_id];
        }
    }

    // Write results
    fprintf(output_file, "=== DIARIZATION RESULTS ===\n\n");
    for (int i = 0; i < total_segments; i++) {
        if (all_segments[i].speaker_id >= 0) {
            fprintf(output_file, "%.1fs - %.1fs: SPEAKER_%d\n",
                   all_segments[i].start_time, all_segments[i].end_time,
                   all_segments[i].speaker_id);
        }
    }

    int num_speakers = 0;
    for (int i = 0; i < total_segments; i++) {
        if (all_segments[i].speaker_id > num_speakers) {
            num_speakers = all_segments[i].speaker_id;
        }
    }
    num_speakers++;

    fprintf(output_file, "\n=== SUMMARY ===\n");
    fprintf(output_file, "Total duration: %.1f seconds\n", (float)num_samples / SAMPLE_RATE);
    fprintf(output_file, "Number of speakers detected: %d\n", num_speakers);
    fprintf(output_file, "Number of speech segments: %d\n", total_segments);

    printf("\n✓ Processing complete!\n");
    printf("Results saved to: poc_output_2.txt\n");
    printf("Speakers detected: %d\n", num_speakers);

    // Cleanup
    for (int i = 0; i < total_segments; i++) {
        if (all_segments[i].embedding) {
            free(all_segments[i].embedding->data);
            free(all_segments[i].embedding);
        }
    }
    free(all_segments);
    free(audio);

    g_ort->ReleaseSession(emb_session);
    g_ort->ReleaseSession(seg_session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);

    fclose(output_file);

    return 0;
}
