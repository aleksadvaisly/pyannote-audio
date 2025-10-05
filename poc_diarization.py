import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import os
import json

# Get HF token from environment or prompt user
hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
if not hf_token:
    print("HuggingFace token nie znaleziony w zmiennych środowiskowych.")
    print("Ustaw zmienną HF_TOKEN lub HUGGINGFACE_TOKEN, lub podaj token tutaj:")
    hf_token = input("Token: ").strip()

# Load pipeline
print("Ładowanie modelu...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=hf_token
)

# Send to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Używam urządzenia: {device}')
pipeline.to(device)

# Process audio
audio_file = '/Users/aleksander/Downloads/Nagrania 03102025/New Recording 43.m4a'
print(f'Przetwarzam: {audio_file}')

with ProgressHook() as hook:
    output = pipeline(audio_file, hook=hook)

# Print results
print('\n=== WYNIKI DIARYZACJI MÓWCÓW ===')
results = []
for turn, speaker in output.speaker_diarization:
    result = {
        'start': round(turn.start, 1),
        'stop': round(turn.end, 1),
        'speaker': speaker
    }
    results.append(result)
    print(f'start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}')

num_speakers = len(set(speaker for _, speaker in output.speaker_diarization))
print(f'\nZnaleziono {num_speakers} różnych mówców')

# Save to JSON
output_data = {
    'audio_file': audio_file,
    'num_speakers': num_speakers,
    'segments': results
}

output_file = 'poc_output.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f'\nWyniki zapisane do: {output_file}')
