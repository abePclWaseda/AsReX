import whisperx
from datetime import timedelta
from pathlib import Path
from tqdm import tqdm
import whisperx.diarize
from dotenv import load_dotenv
import os
import json

load_dotenv()

device = "cuda"
audio_file = "data/audio/15f97caafef4a4b352e54781236cf2b4.wav"
batch_size = 16
compute_type = "float16"
auth_token = os.getenv("HUGGINGFACE_AUTH_TOKEN")

model = whisperx.load_model("large-v2", device, compute_type=compute_type)
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)

model_a, metadata = whisperx.load_align_model(
    language_code=result["language"], device=device
)

result = whisperx.align(
    result["segments"], model_a, metadata, audio, device, return_char_alignments=False
)

diarize_model = whisperx.diarize.DiarizationPipeline(
    use_auth_token=auth_token, device=device
)
diarize_segments = diarize_model(audio_file)
result = whisperx.assign_word_speakers(diarize_segments, result)

output_path = Path("data/text/15f97caafef4a4b352e54781236cf2b4_whisperx.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as jf:
    json.dump(result["segments"], jf, ensure_ascii=False, indent=2)
