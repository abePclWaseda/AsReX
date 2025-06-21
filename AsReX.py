#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2-speaker separation ➜ ASR ➜ WhisperX alignment
"""

# ========== 共通ライブラリ ==========
import os, json, shutil
from pathlib import Path
from tqdm import tqdm
import torch, soundfile as sf, librosa
from dotenv import load_dotenv

# ========== ① 分離 ==========
from asteroid.models import ConvTasNet

IN_DIR = Path("/mnt/kiso-qnap3/yuabe/m1/AsReX/data/audio")
SEP_DIR = Path(
    "/mnt/kiso-qnap3/yuabe/m1/AsReX/data/separated"
)  # spkA / spkB の 2 階層に出力
SEP_DIR.mkdir(parents=True, exist_ok=True)

print("[1/3] loading ConvTasNet …")
sep_model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
sep_model.eval()

for wav in tqdm(sorted(IN_DIR.glob("*.wav")), desc="separating"):
    y, _ = librosa.load(wav, sr=16_000)  # 16 kHz モノラル
    est = sep_model.separate(torch.tensor(y).unsqueeze(0))  # (1, 2, T)
    for idx, spk in enumerate(("spkA", "spkB")):
        out_dir = SEP_DIR / spk
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{wav.stem}_{spk}.wav"
        sf.write(out_path, est[0, idx].cpu().numpy(), 16_000)

print(f"[1/3] separated wav ➜ {SEP_DIR}")

# ========== ② ReazonSpeech-NeMo 文字起こし ==========
from reazonspeech.nemo.asr import load_model, transcribe, audio_from_path

TXT_DIR = Path("/mnt/kiso-qnap3/yuabe/m1/AsReX/data/transcripts")
TXT_DIR.mkdir(parents=True, exist_ok=True)

print("[2/3] loading ReazonSpeech-NeMo …")
asr_model = load_model()


def nemo_asr(wav_path: Path) -> dict:
    audio = audio_from_path(str(wav_path))
    res = transcribe(asr_model, audio).to_dict()
    for seg in res.get("segments", []):
        seg["start"] = round(seg.pop("start_seconds"), 3)
        seg["end"] = round(seg.pop("end_seconds"), 3)
    return res


for wav in tqdm(sorted(SEP_DIR.glob("*/*.wav")), desc="ASR"):
    txt_path = TXT_DIR / wav.with_suffix(".json").name
    if txt_path.exists():  # 既に推論済みならスキップ
        continue
    json.dump(nemo_asr(wav), txt_path.open("w"), ensure_ascii=False, indent=2)

print(f"[2/3] transcripts ➜ {TXT_DIR}")

# ========== ③ WhisperX 単語アライン ==========
import whisperx

load_dotenv()
DEVICE = "cuda"  # または "cuda:1" など
COMPUTE_TYPE = "float16"
HF_TOKEN = os.getenv("HUGGINGFACE_AUTH_TOKEN")

print("[3/3] loading WhisperX …")
w_model = whisperx.load_model("large-v2", DEVICE, compute_type=COMPUTE_TYPE)
align_model, meta = whisperx.load_align_model("ja", DEVICE)

ALN_DIR = Path("/mnt/kiso-qnap3/yuabe/m1/AsReX/data/aligned")
ALN_DIR.mkdir(parents=True, exist_ok=True)

SPEAKER_MAP = {"spkA": "A", "spkB": "B"}  # ラベルはご自由に

for wav in tqdm(sorted(SEP_DIR.glob("*/*.wav")), desc="aligning"):
    base = wav.stem  # foo_spkA
    spk = wav.parent.name  # spkA / spkB
    txt = TXT_DIR / f"{base}.json"
    out = ALN_DIR / f"{base}.json"
    if out.exists():  # 既に整形済みならスキップ
        continue

    segs = json.load(txt.open())["segments"]

    # WhisperX align
    audio = whisperx.load_audio(str(wav))
    aligned = whisperx.align(
        segs, align_model, meta, audio, DEVICE, return_char_alignments=False
    )

    # 単語単位に整形
    words_out = []
    for seg in aligned["segments"]:
        for w in seg["words"]:
            words_out.append(
                {
                    "speaker": SPEAKER_MAP[spk],
                    "word": w["word"],
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                }
            )

    json.dump(words_out, out.open("w"), ensure_ascii=False, indent=2)

print(f"[3/3] aligned words ➜ {ALN_DIR}")
