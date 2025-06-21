#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2-speaker separation ➜ ReazonSpeech-NeMo ASR ➜ WhisperX alignment
最終出力は A/B が混在した 1 本の JSON。
"""

# -------------------- 共通ライブラリ --------------------
import os, json, tempfile
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch, soundfile as sf, librosa
from dotenv import load_dotenv

# -------------------- パス設定 --------------------
IN_DIR = Path("/mnt/kiso-qnap3/yuabe/m1/AsReX/data/audio")  # 元音声 (モノラル)
SEP_DIR = Path("/mnt/kiso-qnap3/yuabe/m1/AsReX/data/separated")  # 2 ch WAV を置く
TXT_DIR = Path("/mnt/kiso-qnap3/yuabe/m1/AsReX/data/transcripts")  # ASR 出力 (中間)
ALN_DIR = Path("/mnt/kiso-qnap3/yuabe/m1/AsReX/data/text")  # 最終 JSON

for p in (SEP_DIR, TXT_DIR, ALN_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ======================================================
# ① ConvTasNet で 2 ch 分離し、1 本のステレオ WAV を保存
# ======================================================
from asteroid.models import ConvTasNet

print("[1/3] loading ConvTasNet …")
sep_model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
sep_model.eval()

for wav in tqdm(sorted(IN_DIR.glob("*.wav")), desc="separating"):
    y, _ = librosa.load(wav, sr=16_000)  # 16 kHz モノラル
    est = sep_model.separate(torch.tensor(y).unsqueeze(0))  # (1, 2, T)
    stereo = est[0].cpu().numpy().T  # (T, 2)

    out_path = SEP_DIR / f"{wav.stem}.wav"  # ★ 2 ch で 1 ファイル
    sf.write(out_path, stereo, 16_000)

print(f"[1/3] separated (2 ch) ➜ {SEP_DIR}")

# ======================================================
# ② ReazonSpeech-NeMo 文字起こし（チャンネル毎）
# ======================================================
from reazonspeech.nemo.asr import load_model, transcribe, audio_from_path

print("[2/3] loading ReazonSpeech-NeMo …")
asr_model = load_model()


def nemo_asr_numpy(audio_np: np.ndarray, sr: int = 16_000) -> dict:
    """
    numpy 配列 → 一時 WAV → NeMo ASR → dict
    （audio_from_array が無い環境向けの簡易実装）
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, audio_np, sr)
        res = transcribe(asr_model, audio_from_path(tmp.name)).to_dict()
    # 秒数フィールドを整形
    for seg in res.get("segments", []):
        seg["start"] = round(seg.pop("start_seconds"), 3)
        seg["end"] = round(seg.pop("end_seconds"), 3)
    return res


SPEAKER_LIST = ("A", "B")  # ch=0 → A, ch=1 → B

for wav in tqdm(sorted(SEP_DIR.glob("*.wav")), desc="ASR"):
    y, sr = sf.read(wav)  # y.shape == (T, 2)

    for ch, spk in enumerate(SPEAKER_LIST):
        txt_path = TXT_DIR / f"{wav.stem}_{spk}.json"
        if txt_path.exists():
            continue
        res = nemo_asr_numpy(y[:, ch], sr)
        json.dump(res, txt_path.open("w"), ensure_ascii=False, indent=2)

print(f"[2/3] transcripts ➜ {TXT_DIR}")

# ======================================================
# ③ WhisperX で単語アラインし、A/B を時間順に結合
# ======================================================
import whisperx

load_dotenv()
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
HF_TOKEN = os.getenv("HUGGINGFACE_AUTH_TOKEN")

print("[3/3] loading WhisperX …")
w_model = whisperx.load_model("large-v2", DEVICE, compute_type=COMPUTE_TYPE)
align_model, meta = whisperx.load_align_model("ja", DEVICE)

for wav in tqdm(sorted(SEP_DIR.glob("*.wav")), desc="aligning"):
    y, sr = sf.read(wav)  # (T, 2)
    merged = []

    for ch, spk in enumerate(SPEAKER_LIST):
        txt_path = TXT_DIR / f"{wav.stem}_{spk}.json"
        segs = json.load(txt_path.open())["segments"]

        # WhisperX align
        aligned = whisperx.align(
            segs,
            align_model,
            meta,
            y[:, ch],  # numpy 1-ch
            DEVICE,
            return_char_alignments=False,
        )

        # 単語を取り出して merged へ追加
        merged.extend(
            {
                "speaker": spk,
                "word": w["word"],
                "start": round(w["start"], 3),
                "end": round(w["end"], 3),
            }
            for seg in aligned["segments"]
            for w in seg["words"]
        )

    # ★ start 時刻でソートして A/B 混在の 1 本に
    merged.sort(key=lambda x: x["start"])

    out_path = ALN_DIR / f"{wav.stem}.json"
    json.dump(merged, out_path.open("w"), ensure_ascii=False, indent=2)

print(f"[3/3] aligned words (A+B) ➜ {ALN_DIR}")
