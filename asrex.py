import os
import librosa
import torch
import soundfile as sf
from asteroid.models import ConvTasNet
from tqdm import tqdm

# -----------------------<音源分離>---------------------------
# ディレクトリ設定
input_dir = "/mnt/kiso-qnap3/yuabe/m1/AsReX/data/audio"
output_dir = "/mnt/kiso-qnap3/yuabe/m1/AsReX/data/separated_audio"
os.makedirs(output_dir, exist_ok=True)

# モデル読み込み（16kHz専用）
model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
model.eval()

# .wavファイルの一括処理
for file in tqdm(os.listdir(input_dir)):
    if not file.endswith(".wav"):
        continue

    input_path = os.path.join(input_dir, file)
    output_path = os.path.join(output_dir, file)

    # 音声読み込みとResample（→16kHz）
    y, _ = librosa.load(input_path, sr=16000)  # librosaでリサンプリング済み
    wav_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        separated = model.separate(wav_tensor)  # shape: (1, 2, T)

    stereo_data = separated[0].cpu().numpy().T  # shape: (T, 2)

    # 保存
    sf.write(output_path, stereo_data, samplerate=16000)

print(f"保存先: {output_dir}")

# -----------------------<音声認識>---------------------------
from reazonspeech.nemo.asr import load_model, transcribe, audio_from_path
import json
from dataclasses import asdict
from pathlib import Path

# 入力のルートディレクトリ
root_dir = Path("/mnt/work-qnap/llmc/J-CHAT/audio/youtube_train")
# 出力先ルートディレクトリ
output_root = Path("data/J-CHAT/text/youtube_train")
output_root.mkdir(parents=True, exist_ok=True)

# モデルの読み込み（1回だけ）
model = load_model()

# すべてのシャードを再帰的に探索
for shard_dir in sorted(root_dir.glob("*/cuts.*")):
    for wav_path in sorted(shard_dir.glob("*.wav")):
        print(f"Processing: {wav_path}")

        # 音声読み込みと文字起こし
        audio = audio_from_path(str(wav_path))
        ret = transcribe(model, audio)
        result = asdict(ret)

        # segment の時刻を start/end に変換し、秒数を丸める
        for seg in result.get("segments", []):
            seg["start"] = round(seg.pop("start_seconds"), 3)
            seg["end"] = round(seg.pop("end_seconds"), 3)

        # 出力ディレクトリを元に作成
        rel_path = wav_path.relative_to(root_dir)
        output_dir = output_root / rel_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # ファイル名: xxx.json として保存
        json_path = output_dir / f"{wav_path.stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)



# -----------------------<時間情報の付与>---------------------------
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
J-CHAT podcast_train ─ ReazonSpeech JSON に話者ラベル (A/B) を付与して再保存
"""

import os
import json
from pathlib import Path
from dataclasses import asdict
from tqdm import tqdm

import whisperx
from dotenv import load_dotenv
import torch

# ----------------------------- 事前設定 -----------------------------
load_dotenv()  # HF_TOKEN などが .env に入っていれば読む

# GPU を固定したい場合だけ "cuda:2" などにする
DEVICE = "cuda"  # 例: "cuda:2"
COMPUTE_TYPE = "float16"
HF_TOKEN = os.getenv("HUGGINGFACE_AUTH_TOKEN")

# 入力 (音声・ReazonSpeech JSON) のルート
WAV_ROOT = Path("/mnt/work-qnap/llmc/J-CHAT/audio/podcast_train")
NEMO_JSON_ROOT = Path(
    "/mnt/kiso-qnap3/yuabe/m1/useReazonSpeech/data/J-CHAT/text/podcast_train"
)

# 出力ルート
OUT_ROOT = Path(
    "/mnt/kiso-qnap3/yuabe/m1/moshi-finetune/data/J-CHAT/text/podcast_train"
)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# 話者 ID → ラベル
SPEAKER_MAP = {"SPEAKER_00": "A", "SPEAKER_01": "B"}

# ----------------------------- WhisperX 準備 -----------------------------
print("[INFO] loading WhisperX models …")
asr_model = whisperx.load_model("large-v2", DEVICE, compute_type=COMPUTE_TYPE)
align_model, meta = whisperx.load_align_model("ja", DEVICE)
diarize_pipe = whisperx.diarize.DiarizationPipeline(
    use_auth_token=HF_TOKEN, device=DEVICE
)

# ----------------------------- 走査 & 変換 -----------------------------
failed_list = []

# **/*.wav を再帰的に探索
wav_files = sorted(WAV_ROOT.glob("*/*/*.wav"))  # 00000-of-01432/cuts.000000/*.wav …

for wav_path in tqdm(wav_files, desc="processing"):
    base = wav_path.stem
    # 音声ファイルに対応する ReazonSpeech JSON を探す
    json_path = NEMO_JSON_ROOT / wav_path.relative_to(WAV_ROOT).with_suffix(".json")
    flat_name = "_".join(wav_path.relative_to(WAV_ROOT).with_suffix("").parts) + ".json"
    out_path = OUT_ROOT / flat_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        print(f"[WARN] missing Reazon JSON: {json_path}")
        failed_list.append(wav_path.name)
        continue

    try:
        # ── ① ReazonSpeech JSON を読み込む ───────────────────────────
        with json_path.open(encoding="utf-8") as f:
            rs_json = json.load(f)
        segments = rs_json.get("segments", [])

        # ── ② WhisperX で単語アライメント ───────────────────────
        audio = whisperx.load_audio(str(wav_path))
        aligned = whisperx.align(
            segments, align_model, meta, audio, DEVICE, return_char_alignments=False
        )

        # ── ③ 2話者ダイアリゼーション ─────────────────────────
        dia = diarize_pipe(str(wav_path), num_speakers=2)

        # 2話者検出できなければスキップ
        if dia["speaker"].nunique() != 2:
            print(f"[SKIP] {wav_path.name}: detected speaker count ≠ 2")
            failed_list.append(wav_path.name)
            continue

        # ── ④ 単語に話者を割当て ─────────────────────────────
        word_spk = whisperx.assign_word_speakers(dia, aligned)
        if not word_spk or not word_spk.get("segments"):
            print(f"[SKIP] empty assignment: {wav_path.name}")
            failed_list.append(wav_path.name)
            continue

        # ── ⑤ 整形して保存 ───────────────────────────────────
        out_items = []
        for seg in word_spk["segments"]:
            backup = seg.get("speaker", "Unknown")
            for w in seg["words"]:
                spk_lbl = SPEAKER_MAP.get(w.get("speaker", backup), "Unknown")
                out_items.append(
                    {
                        "speaker": spk_lbl,
                        "word": w["word"],
                        "start": round(w["start"], 3),
                        "end": round(w["end"], 3),
                    }
                )

        # ファイル内に A/B がそろわなければ失敗扱い
        if {i["speaker"] for i in out_items} != {"A", "B"}:
            print(
                f"[SKIP] only {set(i['speaker'] for i in out_items)} in {wav_path.name}"
            )
            failed_list.append(wav_path.name)
            continue

        # pretty-print で保存（1行 JSON を並べた配列）
        with out_path.open("w", encoding="utf-8") as fo:
            json.dump(out_items, fo, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"[ERROR] {wav_path.name}: {e}")
        failed_list.append(wav_path.name)

# ----------------------------- ログ -----------------------------
logfile = Path("failed_speaker_assignment_podcast_train.txt")
with logfile.open("w", encoding="utf-8") as log:
    for fn in failed_list:
        log.write(fn + "\n")

print(f"[INFO] completed. {len(failed_list)} files skipped or failed → {logfile}")
