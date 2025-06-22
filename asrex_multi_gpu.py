#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2-speaker separation ➜ ReazonSpeech-NeMo ASR ➜ WhisperX alignment
GPU = 2 枚並列版
"""

import os, json, tempfile, traceback, multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch, soundfile as sf, librosa
from dotenv import load_dotenv

# ===================== パス設定 =====================
IN_ROOT = Path("/mnt/work-qnap/llmc/J-CHAT/audio/podcast_valid")
SEP_DIR = Path("/mnt/kiso-qnap3/yuabe/m1/AsReX/data/J-CHAT/audio/podcast_valid")
TXT_DIR = Path("/mnt/kiso-qnap3/yuabe/m1/AsReX/data/J-CHAT/transcripts/podcast_valid")
ALN_DIR = Path("/mnt/kiso-qnap3/yuabe/m1/AsReX/data/J-CHAT/text/podcast_valid")
for p in (SEP_DIR, TXT_DIR, ALN_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ===== 共通定数 =====
SPEAKER_LIST = ("A", "B")  # ch=0 → A, ch=1 → B
COMPUTE_TYPE = "float16"
HF_TOKEN = os.getenv("HUGGINGFACE_AUTH_TOKEN")


# ---------------------------------------------------
# ★ ワーカー本体：1GPU で 1/2 のファイルを処理する ★
# ---------------------------------------------------
def worker(device: str, wav_paths: list[Path]) -> None:
    """フルパイプラインを 1GPU で処理"""
    torch.cuda.set_device(device)

    # ① ConvTasNet
    from asteroid.models import ConvTasNet

    print(f"[GPU {device}] loading ConvTasNet …")
    sep_model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
    sep_model.to(device).eval()

    # ② ReazonSpeech-NeMo
    from reazonspeech.nemo.asr import load_model, transcribe, audio_from_numpy

    print(f"[GPU {device}] loading ReazonSpeech-NeMo …")
    asr_model = load_model(device=device)

    def nemo_asr_numpy(audio_np: np.ndarray, sr: int = 16_000) -> dict:
        ret = transcribe(asr_model, audio_from_numpy(audio_np, sr))
        segs = [
            {
                "start": round(s.start_seconds, 3),
                "end": round(s.end_seconds, 3),
                "text": s.text,
            }
            for s in ret.segments
        ]
        return {"text": ret.text, "segments": segs}

    # ③ WhisperX
    import whisperx

    print(f"[GPU {device}] loading WhisperX align model …")
    align_model, meta = whisperx.load_align_model("ja", device)

    log_path = f"align_errors_{device}.log"
    with open(log_path, "a") as LOG_FILE:

        def log(msg: str):
            print(msg)
            LOG_FILE.write(msg + "\n")
            LOG_FILE.flush()

        # ===== ループ =====
        for wav in tqdm(wav_paths, desc=f"[GPU {device}] processing"):
            try:
                # ---------- ① separation ----------
                y_mono, _ = librosa.load(wav, sr=16_000)
                est = sep_model.separate(torch.tensor(y_mono).unsqueeze(0))
                stereo = est[0].cpu().numpy().T  # (T, 2)
                sep_path = SEP_DIR / f"{wav.stem}.wav"
                sf.write(sep_path, stereo, 16_000)

                # ---------- ② ASR ----------
                for ch, spk in enumerate(SPEAKER_LIST):
                    txt_path = TXT_DIR / f"{wav.stem}_{spk}.json"
                    if not txt_path.exists():
                        res = nemo_asr_numpy(stereo[:, ch])
                        json.dump(res, txt_path.open("w"), ensure_ascii=False, indent=2)

                # ---------- ③ align ----------
                y, sr = sf.read(sep_path, dtype="float32")  # (T, 2)
                merged = []
                for ch, spk in enumerate(SPEAKER_LIST):
                    txt_path = TXT_DIR / f"{wav.stem}_{spk}.json"
                    segs = json.load(txt_path.open())["segments"]
                    aligned = whisperx.align(
                        segs,
                        align_model,
                        meta,
                        y[:, ch],
                        device,
                        return_char_alignments=False,
                    )
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
                merged.sort(key=lambda x: x["start"])
                out_path = ALN_DIR / f"{wav.stem}.json"
                json.dump(merged, out_path.open("w"), ensure_ascii=False, indent=2)

            except Exception:
                log(f"\n!! ERROR while processing {wav.name} !!")
                traceback.print_exc(file=LOG_FILE)
                continue


# ===================== エントリポイント =====================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # CUDA 周りの既知バグ回避
    GPU_LIST = ["cuda:0", "cuda:1"]  # ★ ここで割り当て GPU を指定
    wav_all = sorted(IN_ROOT.rglob("*.wav"))
    # ほぼ同じ長さになるよう 2 分割
    chunks = [wav_all[i :: len(GPU_LIST)] for i in range(len(GPU_LIST))]

    procs = []
    for dev, paths in zip(GPU_LIST, chunks):
        p = mp.Process(target=worker, args=(dev, paths), daemon=False)
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("=== all done! ===")
