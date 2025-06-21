import os
import librosa
import torch
import soundfile as sf
from asteroid.models import ConvTasNet
from tqdm import tqdm

# ディレクトリ設定
input_dir = "/mnt/kiso-qnap3/yuabe/m1/AsReX/data/audio"
output_dir = "/mnt/kiso-qnap3/yuabe/m1/AsReX/data/text"
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
