import os

# 探索するルートディレクトリ
search_root = "/mnt/work-qnap/llmc/J-CHAT/audio/youtube_other"
target_filename = "8e72daced1c35edc565904d9526be348.wav"

# 再帰的にファイル探索
for root, dirs, files in os.walk(search_root):
    if target_filename in files:
        full_path = os.path.join(root, target_filename)
        print(full_path)
        break  # 最初に見つかったら終了したい場合
