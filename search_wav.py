import os

# 探索するルートディレクトリ
search_root = "/mnt/work-qnap/llmc/J-CHAT/audio/youtube_other"
target_filename = "9ece1b5c6aa0ca6e160cbbe6223f09d2.wav"

# 再帰的にファイル探索
for root, dirs, files in os.walk(search_root):
    if target_filename in files:
        full_path = os.path.join(root, target_filename)
        print(full_path)
        break  # 最初に見つかったら終了したい場合
