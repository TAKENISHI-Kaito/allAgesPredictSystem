import os
import csv
import re

# ディレクトリのパス
image_dir = "images"
output_csv = "annotation.csv"

data = []

# JPEGファイルを走査
for filename in os.listdir(image_dir):
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
        name_no_ext = os.path.splitext(filename)[0]  # 例: 066A06b → 066A06b

        # person_id: 先頭3文字 → ゼロ除去
        person_id_str = name_no_ext[:3]
        try:
            person_id = int(person_id_str)
        except ValueError:
            continue  # 不正なperson_idはスキップ

        # ageの抽出（末尾に英字がある場合、それを除外して末尾2桁を取得）
        digits = re.findall(r'\d+', name_no_ext)
        if digits:
            age_candidate = digits[-1]  # 最後の数字列
            try:
                age = int(age_candidate)
            except ValueError:
                continue
        else:
            continue

        # データを追加
        data.append([filename, person_id, age])

# person_idとageでソート
data.sort(key=lambda x: (x[1], x[2]))

# CSVファイルを書き出す
with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "person_id", "age"])  # ヘッダー
    writer.writerows(data)

print(f"{output_csv} を作成しました（{len(data)}件）。")