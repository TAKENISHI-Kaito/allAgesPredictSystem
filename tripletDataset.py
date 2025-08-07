import torch
from torch.utils.data import Dataset
from PIL import Image
import os, random

# PyTorchのDatasetを継承したクラス
class FGNetTripletDataset(Dataset):

    # クラスの初期化（コンストラクタ）
    def __init__(self, df, image_dir, transform, person_ids, min_age_diff=10):

        """
        df：データフレーム（顔画像のメタ情報が入った表形式データ）
        image_dir：画像が保存されているフォルダのパス
        transform：画像の前処理（リサイズ、正規化など）
        persion_ids：対象とする人物IDのリスト
        min_age_diff：トリプレットを作る際に必要な最小の年齢差（デフォルトは10歳）
        """

        # dfの中から、指定された人物IDに該当する行だけを抽出
        self.df = df[df['person_id'].isin(person_ids)]
        # フォルダのパスや変換ルール、年齢差などを保存
        self.image_dir = image_dir
        self.transform = transform
        self.min_age_diff = min_age_diff
        # 各人物IDごとに、その人の画像をまとめて辞書形式で保存
        self.person_to_images = self._group_images_by_person()
        # 登録された人物IDだけをリストとして保存
        self.person_ids = list(self.person_to_images.keys())
        # トリプレット（三つ組：anchor, positive, negative）を全て生成して、保存
        self.triplets = self._generate_triplets()

    def _group_images_by_person(self):
        d = {}
        # self.dfの各行（=各画像情報）をループ処理
        for _, row in self.df.iterrows():
            # person_idをキーにして、画像情報をリストとして辞書dに追加
            d.setdefault(row['person_id'], []).append(row)
        # 最終的に{person_id: 画像情報リスト}の辞書を返す
        return d

    def _generate_triplets(self): # 各人物ごとにトリプレットを作る
        triplets = []
        for pid in self.person_ids:
            # 画像が2枚未満の人はスキップ（トリプレットが作れないため）
            imgs = self.person_to_images[pid]
            if len(imgs) < 2: continue
            # 同じ人物の異なる画像ペアを全通り生成（i≠j）
            for i in range(len(imgs)):
                for j in range(i+1, len(imgs)):
                    # 画像ペアの年齢差が指定より小さい場合はスキップ
                    if abs(imgs[i]['age'] - imgs[j]['age']) < self.min_age_diff:
                        continue
                    # anchor（基準画像）とpositive（同一人物・年齢が違う）をセット
                    anchor, positive = imgs[i], imgs[j]
                    # hardest negative（別人・年齢差が指定以上）を選ぶ
                    for _ in range(5):  # 5回ランダムに選んで、最も近いnegativeを選ぶ
                        neg_pid = random.choice([p for p in self.person_ids if p != pid])
                        neg_imgs = self.person_to_images[neg_pid]
                        for neg in neg_imgs:
                            if abs(anchor['age'] - neg['age']) >= self.min_age_diff:
                                triplets.append((anchor, positive, neg))
                                break
        # すべてのトリプレットをリストとして返す
        return triplets

    # データセットの長さ（トリプレットの数）を返す（DataLoaderで使用）
    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        # 指定されたインデックスのトリプレット（anchor, positive, negative）を取得
        a, p, n = self.triplets[idx]
        # 画像ファイルを開いてRGBに変換し、あらかじめ指定された前処理（リサイズや正規化）を適用する関数
        def load(row):
            img = Image.open(os.path.join(self.image_dir, row['filename'])).convert('RGB')
            return self.transform(img)
        # 前処理されたanchor・positive・negative画像を返す（Tensor形式）
        return load(a), load(p), load(n)