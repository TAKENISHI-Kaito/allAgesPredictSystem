# ライブラリ
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import os, json, random
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms.functional import to_tensor
import cv2

# Triplet Learning用のモデル
class TripletNetPL(pl.LightningModule): # PyTorch Lightning の基本クラス

    def __init__(self, embedding_dim=128, lr=1e-4): # 出力する特徴ベクトルの次元を128に指定
        # 親クラスの初期化処理
        super().__init__()
        # 渡されたembedding_dimなどの引数を自動的に保存してくれる便利機能（学習ログなどに使える）
        self.save_hyperparameters()
        # ResNet50という画像認識モデル（事前学習済み）を読み込み、ImageNetで学習済みの重みを使用
        base = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # ResNet50の最後にある「全結合層（分類用）」を消して、その代わりに「何もしない層（Identity）」に置き換え
        base.fc = nn.Identity()
        # 変更後のResNet50（最後の分類層を削除した状態）をこのモデルの一部として保存
        self.base = base
        # ResNet50の出力（2048次元）を、指定したembedding_dim（例：128次元）に変換する全結合層
        self.fc = nn.Linear(2048, embedding_dim)
        # Triplet Lossを定義
        # anchor, positive, negativeという3枚の画像の距離関係を学習。
        # margin=1.0は「anchorとnegativeは最低でもこれくらい離れててほしい」という制約
        self.criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(self, x): # モデルが画像を受け取って出力を返すための「前向き処理（forward）」関数
        # 入力画像xをResNet50本体に通し、画像の高次元特徴（2048次元）を取得
        x = self.base(x)
        # その特徴を、128次元などの埋め込みベクトルに変換
        x = self.fc(x)
        # 出力されたベクトルをL2ノルム（長さ）で正規化（ベクトルの長さを1にする）
        # これにより、「ベクトルの向き」だけで類似度を計算できるようになる（コサイン類似度などに便利）
        return F.normalize(x, p=2, dim=1)

    def training_step(self, batch, batch_idx): # 学習ループで1バッチごとに呼び出される関数
        # Tripletの画像を取り出します
        a, p, n = batch
        # それぞれの画像をモデルに通して、埋め込みベクトル（128次元など）を取得
        emb_a = self(a)
        emb_p = self(p)
        emb_n = self(n)
        # Triplet Lossを計算
        # Anchorに対して、Positiveは近く、Negativeは遠くなるように学習
        loss = self.criterion(emb_a, emb_p, emb_n)
        # 学習損失をログに記録（TensorBoardなどで確認できる）
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        # PyTorch Lightningに損失を返して学習を進めてもらう
        return loss

    def configure_optimizers(self): # 学習時に使う最適化手法とスケジューラを設定
        # Adamというよく使われる最適化手法を使用
        # self.hparams.lr から学習率を取得
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # 学習率スケジューラを設定
        # 10エポックごとに学習率を半分に減らす設定
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        # PyTorch Lightningが学習時にこの設定を使うように指定
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def validation_step(self, batch, batch_idx):
        a, p, n = batch
        emb_a = self(a)
        emb_p = self(p)
        emb_n = self(n)
        loss = self.criterion(emb_a, emb_p, emb_n)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

# PyTorchのDatasetを継承したクラス
class FGNetTripletDataset(Dataset):

    # クラスの初期化（コンストラクタ）
    def __init__(self, df, image_dir, transform, person_ids, min_age_diff=10):

        # df：データフレーム（顔画像のメタ情報が入った表形式データ）
        # image_dir：画像が保存されているフォルダのパス
        # transform：画像の前処理（リサイズ、正規化など）
        # persion_ids：対象とする人物IDのリスト
        # min_age_diff：トリプレットを作る際に必要な最小の年齢差（デフォルトは10歳）

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

# 画像に対する前処理（=データ変換）をまとめて定義
transform = transforms.Compose([
    # ランダムな位置から画像を切り出してリサイズ（サイズ：224×224）する → 位置・スケールのバリエーションを増やす
    transforms.RandomResizedCrop(224),
    # 画像をランダムで左右反転 → 左右対称性を学習
    transforms.RandomHorizontalFlip(),
    # 明るさ・コントラスト・彩度を少し変更 → 色の違いに強いモデルにする
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # 回転や少しの平行移動を加える → 位置ズレに強くする
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    # PIL画像（もしくはNumPy）をPyTorchのテンソル形式に変換
    transforms.ToTensor()
])

# 顔画像のファイル名、人物ID、年齢などの情報が入ったCSVファイルを読み込む（dfはDataFrame形式）
df = pd.read_csv('annotations.csv')
# 登場するすべての人物ID（重複なし）を取得し、ソート
person_ids = sorted(df['person_id'].unique())
# ランダムにシャッフルして、訓練と検証用に分割できるようにする
# 再現性のため、seed=42とすることで実行するたびに結果が変わらないようにする
random.seed(42)
random.shuffle(person_ids)
# データの8割を訓練用（train）、2割を検証用（val）に分割
# set()にすることで、後で検索しやすくする
split = int(0.8 * len(person_ids))
train_ids = set(person_ids[:split])
val_ids = set(person_ids[split:])

# FGNetTripletDataset: 顔画像から (anchor, positive, negative) の3つの画像ペアを作る特別なDatasetクラス
train_dataset = FGNetTripletDataset(df, "images", transform, train_ids)
val_dataset = FGNetTripletDataset(df, "images", transform, val_ids)
# DataLoader: ミニバッチでデータを読み込む。batch_size=32で32ペアずつ読み込み、shuffle=Trueで毎エポック順番をシャッフル。
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

def evaluate_model(model, df, val_ids):

    # model: 学習済みの顔認識モデル
    # df: データフレーム（画像情報・年齢・person_idなどを持つ）
    # val_ids: 検証に使う人物IDのリスト

    # 評価モードに切り替える
    # DropoutやBatchNormといった訓練時特有の動作を無効化（評価の安定性確保のため）
    model.eval()
    # 評価用データ（検証セット）だけを取り出す
    # pairs：評価に使う画像ペア、labels: 同一人物なら1、異なるなら0。
    pairs, labels = [], []
    val_df = df[df['person_id'].isin(val_ids)]
    # 各人物IDについて、same: 同一人物の画像、other: 異なる人物の画像
    for pid in val_ids:
        same = val_df[val_df['person_id'] == pid]
        other = val_df[val_df['person_id'] != pid]
        # 同一人物の中から、年齢差が10歳以上あるペアを探す
        for i, r1 in same.iterrows():
            for j, r2 in same.iterrows():
                if i >= j or abs(r1['age'] - r2['age']) < 10:
                    continue
                # 同一人物かつ年齢差10歳以上のペアを追加。label=1（同一人物）。
                pairs.append((r1['filename'], r2['filename']))
                labels.append(1)
            # 異なる人物からランダムに2枚選び、年齢差10歳以上であればペアとして追加。label=0（別人）。
            sampled = other.sample(n=2)
            for _, r2 in sampled.iterrows():
                if abs(r1['age'] - r2['age']) < 10: continue
                pairs.append((r1['filename'], r2['filename']))
                labels.append(0)

    # 評価用の画像変換処理。学習時と同じく224×224にリサイズし、テンソルに変換。
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 指定されたファイル名から画像を読み込み、モデルを通して特徴ベクトル（埋め込み）を得る関数
    def get_emb(fname):
        img = Image.open(os.path.join("images", fname)).convert("RGB")
        img = transform_eval(img).unsqueeze(0).to(model.device)
        # 評価時は勾配計算を省略してメモリを節約
        with torch.no_grad():
            return model(img)

    # 評価結果（距離・予測など）を記録するリストを初期化
    results = []
    # 各画像ペアとその正解ラベルをループ
    for (f1, f2), label in zip(pairs, labels):
        # それぞれの画像をモデルに通して埋め込みベクトル（顔の特徴量）を取得
        emb1 = get_emb(f1)
        emb2 = get_emb(f2)
        # 2つのベクトル間の距離（L2ノルム）を計算
        dist = F.pairwise_distance(emb1, emb2).item()
        # 類似度スコアを1 - distとして計算します（距離が小さいほど似ているとみなす）
        score = 1 - dist
        # 類似度が0.5以上なら「同一人物」と判定（予測ラベル1）、そうでなければ0
        pred = int(score >= 0.5)
        # 評価結果（画像ペア、正解・予測ラベル、スコア、距離）を保存
        results.append({
            "filename1": f1,
            "filename2": f2,
            "label": label,
            "pred": pred,
            "score": score,
            "dist": dist
        })

    # 正解ラベルと予測ラベルを比較してAccuracy（正答率）を計算
    acc = accuracy_score([r['label'] for r in results], [r['pred'] for r in results])
    # 類似度スコアを使って、AUC（Area Under the Curve）を計算
    # AUCはモデルが「同一人物」と「別人」をうまく区別できているかを評価する指標
    auc = roc_auc_score([r['label'] for r in results], [r['score'] for r in results])

    # 正しく分類できた結果（correct）と、間違えた結果（incorrect）を分けて保存
    correct = [r for r in results if r["label"] == r["pred"]]
    incorrect = [r for r in results if r["label"] != r["pred"]]

    # 精度（Accuracy）、AUC、正答例、誤答例を返す
    return acc, auc, correct, incorrect

# TripletNetPL（PyTorch Lightningで定義されたTriplet Lossを使うネットワーク・モデル）をインスタンス化
# 学習対象のモデルが準備
model = TripletNetPL()
# 学習過程（損失や精度など）を記録して、TensorBoardで可視化できるようにするロガーを作成
# ログはGoogle Drive内の /all-age_predict_system/fgnet_logs に保存
logger = TensorBoardLogger("tensorboard", name="fgnet_logs")
# モデルの重み（学習結果）を保存する設定
# 学習中のtrain_loss（訓練誤差）を監視し、最も小さいときのモデルだけを保存（save_top_k=1）
checkpoint_callback = ModelCheckpoint(monitor="train_loss", save_top_k=1, mode="min")
# 学習が進まなくなったときに、自動で止めるための設定（早期終了）
# train_lossが5エポック連続で改善しない場合、学習をストップ
# early_stop_callback = EarlyStopping(monitor="train_loss", patience=5, mode="min")

# モデルをトレーニング（学習）するためのtrainerを定義
trainer = pl.Trainer(
    max_epochs=50, # 最大50エポックまで学習を行う
    logger=logger,
    accelerator='auto', # GPU/CPUのどちらを使うか自動で判断
    callbacks=[checkpoint_callback] # ロガーやチェックポイントなどのコールバック機能を組み込む
)
# 上で作ったmodelを、指定したデータローダー（train_loaderとval_loader）で学習させる
trainer.fit(model, train_loader, val_loader)

# 最も性能の良かった（train_lossが最小だった）モデルの保存パスを取得
best_model_path = checkpoint_callback.best_model_path
# 上で保存された「ベストなモデル」の重みを読み込んで、評価に使う準備
model = TripletNetPL.load_from_checkpoint(best_model_path)
# evaluate_model関数を使ってモデルを評価
acc, auc, correct, incorrect = evaluate_model(model, df, val_ids)
print(f"Final Val Accuracy: {acc:.4f}, AUC: {auc:.4f}")
print(f"Correct: {len(correct)} | Incorrect: {len(incorrect)}")

# 正しく分類されたサンプルがあるなら、その最初の1つを取り出してexampleに代入
example = correct[0] if correct else None
print("Sample Case:")
print(example)

# 誤って分類されたサンプルがあるなら、その1つを表示
example = incorrect[0] if incorrect else None
print("Sample Case:")
print(example)

# 可視化対象画像の選択（anchor）
img_name = example["filename1"]
img_path = os.path.join("images", img_name)

# 画像の読み込み
# .convert('RGB')：色をRGB形式に統一
# .resize((224, 224))：ResNetが受け取れるサイズに変換（入力は224×224ピクセル）
original_img = Image.open(img_path).convert('RGB').resize((224, 224))
# 画像をPyTorch形式のテンソルに変換
# to_tensor は transforms.ToTensor() と同じような変換
# .unsqueeze(0)：バッチ次元を追加（モデルがバッチ前提で受け取るため）
# .to(model.device)：CPUまたはGPUに転送
input_tensor = to_tensor(original_img).unsqueeze(0).to(model.device)
# PIL形式の画像をNumPy配列に変換（np.array）
# 値の範囲を[0, 1]に正規化（PyTorchの可視化ライブラリがこの形式を要求）
rgb_img = np.array(original_img).astype(np.float32) / 255.0

# Grad-CAMは「どの層を見るか」を指定する必要がある
# ResNetの最後の畳み込み層（layer4 の最後のブロック）を対象に設定
target_layers = [model.base.layer4[-1]]
# GradCAMオブジェクトを作成
# 使用するモデルと対象レイヤーを渡す
cam = GradCAM(model=model.base, target_layers=target_layers)

# Grad-CAMマップ（モデルが注目した箇所の熱マップ）を計算
# バッチの先頭画像（[0]）のみを使用
grayscale_cam = cam(input_tensor=input_tensor)[0]
# 熱マップ（グレースケール）を元の画像（RGB）に重ねて視覚的に見えるように合成
# use_rgb=Trueにすると、色がきれいに表示
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# 可視化
plt.figure(figsize=(6, 6))
plt.imshow(visualization)
plt.title(f"Grad-CAM on {img_name}")
plt.axis("off")
plt.show()