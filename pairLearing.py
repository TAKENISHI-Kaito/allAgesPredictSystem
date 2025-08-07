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

# Siames Network用のモデル
class SiameseNetPL(pl.LightningModule): # Pytorch Lightning の基本クラス

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
        # 2つの画像の特徴を結合した後、それを使って「同一人物かどうか」を分類するニューラルネットワーク
        self.classifier = nn.Sequential(
            # 256次元（128×2）→128次元に変換
            nn.Linear(2 * embedding_dim, 128),
            # 活性化関数ReLUで非線形性を加える
            nn.ReLU(),
            # 最後に1つの値（スコア）を出力
            nn.Linear(128, 1)
        )
        # 2クラス分類用の損失関数（Binary Cross Entropy）
        # 出力は「ロジット」（確率に変換する前の値）であることを前提
        self.criterion = nn.BCEWithLogitsLoss()

    def forward_once(self, x): # 1枚の画像だけを使って、特徴ベクトル（埋め込み）を作る関数
        # 入力画像xをResNet50本体に通し、画像の高次元特徴（2048次元）を取得
        x = self.base(x)
        # その特徴を、128次元などの埋め込みベクトルに変換
        x = self.fc(x)
        # 出力されたベクトルをL2ノルム（長さ）で正規化（ベクトルの長さを1にする）
        # これにより、「ベクトルの向き」だけで類似度を計算できるようになる（コサイン類似度などに便利）
        return F.normalize(x, p=2, dim=1)

    def forward(self, x1, x2): # モデルが画像を受け取って出力を返すための「前向き処理（forward）」関数
        # それぞれの画像から特徴ベクトルを取り出す
        emb1 = self.forward_once(x1)
        emb2 = self.forward_once(x2)
        # 2つの特徴ベクトルをつなげて1つにする（横に連結、次元は256）
        combined = torch.cat([emb1, emb2], dim=1)
        # classifierに渡して、1つのスコア（実数）を出力
        # .squeeze(1) は余分な次元（batch_size × 1 → batch_size）を取り除くための処理
        return self.classifier(combined).squeeze(1)

    def training_step(self, batch, batch_idx): # 学習ループで1バッチごとに呼び出される関数
        # ミニバッチの中身（画像1・画像2・正解ラベル）を取り出す
        x1, x2, label = batch
        # モデルに2枚の画像を渡してスコア（ロジット）を取得
        logits = self(x1, x2)
        # 損失関数で「予測」と「正解ラベル」を比べて誤差（loss）を計算
        # ラベルをfloatに変換するのは、BCEWithLogitsLossがfloat を要求するため
        loss = self.criterion(logits, label.float())
        # ロジットにシグモイド関数をかけて「確率」に変換し、0.5以上なら1、未満なら0と判断
        preds = (torch.sigmoid(logits) >= 0.5).int()
        # 正解ラベルと一致した割合（Accuracy）を計算
        acc = (preds == label).float().mean()
        # lossと精度をTensorBoardにログとして記録
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        # 損失を返して、学習を続ける
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, label = batch
        logits = self(x1, x2)
        loss = self.criterion(logits, label.float())
        preds = (torch.sigmoid(logits) >= 0.5).int()
        acc = (preds == label).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
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

class FGNetPairDataset(Dataset):
    def __init__(self, df, image_dir, transform, person_ids, min_age_diff=10, n_neg=2):
        self.df = df[df['person_id'].isin(person_ids)]
        self.image_dir = image_dir
        self.transform = transform
        self.min_age_diff = min_age_diff
        self.samples = []
        self._build_pairs(n_neg)

    def _build_pairs(self, n_neg):
        grouped = self.df.groupby('person_id')
        person_ids = grouped.groups.keys()
        for pid in person_ids:
            group = grouped.get_group(pid)
            others = self.df[self.df['person_id'] != pid]

            for i, r1 in group.iterrows():
                for j, r2 in group.iterrows():
                    if i >= j: continue
                    if abs(r1['age'] - r2['age']) < self.min_age_diff:
                        continue
                    self.samples.append((r1['filename'], r2['filename'], 1))

                sampled = others.sample(n=n_neg)
                for _, r2 in sampled.iterrows():
                    if abs(r1['age'] - r2['age']) < self.min_age_diff:
                        continue
                    self.samples.append((r1['filename'], r2['filename'], 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f1, f2, label = self.samples[idx]
        def load(fname):
            img = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")
            return self.transform(img)
        return load(f1), load(f2), torch.tensor(label, dtype=torch.long)


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

train_dataset = FGNetPairDataset(df, "images", transform, train_ids)
val_dataset = FGNetPairDataset(df, "images", transform, val_ids)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print(f"Train Dataset Size: {len(train_dataset)}")
print(f"Validation Dataset Size: {len(val_dataset)}")

# TripletNetPL（PyTorch Lightningで定義されたTriplet Lossを使うネットワーク・モデル）をインスタンス化
# 学習対象のモデルが準備
model = SiameseNetPL()
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
model = SiameseNetPL.load_from_checkpoint(best_model_path)
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