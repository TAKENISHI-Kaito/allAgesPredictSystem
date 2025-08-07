import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import pytorch_lightning as pl

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
        #その特徴を、128次元などの埋め込みベクトルに変換
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