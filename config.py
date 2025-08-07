from torchvision import transforms

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

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])