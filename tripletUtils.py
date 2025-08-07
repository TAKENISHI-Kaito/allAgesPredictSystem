import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from PIL import Image
import os
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
from config import transform_eval

def evaluate_model(model, df, val_ids, image_dir="images"):
    model.eval()
    pairs, labels = [], []
    val_df = df[df['person_id'].isin(val_ids)]
    for pid in val_ids:
        same = val_df[val_df['person_id'] == pid]
        other = val_df[val_df['person_id'] != pid]
        for i, r1 in same.iterrows():
            for j, r2 in same.iterrows():
                if i >= j or abs(r1['age'] - r2['age']) < 10:
                    continue
                pairs.append((r1['filename'], r2['filename']))
                labels.append(1)
            sampled = other.sample(n=2)
            for _, r2 in sampled.iterrows():
                if abs(r1['age'] - r2['age']) < 10: continue
                pairs.append((r1['filename'], r2['filename']))
                labels.append(0)

    def get_emb(fname):
        img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
        img = transform_eval(img).unsqueeze(0).to(model.device)
        with torch.no_grad():
            return model(img)

    results = []
    for (f1, f2), label in zip(pairs, labels):
        emb1 = get_emb(f1)
        emb2 = get_emb(f2)
        dist = F.pairwise_distance(emb1, emb2).item()
        score = 1 - dist
        pred = int(score >= 0.5)
        results.append({"filename1": f1, "filename2": f2, "label": label, "pred": pred, "score": score, "dist": dist})

    acc = accuracy_score([r['label'] for r in results], [r['pred'] for r in results])
    auc = roc_auc_score([r['label'] for r in results], [r['score'] for r in results])
    correct = [r for r in results if r["label"] == r["pred"]]
    incorrect = [r for r in results if r["label"] != r["pred"]]
    return acc, auc, correct, incorrect

def visualize_gradcam(model, img_path):
    model.eval()
    original_img = Image.open(img_path).convert('RGB').resize((224, 224))
    input_tensor = to_tensor(original_img).unsqueeze(0).to(model.device)
    rgb_img = np.array(original_img).astype(np.float32) / 255.0
    target_layers = [model.base.layer4[-1]]
    cam = GradCAM(model=model.base, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    plt.imshow(visualization)
    plt.title(f"Grad-CAM on {os.path.basename(img_path)}")
    plt.axis("off")
    plt.show()