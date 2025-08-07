import torch
from torch.utils.data import Dataset
from PIL import Image
import os, random

# PyTorchのDatasetを継承したクラス
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