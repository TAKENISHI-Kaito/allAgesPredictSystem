import pandas as pd, random, os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from tripletModel import TripletNetPL
from tripletDataset import FGNetTripletDataset
from utils import evaluate_model, visualize_gradcam
from config import transform

df = pd.read_csv("annotations.csv")
person_ids = sorted(df['person_id'].unique())
random.seed(42)
random.shuffle(person_ids)
split = int(0.8 * len(person_ids))
train_ids = set(person_ids[:split])
val_ids = set(person_ids[split:])

train_dataset = FGNetTripletDataset(df, "images", transform, train_ids)
val_dataset = FGNetTripletDataset(df, "images", transform, val_ids)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

model = TripletNetPL()

logger = TensorBoardLogger("tensorboard", name="fgnet_logs")
checkpoint = ModelCheckpoint(monitor="train_loss", save_top_k=1, mode="min")

trainer = pl.Trainer(max_epochs=50, logger=logger, accelerator="auto", callbacks=[checkpoint])
trainer.fit(model, train_loader, val_loader)

model = TripletNetPL.load_from_checkpoint(checkpoint.best_model_path)
acc, auc, correct, incorrect = evaluate_model(model, df, val_ids)
print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}")
print(f"Correct: {len(correct)} | Incorrect: {len(incorrect)}")

if incorrect:
    fname = incorrect[0]["filename1"]
    visualize_gradcam(model, os.path.join("images", fname))