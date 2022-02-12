import os
import logging
import pathlib

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.baselines import LogisticRegression
from models.embeddings import BOWEmbedding
from utils.data import LiarDataset

dataset = LiarDataset("train")
train_ldr = DataLoader(dataset, batch_size=2)

embedding_model = BOWEmbedding()
embedding_model.fit(dataset.get_data_for_bow(), n_iter=-1, verbose=True)
prediction_model = LogisticRegression(embedding_model.get_embedding_size())  
prediction_model.train()

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(prediction_model.parameters(), lr=0.01)

logging.info("Starting training loop...")
for epoch in tqdm(range(1)):
  for (batch_idx, batch) in enumerate(train_ldr):
    optimizer.zero_grad()
    # Forward pass
    embeddings = embedding_model(batch["data"]) 
    y_pred = prediction_model(embeddings)
    y_label = batch["label"].reshape(-1, 1).float()
    # Compute Loss
    loss = criterion(y_pred, y_label)
    # Backward pass
    loss.backward()
    optimizer.step()
logging.info("Done.")

eval_sample = dataset[23]
eval_embedding = embedding_model(eval_sample["data"])
eval_pred = prediction_model(eval_embedding)

print(eval_pred)
print(f"predicted Y value: {eval_pred.data[0][0]}, true Y value: {eval_sample['label']}")

# Test saving/loading
cwd = pathlib.Path(__file__).parent.resolve()
saved_models_dir = os.path.join(cwd, "saved_models")
if not os.path.exists(saved_models_dir): os.mkdir(saved_models_dir)

print("Saving models...")
if not os.path.exists(os.path.join(saved_models_dir, "embedding_model.pt")):
  embedding_model.save(os.path.join(saved_models_dir, "embedding_model.pt"))
if not os.path.exists(os.path.join(saved_models_dir, "prediction_model.pt")):
  prediction_model.save(os.path.join(saved_models_dir, "prediction_model.pt"))
print("Done!")

print("Loading models...")
new_embedding_model = BOWEmbedding()
new_embedding_model.load(os.path.join(saved_models_dir, "embedding_model.pt"))
new_prediction_model = LogisticRegression(new_embedding_model.get_embedding_size())
new_prediction_model.load(os.path.join(saved_models_dir, "prediction_model.pt"))
print("Done!")

print(type(new_embedding_model))
print(type(new_prediction_model))

new_eval_embedding = new_embedding_model(eval_sample["data"])
new_eval_pred = new_prediction_model(new_eval_embedding)

print(new_eval_pred)
print(f"predicted Y value: {new_eval_pred.data[0][0]}, true Y value: {eval_sample['label']}")