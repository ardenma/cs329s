import os
import logging
import pathlib

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from models.baselines import LogisticRegression
from models.embeddings import BOWEmbedding
from utils.data import LiarDataset

train_dataset = LiarDataset("train")
train_ldr = DataLoader(train_dataset, batch_size=10)
test_dataset = LiarDataset("test")
test_ldr = DataLoader(test_dataset, batch_size=10)

embedding_model = BOWEmbedding()
embedding_model.fit(train_dataset.get_data_for_bow(), n_iter=-1, verbose=True)
prediction_model = LogisticRegression(embedding_model.get_embedding_size())
prediction_model.train()

criterion = torch.nn.BCELoss(reduction="mean")
optimizer = torch.optim.SGD(prediction_model.parameters(), lr=0.01)

logging.info("Starting training loop...")
for epoch in tqdm(range(10)):
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


logging.info("Staring evalatuion...")
predictions = []
labels = []
for (batch_idx, batch) in enumerate(test_ldr):
    embeddings = embedding_model(batch["data"])
    y_pred = prediction_model(embeddings)
    y_label = batch["label"].reshape(-1, 1)
    for pred in y_pred:
        predictions.append(round(float(pred)))
    for label in y_label:
        labels.append(int(label))

percent_one = (sum(labels) / len(labels)) * 100
print(
    f"Test label distribution: {percent_one:.2f}% with label '1' and {100 - percent_one:.2f}% with label '0'"
)
print(f"Test accuracy: {accuracy_score(labels, predictions)}")


# eval_sample = train_dataset[23]
# eval_embedding = embedding_model(eval_sample["data"])
# eval_pred = prediction_model(eval_embedding)

# print(eval_pred)
# print(f"predicted Y value: {eval_pred.data[0][0]}, true Y value: {eval_sample['label']}")

# Test saving/loading
cwd = pathlib.Path(__file__).parent.resolve()
saved_models_dir = os.path.join(cwd, "saved_models")
if not os.path.exists(saved_models_dir):
    os.mkdir(saved_models_dir)

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

# print(type(new_embedding_model))
# print(type(new_prediction_model))

# new_eval_embedding = new_embedding_model(eval_sample["data"])
# new_eval_pred = new_prediction_model(new_eval_embedding)

# print(new_eval_pred)
# print(f"predicted Y value: {new_eval_pred.data[0][0]}, true Y value: {eval_sample['label']}")
