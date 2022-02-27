import os
import logging
import pathlib

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from models.distilbert import DistilBertForSequenceEmbedding
from models.heads import SoftmaxHead
from utils.data import LiarDataset
from utils.loss import contrastive_loss
logging.getLogger().setLevel(logging.INFO)

embedding_size = 512
num_epochs = 5
                      
def main():
  train_dataset = LiarDataset("train")
  train_ldr = DataLoader(train_dataset, batch_size=10)

  embedding_model = DistilBertForSequenceEmbedding(embedding_size)

  if torch.cuda.is_available():
    print("GPU available!")
    embedding_model.to('cuda')

  optimizer = torch.optim.Adam(embedding_model.parameters(), lr=1e-3)

  logging.info("Starting training loop...")
  for epoch in tqdm(range(num_epochs)):
    for (batch_idx, batch) in tqdm(enumerate(train_ldr)):
      optimizer.zero_grad()
      
      # Forward pass
      # Embedding model implicitly moves tensors to GPU 
      embeddings = embedding_model(batch["data"]) 

      # # Getting label and moving onto GPU if necessary
      if torch.cuda.is_available():
        batch["label"] = batch["label"].to('cuda', dtype=torch.long)
      y_label = batch["label"]

      # # Compute Loss
      loss = contrastive_loss(embeddings, y_label)

      # Backward pass
      loss.backward()
      optimizer.step()
  
  logging.info("Done.")

  # Saving model
  cwd = pathlib.Path(__file__).parent.resolve()
  saved_models_dir = os.path.join(cwd, "saved_models")
  if not os.path.exists(saved_models_dir): os.mkdir(saved_models_dir)

  print("Saving models...")
  if not os.path.exists(os.path.join(saved_models_dir, "embedding_model.pt")):
      embedding_model.save(os.path.join(saved_models_dir, "embedding_model.pt"))
  else:
      i = 1
      while os.path.exists(os.path.join(saved_models_dir, f"tmp_embedding_model_{i}.pt")):
          i += 1
      embedding_model.save(os.path.join(saved_models_dir, f"tmp_embedding_model_{i}.pt"))

  print("Done!")

if __name__=="__main__":
    main()
