import os
import logging
import pathlib
import argparse

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.distilbert import DistilBertForSequenceEmbedding
from models.heads import SoftmaxHead
from utils.data import LiarDataset
from utils.loss import contrastive_loss
logging.getLogger().setLevel(logging.INFO)

wandb.init(
  project="cs329s", 
  entity="ardenma", 
  config = {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 10,
    "embedding_size": 512
  })

def train():
  train_dataset = LiarDataset("train")
  train_ldr = DataLoader(train_dataset, batch_size=wandb.config.batch_size)

  embedding_model = DistilBertForSequenceEmbedding(wandb.config.embedding_size)
  prediction_model = SoftmaxHead(input_length=wandb.config.embedding_size, num_classes=train_dataset.get_num_classes())

  if torch.cuda.is_available():
    print("GPU available!")
    embedding_model.to('cuda')
    prediction_model.to('cuda')

  criterion = torch.nn.CrossEntropyLoss(reduction='mean')
  parameters = list(embedding_model.parameters()) + list(prediction_model.parameters())
  optimizer = torch.optim.Adam(parameters, lr=wandb.config.learning_rate)

  logging.info("Starting training loop...")
  for epoch in tqdm(range(wandb.config.epochs)):
    for (batch_idx, batch) in tqdm(enumerate(train_ldr)):
      optimizer.zero_grad()
      
      # Forward pass
      # Embedding model implicitly moves tensors to GPU 
      embeddings = embedding_model(batch["data"]) 
      y_pred = prediction_model(embeddings)

      # Getting label and moving onto GPU if necessary
      if torch.cuda.is_available():
        batch["label"] = batch["label"].to('cuda', dtype=torch.long)
      y_label = batch["label"]

      # Compute Loss
      loss = criterion(y_pred, y_label)

      wandb.log({"loss": loss})
      wandb.watch(embedding_model)

      # Backward pass
      loss.backward()
      optimizer.step()
    
    if epoch % 10 == 0:
      wandb.log()
  
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

  if not os.path.exists(os.path.join(saved_models_dir, "prediction_model.pt")):
      prediction_model.save(os.path.join(saved_models_dir, "prediction_model.pt"))
  else:
      i = 1
      while os.path.exists(os.path.join(saved_models_dir, f"tmp_prediction_model_{i}.pt")):
          i += 1
      prediction_model.save(os.path.join(saved_models_dir, f"tmp_prediction_model_{i}.pt"))

  print("Done!")

def train_contrastive():
  # Create saved models dir if it doesn't exist yet
  cwd = pathlib.Path(__file__).parent.resolve()
  saved_models_dir = os.path.join(cwd, "saved_models")
  if not os.path.exists(saved_models_dir): os.mkdir(saved_models_dir)
  
  # Generate filename
  if not os.path.exists(os.path.join(saved_models_dir, "embedding_model.pt")):
      filename = os.path.join(saved_models_dir, "embedding_model.pt")
  else:
      i = 1
      while os.path.exists(os.path.join(saved_models_dir, f"tmp_embedding_model_{i}.pt")):
          i += 1
      filename = os.path.join(saved_models_dir, f"tmp_embedding_model_{i}.pt")

  # Get Dataset
  train_dataset = LiarDataset("train")
  train_ldr = DataLoader(train_dataset, batch_size=wandb.config.batch_size)

  # Load model
  embedding_model = DistilBertForSequenceEmbedding(wandb.config.embedding_size)
  if torch.cuda.is_available():
    print("GPU available!")
    embedding_model.to('cuda')

  # Create optimizer
  optimizer = torch.optim.Adam(embedding_model.parameters(), lr=wandb.config.learning_rate)

  logging.info("Starting training loop...")
  for epoch in tqdm(range(wandb.config.epochs)):
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
      
      # Logging
      wandb.log({"loss": loss})
      wandb.watch(embedding_model)

      # Backward pass
      loss.backward()
      optimizer.step()
    
    if epoch % 10 == 0:
      embedding_model.save(f"{filename.split('.')[0]}_epoch_{epoch}.pt")
  
  logging.info("Done.")



  print("Saving models...")
  embedding_model.save(filename)
  print("Done!")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--contrastive', action='store_true')
    args = parser.parse_args()

    if args.contrastive:
      train_contrastive()
    else:
      train()
