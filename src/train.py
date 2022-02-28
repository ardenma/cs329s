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
from utils.optim import get_optimizer

logging.getLogger().setLevel(logging.INFO)
cwd = pathlib.Path(__file__).parent.resolve()

wandb.init(
  project="cs329s", 
  entity="ardenma", 
  config = os.path.join(cwd, "config", "config_default.yaml"),
  reinit=True,
  # mode="disabled"  # for debug
  )

# Create saved models dir if it doesn't exist yet
cwd = pathlib.Path(__file__).parent.resolve()
saved_models_dir = os.path.join(cwd, "saved_models")
if not os.path.exists(saved_models_dir): os.mkdir(saved_models_dir)
experiment_dir = os.path.join(saved_models_dir, f"{wandb.run.name}")
if not os.path.exists(experiment_dir): os.mkdir(experiment_dir)

def train():
  train_dataset = LiarDataset("train", num_labels=wandb.config.num_labels)
  train_ldr = DataLoader(train_dataset, batch_size=wandb.config.batch_size)

  embedding_model = DistilBertForSequenceEmbedding(wandb.config.embedding_size)
  prediction_model = SoftmaxHead(input_length=wandb.config.embedding_size, num_classes=train_dataset.get_num_classes())

  if torch.cuda.is_available():
    print("GPU available!")
    embedding_model.to('cuda')
    prediction_model.to('cuda')

  criterion = torch.nn.CrossEntropyLoss(reduction='mean')
  parameters = list(embedding_model.parameters()) + list(prediction_model.parameters())
  optimizer = get_optimizer(name=wandb.config.optimizer, parameters=parameters, lr=wandb.config.learning_rate)
  logging.info(f"Using optimizer: {wandb.config.optimizer}.")

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

      # Logging
      wandb.log({"loss": loss, "epoch": epoch})
      #wandb.watch(embedding_model)

      # Backward pass
      loss.backward()
      optimizer.step()
  
  logging.info("Done.")

  # Saving model
  print("Saving models...")
  embedding_model.save(os.path.join(experiment_dir, f"{wandb.run.name}_embedding_model.pt"))
  prediction_model.save(os.path.join(experiment_dir, f"{wandb.run.name}_prediction_model.pt"))
  print("Done!")

def train_contrastive():
  # Generate filename
  filename = os.path.join(experiment_dir, f"{wandb.run.name}_embedding_model.pt")

  # Get Dataset
  train_dataset = LiarDataset("train", num_labels=wandb.config.num_labels)
  train_ldr = DataLoader(train_dataset, batch_size=wandb.config.batch_size)

  # Load model
  embedding_model = DistilBertForSequenceEmbedding(wandb.config.embedding_size)
  if torch.cuda.is_available():
    print("GPU available!")
    embedding_model.to('cuda')

  # Create optimizer
  optimizer = get_optimizer(name=wandb.config.optimizer, parameters=embedding_model.parameters(), lr=wandb.config.learning_rate)
  logging.info(f"Using optimizer: {wandb.config.optimizer}.")


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
      loss = contrastive_loss(embeddings, y_label, wandb.config.same_label_multiplier)
      
      # Logging
      wandb.log({"loss": loss, "epoch": epoch})
      #wandb.watch(embedding_model)

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
    wandb.require(experiment="service")
    parser = argparse.ArgumentParser(description='Training script.')
    parser.add_argument('--contrastive', action='store_true')
    args = parser.parse_args()

    if args.contrastive:
      train_contrastive()
    else:
      train()
