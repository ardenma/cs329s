import os
import logging
import pathlib

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import eval_contrastive
from models.distilbert import DistilBertForSequenceEmbedding
from models.heads import SoftmaxHead
from models.voting import MajorityVoter
from utils.data import LiarDataset
from utils.loss import contrastive_loss
from utils.optim import get_optimizer
from utils.index import create_index

logging.getLogger().setLevel(logging.INFO)
cwd = pathlib.Path(__file__).parent.resolve()

run = wandb.init(
  project="cs329s", 
  entity="cs329s", 
  config=os.path.join(cwd, "config", "config_default.yaml"),
  reinit=True,
  # mode="disabled"  # for debug
  )

logging.info(f"\nwandb.config:\n{wandb.config}\n")

# Create saved models dir if it doesn't exist yet
cwd = pathlib.Path(__file__).parent.resolve()
saved_models_dir = os.path.join(cwd, "saved_models")
if not os.path.exists(saved_models_dir): os.mkdir(saved_models_dir)
experiment_dir = os.path.join(saved_models_dir, f"{wandb.run.name}")
if not os.path.exists(experiment_dir): os.mkdir(experiment_dir)

def train():
  # Generate filename
  embedding_model_filename = os.path.join(experiment_dir, f"{wandb.run.name}_embedding_model.pt")
  prediction_model_filename = os.path.join(experiment_dir, f"{wandb.run.name}_prediction_model.pt")

  # Get Dataset
  train_dataset = LiarDataset("train", num_labels=wandb.config.num_labels)
  train_ldr = DataLoader(train_dataset, batch_size=wandb.config.batch_size)
  dev_dataset = LiarDataset("validation", num_labels=wandb.config.num_labels)
  dev_ldr = DataLoader(dev_dataset, batch_size=wandb.config.batch_size)
  class_weight = train_dataset.get_class_weight_for_train(as_tensor=True)

  # Load Models
  embedding_model = DistilBertForSequenceEmbedding(wandb.config.embedding_size)
  if wandb.config.loss_type != "contrastive":
    prediction_model = SoftmaxHead(input_length=wandb.config.embedding_size, num_classes=train_dataset.get_num_classes())
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight, reduction='mean')

  # For eval
  eval_model = MajorityVoter()
  id_map = train_dataset.get_id_map()
  best_metric = 0

  if torch.cuda.is_available():
    print("GPU available!")
    embedding_model.to('cuda')
    if wandb.config.loss_type != "contrastive":
      prediction_model.to('cuda')
      criterion.to('cuda')

  # Get model params for optimizer
  parameters = list(embedding_model.parameters())
  if wandb.config.loss_type != "contrastive":
    parameters += list(prediction_model.parameters())

  # Setup optimizer
  optimizer = get_optimizer(name=wandb.config.optimizer, parameters=parameters, lr=wandb.config.learning_rate)
  logging.info(f"Using optimizer: {wandb.config.optimizer}.")

  logging.info("Starting training loop...")
  for epoch in tqdm(range(1, wandb.config.epochs + 1)):
    for (batch_idx, batch) in tqdm(enumerate(train_ldr)):
      optimizer.zero_grad()
      
      # Forward pass
      # Embedding model implicitly moves tensors to GPU 
      embeddings = embedding_model(batch["data"]) 
      if wandb.config.loss_type != "contrastive":
        y_pred = prediction_model(embeddings)

      # Getting label and moving onto GPU if necessary
      if torch.cuda.is_available():
        batch["label"] = batch["label"].to('cuda', dtype=torch.long)
      y_label = batch["label"]

      # Compute Loss
      loss = 0
      if wandb.config.loss_type != "contrastive":
        loss += criterion(y_pred, y_label)
      if wandb.config.loss_type != "normal":
        loss += contrastive_loss(embeddings, y_label, wandb.config.num_labels, wandb.config.same_label_multiplier)

      # Logging
      wandb.log({"loss": loss, "epoch": epoch})
      #wandb.watch(embedding_model)

      # Backward pass
      loss.backward()
      optimizer.step()

    # Evaluation
    if epoch % wandb.config.eval_frequency == 0:
      index = create_index(embedding_model, train_ldr)
      test_metric = eval_contrastive(wandb.config.metric, embedding_model, index, eval_model, wandb.config.K, id_map, dev_ldr)
      wandb.log({wandb.config.metric: test_metric, "epoch": epoch})
      if (test_metric > best_metric):
        wandb.run.summary[f"best_{wandb.config.metric}"] = test_metric
        wandb.run.summary["best_epoch"] = epoch

        best_metric = test_metric
        embedding_model.save(f"{embedding_model_filename.split('.')[0]}_epoch_{epoch}_{test_metric:.3f}.pt")
        
        # Artifact tracking
        artifact = wandb.Artifact(f'{wandb.run.name}-{wandb.config.num_labels}-labels', type='distilbert-embedding-model')
        artifact.add_file(f"{embedding_model_filename.split('.')[0]}_epoch_{epoch}_{test_metric:.3f}.pt", 
                          name=f"{wandb.run.name}_epoch_{epoch}_{test_metric:.3f}.pt")
        run.log_artifact(artifact)

        if wandb.config.loss_type != "contrastive":
          prediction_model.save(f"{prediction_model_filename.split('.')[0]}_epoch_{epoch}_{test_metric:.3f}.pt")
    
  logging.info("Done.")

  # Saving model
  print("Saving models...")
  embedding_model.save(embedding_model_filename)
  if wandb.config.loss_type != "contrastive":
    prediction_model.save(prediction_model_filename)
  print("Done!")

if __name__=="__main__":
    wandb.require(experiment="service")
    assert wandb.config.loss_type in ["contrastive", "normal", "hybrid"], f"Unknown loss type '{wandb.config.loss_type}'"
    train()