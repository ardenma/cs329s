import os
import logging
import pathlib
import argparse

import torch
import faiss
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from models.distilbert import DistilBertForSequenceEmbedding
from models.heads import SoftmaxHead
from models.voting import MajorityVoter
from utils.data import LiarDataset
logging.getLogger().setLevel(logging.INFO)

embedding_size = 512
num_labels = 6

cwd = pathlib.Path(__file__).parent.resolve()
saved_models_dir = os.path.join(cwd, "saved_models")
index_dir = os.path.join(cwd, "indexes")
                      
def eval():
  test_dataset = LiarDataset("test", num_labels=num_labels)
  test_ldr = DataLoader(test_dataset, batch_size=10)

  print("Loading models...")
  embedding_model = DistilBertForSequenceEmbedding(embedding_size=embedding_size)
  embedding_model.load(os.path.join(saved_models_dir, "embedding_model.pt"))
  prediction_model = SoftmaxHead(embedding_model.get_embedding_size(), test_dataset.get_num_classes())
  prediction_model.load(os.path.join(saved_models_dir, "prediction_model.pt"))
  print("Done!")

  if torch.cuda.is_available():
    print("GPU available!")
    embedding_model.to('cuda')
    prediction_model.to('cuda')
  
  logging.info("Staring evaluation...")
  predictions = []
  labels = []
  with torch.no_grad():
    for (batch_idx, batch) in tqdm(enumerate(test_ldr)):
      embeddings = embedding_model(batch["data"]) 
      y_pred = torch.argmax(prediction_model(embeddings), axis=-1)

      if torch.cuda.is_available():
        batch["label"] = batch["label"].to('cuda', dtype=torch.long)
      y_label = batch["label"]

      for pred in y_pred:
        predictions.append(int(pred))
      for label in y_label:
        labels.append(int(label))
  print(f"Test accuracy: {accuracy_score(labels, predictions)}")

def eval_contrastive(args):
  test_dataset = LiarDataset("test", num_labels=num_labels)
  test_ldr = DataLoader(test_dataset, batch_size=10)
  id_map = LiarDataset("train", num_labels=num_labels).get_id_map()

  print("Loading models...")
  embedding_model = DistilBertForSequenceEmbedding(embedding_size=embedding_size)
  embedding_model.load(args.model_path)
  index = faiss.read_index(args.index_path)
  K = 5                          # we want to see 4 nearest neighbors
  prediction_model = MajorityVoter()
  print("Done!")

  if torch.cuda.is_available():
    print("GPU available!")
    embedding_model.to('cuda')
  
  logging.info("Staring evaluation...")
  predictions = []
  labels = []
  with torch.no_grad():
    for (batch_idx, batch) in tqdm(enumerate(test_ldr)):
      # Generate embeddings
      embeddings = embedding_model(batch["data"]) 
      embeddings = torch.nn.functional.normalize(embeddings, p=2.0, dim=-1, eps=1e-12)

      # Cosine similarity search using normalized embeddings
      D, IDs = index.search(embeddings.cpu().numpy(), K)

      # Vote based on label of top 5 nearest examples
      votes = [[id_map[ID]["label"] for ID in K_ids] for K_ids in IDs]
      y_pred = prediction_model(votes)
      y_label = batch["label"]

      # Update list of predictions and labels
      for pred in y_pred:
        predictions.append(int(pred))
      for label in y_label:
        labels.append(int(label))
  
  print(f"Test accuracy: {accuracy_score(labels, predictions)}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Evaluate a model.')
    parser.add_argument('--contrastive', action='store_true')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--index_path', type=str)
    args = parser.parse_args()

    # TODO add model_path arg for eval()
    if args.contrastive:
      if not os.path.exists(args.model_path):
          raise Exception("Need to specify a valid model path!")
      if not os.path.exists(args.index_path):
          raise Exception("Need to specify a valid index path!")


    if args.contrastive:
      eval_contrastive(args)
    else:
      eval()
