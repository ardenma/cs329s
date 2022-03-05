import os
import logging
import pathlib
import argparse

import torch
import faiss
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from models.distilbert import DistilBertForSequenceEmbedding
from models.heads import SoftmaxHead
from models.voting import WeightedMajorityVoter
from utils.data import LiarDataset
from utils.index import create_index
logging.getLogger().setLevel(logging.INFO)

num_labels = 3

cwd = pathlib.Path(__file__).parent.resolve()
saved_models_dir = os.path.join(cwd, "saved_models")
index_dir = os.path.join(cwd, "indexes")
                      
def eval_wrapper(args):
  test_dataset = LiarDataset("test", num_labels=num_labels)
  test_ldr = DataLoader(test_dataset, batch_size=10)
  id_map = LiarDataset("train", num_labels=num_labels).get_id_map()

  print("Loading models...")
  embedding_model = DistilBertForSequenceEmbedding()
  embedding_model.load(args.model_path)
  if args.index_path:
    index = faiss.read_index(args.index_path)
  else:
    train_dataset = LiarDataset("train", num_labels=num_labels)
    train_ldr = DataLoader(train_dataset, batch_size=10)
    model_name = os.path.basename(args.model_path).split('.')[0]
    index_path = os.path.join(index_dir, model_name + ".index")
    if not os.path.exists(index_path):
      index = create_index(embedding_model, train_ldr)
      print(f"Saving index to: {index_path}")
      faiss.write_index(index, index_path)
    else:
      print(f"Loading index at: {index_path}")
      index = faiss.read_index(index_path)

  prediction_model = WeightedMajorityVoter()
  print("Done!")
  
  K = 3

  eval_contrastive(embedding_model, index, prediction_model, K, id_map, test_ldr)
  
def eval_contrastive( embedding_model, index: faiss.IndexIDMap, prediction_model, K: int, id_map, dataloader: DataLoader):
  if torch.cuda.is_available():
    print("GPU available!")
    embedding_model.to('cuda')
  
  logging.info("Staring evaluation...")
  predictions = []
  labels = []
  with torch.no_grad():
    for (batch_idx, batch) in tqdm(enumerate(dataloader)):
      # Generate embeddings
      embeddings = embedding_model(batch["data"]) 

      # Cosine similarity search using normalized embeddings
      S, IDs = index.search(embeddings.cpu().numpy(), K)

      # Vote based on label of top 5 nearest examples
      votes = [[id_map[ID]["label"] for ID in K_ids] for K_ids in IDs]
      y_pred = prediction_model(votes, S)
      y_label = batch["label"]

      # Update list of predictions and labels
      for pred in y_pred:
        predictions.append(int(pred))
      for label in y_label:
        labels.append(int(label))
    
  for label, pred in zip(labels, predictions):
    print(f"model predicted {pred} with label {label}")

  metrics = {"accuracy": accuracy_score(labels, predictions), "f1_score": f1_score(labels, predictions, average='weighted')}

  print(f"Eval accuracy: {metrics['accuracy']}")
  print(f"Eval weighted f1_score: {metrics['f1_score']}")

  return metrics


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Evaluate a model.')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--index_path', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise Exception("Need to specify a valid model path!")

    eval_wrapper(args)
