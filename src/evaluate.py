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
from models.voting import WeightedMajorityVoter
from utils.data import LiarDataset
from utils.index import cache_index
from utils.artifacts import get_model_path_from_artifact
logging.getLogger().setLevel(logging.INFO)

num_labels = 3

cwd = pathlib.Path(__file__).parent.resolve()
saved_models_dir = os.path.join(cwd, "saved_models")
if not os.path.exists(saved_models_dir): os.mkdir(saved_models_dir)
index_dir = os.path.join(cwd, "indexes")
if not os.path.exists(index_dir): os.mkdir(index_dir)
artifacts_dir = os.path.join(cwd, "artifacts")
if not os.path.exists(artifacts_dir): os.mkdir(artifacts_dir)


def eval_wrapper(args):
  test_dataset = LiarDataset("test", num_labels=num_labels)
  test_ldr = DataLoader(test_dataset, batch_size=10)
  id_map = LiarDataset("train", num_labels=num_labels).get_id_map()

  logging.info("Loading models...")
  embedding_model = DistilBertForSequenceEmbedding()
  if args.model_path:
    embedding_model.load(args.model_path)
    if args.index_path:
      index = faiss.read_index(args.index_path)
    else:
      model_name = os.path.basename(args.model_path).split('.')[0]
      index = cache_index(model_name, embedding_model, num_labels)

  elif args.artifact:  # e.g. daily-tree-15-3-labels:v4
    model_name = "_".join(args.artifact.split('_')[0:3])
    model_path = get_model_path_from_artifact(args.artifact)
    embedding_model.load(model_path)
    index = cache_index(model_name, embedding_model, num_labels)

  prediction_model = WeightedMajorityVoter()
  logging.info("Done!")
  
  K = 3

  logging.info(f"Running evaluation with model: '{model_name}'...")
  eval_contrastive(embedding_model, index, prediction_model, K, id_map, test_ldr)
  
def eval_contrastive(embedding_model: DistilBertForSequenceEmbedding, index: faiss.IndexIDMap, prediction_model: WeightedMajorityVoter, K: int, id_map, dataloader: DataLoader):
  if torch.cuda.is_available():
    logging.info("GPU available!")
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
    logging.info(f"model predicted {pred} with label {label}")

  metrics = {"accuracy": accuracy_score(labels, predictions), "f1_score": f1_score(labels, predictions, average='weighted')}

  logging.info(f"Eval accuracy: {metrics['accuracy']}")
  logging.info(f"Eval weighted f1_score: {metrics['f1_score']}")

  return metrics


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Evaluate a model.')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--index_path', type=str)
    parser.add_argument('--artifact', type=str)
    args = parser.parse_args()

    if not args.model_path and not args.artifact:
        raise Exception("Need to specify a model path or artifact!")

    eval_wrapper(args)
