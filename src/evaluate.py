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
from models.voting import MajorityVoter
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
    index = create_index(embedding_model, train_ldr)
  prediction_model = MajorityVoter()
  print("Done!")
  
  K = 3

  eval_contrastive("f1_score", embedding_model, index, prediction_model, K, id_map, test_ldr)
  
def eval_contrastive(metric: str, embedding_model, index: faiss.IndexIDMap, prediction_model, K: int, id_map, dataloader: DataLoader):
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
    
  for label, pred in zip(labels, predictions):
    print(f"model predicted {pred} with label {label}")

  if metric == "accuracy":
    result = accuracy_score(labels, predictions)
    print(f"Eval accuracy: {result}")
  elif metric == "f1_score":
    result = f1_score(labels, predictions, average='weighted')
    print(f"Eval weighted f1_score: {result}")
  else:
    raise Exception(f"Unsupported metric: '{metric}'")

  return result


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Evaluate a model.')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--index_path', type=str)
    args = parser.parse_args()

    # TODO add model_path arg for eval()
    if not os.path.exists(args.model_path):
        raise Exception("Need to specify a valid model path!")

    eval_wrapper(args)


# def eval():
#   test_dataset = LiarDataset("test", num_labels=num_labels)
#   test_ldr = DataLoader(test_dataset, batch_size=10)

#   print("Loading models...")
#   embedding_model = DistilBertForSequenceEmbedding()
#   embedding_model.load(os.path.join(saved_models_dir, "embedding_model.pt"))
#   prediction_model = SoftmaxHead(embedding_model.get_embedding_size(), test_dataset.get_num_classes())
#   prediction_model.load(os.path.join(saved_models_dir, "prediction_model.pt"))
#   print("Done!")

#   if torch.cuda.is_available():
#     print("GPU available!")
#     embedding_model.to('cuda')
#     prediction_model.to('cuda')
  
#   logging.info("Staring evaluation...")
#   predictions = []
#   labels = []
#   with torch.no_grad():
#     for (batch_idx, batch) in tqdm(enumerate(test_ldr)):
#       embeddings = embedding_model(batch["data"]) 
#       y_pred = torch.argmax(prediction_model(embeddings), axis=-1)

#       if torch.cuda.is_available():
#         batch["label"] = batch["label"].to('cuda', dtype=torch.long)
#       y_label = batch["label"]

#       for pred in y_pred:
#         predictions.append(int(pred))
#       for label in y_label:
#         labels.append(int(label))
#   print(f"Test accuracy: {accuracy_score(labels, predictions)}")