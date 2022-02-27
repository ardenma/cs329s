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
logging.getLogger().setLevel(logging.INFO)


embedding_size = 512
cwd = pathlib.Path(__file__).parent.resolve()
saved_models_dir = os.path.join(cwd, "saved_models")
                      
def main():
  test_dataset = LiarDataset("test")
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

if __name__=="__main__":
    main()
