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
from utils.index import create_index

# for num_labels in [3, 6]:
#     for split in ["train", "validation", "test"]:
#         dataset = LiarDataset(split, num_labels=num_labels)
#         print(f"Liar dataset ({split}) with n_labels={num_labels} class balance: \n{dataset.get_class_balance(as_tensor=True)}")

num_labels = 3

def main(args):
    test_dataset = LiarDataset("test", num_labels=num_labels)
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
    
    query = test_dataset[100]["statement"]
    label = test_dataset[100]["label"]

    # Generate embeddings
    with torch.no_grad():    
        embeddings = embedding_model(query) 
        embeddings = torch.nn.functional.normalize(embeddings, p=2.0, dim=-1, eps=1e-12)

        # Cosine similarity search using normalized embeddings
        D, IDs = index.search(embeddings.cpu().numpy(), K)

        # Vote based on label of top 5 nearest examples
        votes = [[id_map[ID]["label"] for ID in K_ids] for K_ids in IDs]
        pred = prediction_model(votes)
        print(f"prediction: {pred}, label: {label}")
        print(D)
        print(IDs)

            

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


    main(args)