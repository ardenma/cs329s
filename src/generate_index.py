import os
import pathlib
import argparse

import torch
import faiss
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.distilbert import DistilBertForSequenceEmbedding
from utils.data import LiarDataset

cwd = pathlib.Path(__file__).parent.resolve()
index_dir = os.path.join(cwd, "indexes")
if not os.path.exists(index_dir): os.mkdir(index_dir)

def main(args):
    train_dataset = LiarDataset("train")
    train_ldr = DataLoader(train_dataset, batch_size=10)

    embedding_model = DistilBertForSequenceEmbedding()
    embedding_model.load(args.model_path)
    model_name = os.path.basename(args.model_path).split("_")[0]
    embedding_model.eval()

    index = faiss.IndexFlatL2(embedding_model.get_embedding_size())   # build the index
    index = faiss.IndexIDMap(index)             # index returns IDs instead of embeddings
    print(index.is_trained)

    if torch.cuda.is_available():
        print("GPU available!")
        embedding_model.to('cuda')
    with torch.no_grad():
        for (batch_idx, batch) in tqdm(enumerate(train_ldr)):
            embeddings = embedding_model(batch["data"]).cpu().numpy()
            index.add_with_ids(embeddings, batch["id"].cpu().numpy())

    print(f"Indexed {index.ntotal} vectors.")

    filename = os.path.join(index_dir, model_name + "-index")
    if not os.path.exists(filename):
        filename = os.path.join(index_dir, model_name + "-index")
        faiss.write_index(index, filename)
        print(f"Saved index to: {filename}")
    else:
        print(f"Filename {filename} already exists, please delete it and try again.")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Build an index.')
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise Exception("Need to specify a valid model path!")
    
    main(args)