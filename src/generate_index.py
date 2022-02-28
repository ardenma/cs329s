import os
import pathlib
import argparse

import torch
import faiss
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.distilbert import DistilBertForSequenceEmbedding
from utils.data import LiarDataset

embedding_size = 512

cwd = pathlib.Path(__file__).parent.resolve()
index_dir = os.path.join(cwd, "indexes")
if not os.path.exists(index_dir): os.mkdir(index_dir)

def main(args):
    index = faiss.IndexFlatL2(embedding_size)   # build the index
    index = faiss.IndexIDMap(index)             # index returns IDs instead of embeddings
    print(index.is_trained)

    train_dataset = LiarDataset("train")
    train_ldr = DataLoader(train_dataset, batch_size=10)

    embedding_model = DistilBertForSequenceEmbedding(embedding_size)
    embedding_model.load(args.model_path)
    embedding_model.eval()

    if torch.cuda.is_available():
        print("GPU available!")
        embedding_model.to('cuda')
    with torch.no_grad():
        for (batch_idx, batch) in tqdm(enumerate(train_ldr)):
            embeddings = embedding_model(batch["data"]).cpu().numpy()
            index.add_with_ids(embeddings, batch["id"].cpu().numpy())

    print(f"Indexed {index.ntotal} vectors.")

    if not os.path.exists(os.path.join(index_dir, "index")):
        filename = os.path.join(index_dir, "index")

    else:
        i = 1
        while os.path.exists(os.path.join(index_dir, f"tmp_index_{i}")):
            i += 1
        filename = os.path.join(index_dir, f"tmp_index_{i}")

    faiss.write_index(index, filename)
    print(f"Saved index to: {filename}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Build an index.')
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise Exception("Need to specify a valid model path!")
    
    main(args)