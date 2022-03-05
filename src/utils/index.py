import os
import pathlib
import logging

import torch
import faiss
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.distilbert import DistilBertForSequenceEmbedding
from utils.data import LiarDataset

def create_index(embedding_model: DistilBertForSequenceEmbedding, num_labels: int) -> faiss.IndexIDMap:
    # Setup training dataloader for the index creation
    train_dataset = LiarDataset("train", num_labels=num_labels)
    train_ldr = DataLoader(train_dataset, batch_size=10)

    logging.info("Creating index...")
    index = faiss.IndexFlatIP(embedding_model.get_embedding_size())   # build the index
    index = faiss.IndexIDMap(index)             # index returns IDs instead of embeddings
    logging.info(index.is_trained)

    if torch.cuda.is_available():
        logging.info("GPU available! Using it to build index.")
        embedding_model.to('cuda')

    with torch.no_grad():
        for (batch_idx, batch) in tqdm(enumerate(train_ldr)):
            embeddings = embedding_model(batch["data"]).cpu().numpy()
            index.add_with_ids(embeddings, batch["id"].cpu().numpy())
    
    # else:
    #     logging.info("No GPU available, using CPU with multiprocessing to build index.")
        
    #     with torch.no_grad():
    #         for (batch_idx, batch) in tqdm(enumerate(train_ldr)):
    #             # embeddings = pool.map(embedding_model.forward, batch["data"])
    #             embeddings = torch.vstack(Parallel(n_jobs=-1)(delayed(embedding_model.forward)(x) for x in batch["data"])).numpy()
    #             print(embeddings)
    #             index.add_with_ids(embeddings, batch["id"].numpy())

    logging.info(f"Indexed {index.ntotal} vectors.")

    return index

def cache_index(model_name: str, embedding_model: DistilBertForSequenceEmbedding, num_labels: int):
    # Setup directories and pathnames
    index_dir = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "indexes")
    if not os.path.exists(index_dir): os.mkdir(index_dir)
    index_path = os.path.join(index_dir, model_name + ".index")

    # Cache a copy of the index on the disk if it's not already there, else load cached copy
    if not os.path.exists(index_path):
        index = create_index(embedding_model, num_labels)
        logging.info(f"Saving index to: {index_path}")
        faiss.write_index(index, index_path)
    else:
        logging.info(f"Loading index at: {index_path}")
        index = faiss.read_index(index_path)  

    return index