import os
import pathlib
import logging
from typing import Union

import torch
import faiss
from torch.utils.data import DataLoader
from tqdm import tqdm

from backend.models.distilbert import DistilBertForSequenceEmbedding
from backend.utils.data import LiarDataset


def create_index(
    embedding_model: DistilBertForSequenceEmbedding, num_labels: int
) -> faiss.IndexIDMap:
    # Setup training dataloader for the index creation
    train_dataset = LiarDataset("train", num_labels=num_labels)
    train_ldr = DataLoader(train_dataset, batch_size=10)

    logging.info("Creating index...")
    index = faiss.IndexFlatIP(embedding_model.get_embedding_size())  # build the index
    index = faiss.IndexIDMap(index)  # index returns IDs instead of embeddings
    logging.info(index.is_trained)

    if torch.cuda.is_available():
        logging.info("GPU available! Using it to build index.")
        embedding_model.to("cuda")

    with torch.no_grad():
        for (batch_idx, batch) in tqdm(enumerate(train_ldr)):
            embeddings = embedding_model(batch["data"]).cpu().numpy()
            index.add_with_ids(embeddings, batch["id"].cpu().numpy())

    logging.info(f"Indexed {index.ntotal} vectors.")

    return index


def load_index(model_name: str) -> Union[None, faiss.IndexIDMap]:
    # Setup directories and pathnames
    index_dir = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "indexes")
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
    index_path = os.path.join(index_dir, model_name + ".index")
    index = None

    if os.path.exists:
        logging.info(f"Loading index at: {index_path}")
        index = faiss.read_index(index_path)

    return index


def cache_index(
    model_name: str, embedding_model: DistilBertForSequenceEmbedding, num_labels: int
) -> faiss.IndexIDMap:
    # Setup directories and pathnames
    index_dir = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "indexes")
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
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
