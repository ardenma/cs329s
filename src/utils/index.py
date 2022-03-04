import torch
import faiss
from torch.utils.data import DataLoader
from tqdm import tqdm

def create_index(embedding_model, dataloader: DataLoader) -> faiss.IndexIDMap:
    print("Creating index...")
    index = faiss.IndexFlatIP(embedding_model.get_embedding_size())   # build the index
    index = faiss.IndexIDMap(index)             # index returns IDs instead of embeddings
    print(index.is_trained)

    if torch.cuda.is_available():
        print("GPU available!")
        embedding_model.to('cuda')

    with torch.no_grad():
        for (batch_idx, batch) in tqdm(enumerate(dataloader)):
            embeddings = embedding_model(batch["data"]).cpu().numpy()
            index.add_with_ids(embeddings, batch["id"].cpu().numpy())

    print(f"Indexed {index.ntotal} vectors.")

    return index