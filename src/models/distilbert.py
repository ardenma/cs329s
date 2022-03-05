import os
import logging
from turtle import forward
from typing import Union, List, Callable

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, BatchEncoding
from src.models.heads import SoftmaxHead

class DistilBertForSequenceEmbedding(torch.nn.Module):
    def __init__(self, embedding_size: int=100):
        super(DistilBertForSequenceEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=embedding_size)

    def forward(self, x: List[str]) -> torch.tensor:
        encoded_input = self.tokenizer(x, padding="max_length", truncation=True, return_tensors='pt')
        encoded_input = {key: val.to('cuda') if next(self.model.parameters()).is_cuda else val for key, val in encoded_input.items()}
        output = self.model(**encoded_input).logits
        normalized_output = torch.nn.functional.normalize(output, p=2.0, dim=-1, eps=1e-12)
        return normalized_output

    def get_embedding_size(self) -> int:
        assert self.embedding_size != -1, "Need to fit() or load() model before retrieving embedding size."
        return self.embedding_size
    
    def save(self, filepath: str):
        assert not os.path.exists(filepath), f"{filepath} already exists!"
        torch.save({"embedding_size": self.embedding_size, "state_dict": self.state_dict()}, filepath)
        logging.info(f"Saved DistilBertForSequenceEmbedding model to: {filepath}")

    def load(self, filepath: str):
        assert os.path.exists(filepath), f"{filepath} does not exist!"
        save_dict = torch.load(filepath)
        self.embedding_size = save_dict["embedding_size"]
        self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=self.embedding_size)
        self.load_state_dict(save_dict["state_dict"])
        logging.info(f"Loaded DistilBertForSequenceEmbedding model from: {filepath}")
    