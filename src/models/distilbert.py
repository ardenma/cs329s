import os
from typing import Union, List, Callable

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, BatchEncoding

class DistilBertForSequenceEmbedding(torch.nn.Module):
    def __init__(self, embedding_size: int=100):
        super(DistilBertForSequenceClassification, self).__init__()
        self.embedding_size = embedding_size
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=embedding_size)

    def forward(self, x: Union[List[str], BatchEncoding], tokenize: bool=False) -> torch.tensor:
        encoded_input = self.tokenizer(x, return_tensors='pt') if tokenize else x
        output = self.model(**encoded_input)
        return output
    
    def save(self, filepath: str):
        assert not os.path.exists(filepath), f"{filepath} already exists!"
        torch.save({"embedding_size": self.embedding_size, "state_dict": self.state_dict()}, filepath)
        print(f"Saved DistilBertForSequenceEmbedding model to: {filepath}")


    def load(self, filepath: str):
        assert os.path.exists(filepath), f"{filepath} does not exist!"
        save_dict = torch.load(filepath)
        self.embedding_size = save_dict["embedding_size"]
        self.load_state_dict(save_dict["state_dict"])
    
    @staticmethod
    def get_tokenize_function() -> Callable:
        def tokenize_function(examples, *args, **kwargs):
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            return tokenizer(["statement"], padding="max_length", truncation=True)
        return tokenize_function