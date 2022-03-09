import os
import logging
import re
from typing import List, Dict

import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from tqdm import tqdm

class BOWEmbedding:
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('punkt')
        self.stop_words = set(stopwords.words('english'))

        # Intitialized in fit() or load()
        self.bag_of_words = set()
        self.bow_map = {}
        self.embedding_size = -1
        self.fitted = False
    
    def __call__(self, samples: List[Dict[str, str]]) -> List[torch.tensor]:
        assert self.fitted is True, "Please call fit() or load() before making a prediction."
        if not isinstance(samples, list): samples = [samples]

        # 3. Bag of Words encoding
        embeddings = torch.zeros(len(samples), self.embedding_size)
        for i, sample in enumerate(samples):
            text = re.sub(r'[^\w\s]', '', sample)
            tokens = word_tokenize(text)
            for tok in tokens:
                if not tok.lower() in self.stop_words and tok.lower() in self.bow_map:
                    embeddings[i][self.bow_map[tok.lower()]] += 1
        
        return embeddings

    def fit(self, X: List[str], n_iter=-1, verbose=False):
        logging.info("Beginning to fit embedding_model...")
        X = tqdm(X) if verbose is True else X

        # 1. Get Bag of Words
        for iter, sample_i in enumerate(X):
            sample_i_text = re.sub(r'[^\w\s]', '', sample_i)
            sample_i_tokens = word_tokenize(sample_i_text)
            filtered_tokens = [tok.lower() for tok in sample_i_tokens if not tok.lower() in self.stop_words]
            self.bag_of_words |= set(filtered_tokens)

            if iter > 0 and iter == n_iter:
                break
             
        self.embedding_size = len(self.bag_of_words)
   
        # 2. Map each word in BoW to an index
        for index, word in enumerate(self.bag_of_words):
            self.bow_map[word] = index
        
        self.fitted = True
        logging.info("Done.")

    def get_embedding_size(self) -> int:
        assert self.embedding_size != -1, "Need to fit() or load() model before retrieving embedding size."
        return self.embedding_size

    def save(self, filepath: str):
        assert not os.path.exists(filepath), f"{filepath} already exists!"
        torch.save({"bow_map": self.bow_map, "embedding_size": self.embedding_size, "fitted": self.fitted}, filepath)

    def load(self, filepath: str):
        assert os.path.exists(filepath), f"{filepath} does not exist!"
        save_dict = torch.load(filepath)
        self.bow_map = save_dict["bow_map"]
        self.embedding_size = save_dict["embedding_size"]
        self.fitted = save_dict["fitted"]