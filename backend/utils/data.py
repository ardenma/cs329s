import logging
from typing import Dict, List, Callable, Union

import torch
import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import DistilBertTokenizer
logging.getLogger().setLevel(logging.INFO)

# Probably a more efficient way to do this: https://huggingface.co/docs/datasets/use_dataset.html
LABELS = {0: "false", 1: "half-true", 2: "mostly-true", 3: "true", 4: "barely-true", 5: "pants-fire"}

# Want -> 5 (pants-fire), 0 (false), 4 (barely-true), 1 (half-true), 2 (mostly-true), 3 (true)
CONVERSION = {0:5, 1:0, 2:4, 3:1, 4:2, 5:3}

ACCEPTABLE_NUM_LABELS = [2, 3, 6]

def convert_label(label: int, num_labels: int) -> int:
    label = CONVERSION[label]
    if num_labels == 2:
        label = 1.0 if label > 1 else 0.0
    elif num_labels == 3:
        label = label // 2
    else:
        label = label
    return label

def string_to_id(string: str) -> int:
    return int(string.split('.')[0])

def get_label_to_classname_map(num_labels: int) -> Dict[int, str]:
    if num_labels == 2:
        return {0: "false", 1:"true"}
    elif num_labels == 3:
        return {0: "false", 1: "unsure", 2: "true"}
    elif num_labels == 6:
        return LABELS
    else:
        raise Exception("Need to choose 2, 3, or 6 labels")
class LiarDataset(Dataset):
    def __init__(self, split: str="train", num_labels: int=3):
        super(LiarDataset, self).__init__()
        assert split in ("train", "validation", "test"), f"Unrecognized data split '{split}'"
        self.dataset = load_dataset("liar")[split]
        self.num_labels = num_labels
        assert num_labels in ACCEPTABLE_NUM_LABELS, f"num_labels should be 2, 3, or 6"
        self.tokenized = False
        logging.info(f"Loaded 'liar' dataset's {split} split")

    def __getitem__(self, idx: int) -> Dict[str, Union[int, float]]:
        if self.tokenized:
            keys = ["input_ids", "attention_mask"]
            data = {key: self.dataset[idx][key] for key in keys}
        else:
            data = self.dataset[idx]['statement']

        label = convert_label(self.dataset[idx]['label'], self.num_labels)

        return {"data": data, "label": label, "id": string_to_id(self.dataset[idx]["id"])}
    
    def __len__(self):
        return len(self.dataset)

    def get_data_for_bow(self) -> List[str]:
        return [sample['statement'] for sample in self.dataset]
    
    def get_num_classes(self) -> int:
        return self.num_labels
    
    def get_class_balance(self, as_tensor: bool=False) -> Union[np.array, torch.tensor]:
        class_balance = np.zeros(self.num_labels)
        for ex in self.dataset:
            label = convert_label(ex['label'], self.num_labels)
            class_balance[label] += 1
        
        class_balance = np.array(list(map(lambda x: x / len(self.dataset), class_balance)))
        if as_tensor:
            class_balance = torch.tensor(class_balance, dtype=torch.float)

        return class_balance

    def get_class_weight_for_train(self, as_tensor: bool=False) -> Union[np.array, torch.tensor]:
        class_balance = np.zeros(self.num_labels)
        for ex in self.dataset:
            label = convert_label(ex['label'], self.num_labels)
            class_balance[label] += 1
        
        inv_class_balance = np.array(list(map(lambda x: len(self.dataset) / x, class_balance)))
        if as_tensor:
            inv_class_balance = torch.tensor(inv_class_balance, dtype=torch.float)

        return inv_class_balance

    def get_id_map(self) -> Dict[int, Dict[str, Union[int, str]]]:
        return {string_to_id(ex["id"]): {"label": convert_label(ex["label"], self.num_labels), "statement": ex["statement"]} for ex in self.dataset}

    ### THIS DOES NOT WORK RIGHT NOW ###
    def tokenize(self):
        logging.info("Tokenizing dataset...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        def tokenize_function(examples):
            return tokenizer(examples["statement"], padding="max_length", truncation=True, return_tensors='pt')
        self.dataset = self.dataset.map(tokenize_function, batched=False)
        self.tokenized = True
        return self