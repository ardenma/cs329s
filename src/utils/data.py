import logging
from typing import Dict, List, Callable, Union

from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import DistilBertTokenizer
logging.getLogger().setLevel(logging.INFO)

# Probably a more efficient way to do this: https://huggingface.co/docs/datasets/use_dataset.html
LABELS = {0: "false", 1: "half-true", 2: "mostly-true", 3: "true", 4: "barely-true", 5: "pants-fire"}

# Want -> 5 (pants-fire), 0 (false), 4 (barely-true), 1 (half-true), 2 (mostly-true), 3 (true)
CONVERSION = {0:5, 1:0, 2:4, 3:1, 4:2, 5:3}

def convert_label(label):
    return CONVERSION[label]

def string_to_id(string: str) -> int:
    return int(string.split('.')[0])
class LiarDataset(Dataset):
    def __init__(self, split: str="train", binary_labels: bool=False):
        super(LiarDataset, self).__init__()
        assert split in ("train", "validation", "test"), f"Unrecognized data split '{split}'"
        self.dataset = load_dataset("liar")[split]
        self.binary_labels = binary_labels
        self.tokenized = False
        logging.info(f"Loaded 'liar' dataset's {split} split")

    def __getitem__(self, idx: int) -> Dict[str, Union[int, float]]:
        if self.tokenized:
            keys = ["input_ids", "attention_mask"]
            x = {key: self.dataset[idx][key] for key in keys}
        else:
            x = self.dataset[idx]['statement']

        if self.binary_labels:
            y = 1.0 if self.dataset[idx]['label'] > 0 and self.dataset[idx]['label'] < 4 else 0.0
        else:
            y = int(self.dataset[idx]['label'])

        return {"data": x, "label": convert_label(y), "id": string_to_id(self.dataset[idx]["id"])}
    
    def __len__(self):
        return len(self.dataset)

    def get_data_for_bow(self) -> List[str]:
        return [sample['statement'] for sample in self.dataset]
    
    def get_num_classes(self) -> int:
        return 2 if self.binary_labels else 6
    
    def get_id_map(self) -> Dict[int, Dict[str, Union[int, str]]]:
        return {string_to_id(ex["id"]): {"label": ex["label"], "statement": ex["statement"]} for ex in self.dataset}

    ### THIS DOES NOT WORK RIGHT NOW ###
    def tokenize(self):
        logging.info("Tokenizing dataset...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        def tokenize_function(examples):
            return tokenizer(examples["statement"], padding="max_length", truncation=True, return_tensors='pt')
        self.dataset = self.dataset.map(tokenize_function, batched=False)
        self.tokenized = True
        return self