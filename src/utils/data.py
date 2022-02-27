import logging
from typing import Dict, List, Callable

from torch.utils.data import Dataset
from datasets import load_dataset

# Probably a more efficient way to do this: https://huggingface.co/docs/datasets/use_dataset.html
class LiarDataset(Dataset):
    def __init__(self, split: str="train", binary_labels: bool=False):
        super(LiarDataset, self).__init__()
        assert split in ("train", "validation", "test"), f"Unrecognized data split '{split}'"
        self.dataset = load_dataset("liar")[split]
        self.binary_labels = binary_labels
        logging.info(f"Loaded 'liar' dataset's {split} split")

    def __getitem__(self, idx: int) -> Dict[str, float]:
        x = self.dataset[idx]['statement']
        if self.binary_labels:
            y = 1.0 if self.dataset[idx]['label'] > 0 and self.dataset[idx]['label'] < 4 else 0.0
        else:
            y = float(self.dataset[idx]['label'])
        return {"data": x, "label": y}
    
    def __len__(self):
        return len(self.dataset)

    def get_data_for_bow(self) -> List[str]:
        return [sample['statement'] for sample in self.dataset]
    
    def get_num_classes(self) -> int:
        return 2 if self.binary_labels else 6

    def map(self, f: Callable, batched: bool=True):
        self.dataset.map(f, batched)
        return self