import logging
from typing import Dict

from torch.utils.data import Dataset
from datasets import load_dataset

# Probably a more efficient way to do this: https://huggingface.co/docs/datasets/use_dataset.html
class LiarDataset(Dataset):
    def __init__(self, split="train"):
        super(LiarDataset, self).__init__()
        assert split in ("train", "validation", "test"), f"Unrecognized data split '{split}'"
        self.dataset = load_dataset("liar")[split]
        logging.info(f"Loaded 'liar' dataset's {split} split")

    def __getitem__(self, idx: int) -> Dict[str, float]:
        x = self.dataset[idx]['statement']
        y = 1.0 if self.dataset[idx]['label'] > 0 and self.dataset[idx]['label'] < 4 else 0.0
        return {"data": x, "label": y}
    
    def __len__(self):
        return len(self.dataset)

    def get_data_for_bow(self):
        return [sample['statement'] for sample in self.dataset]