from typing import Iterator

import torch
from torch.nn.parameter import Parameter

def get_optimizer(name: str, parameters: Iterator[Parameter], lr: float) -> torch.optim.Optimizer:
    if name == "Adam":
        return torch.optim.Adam(parameters, lr=lr)
    elif name == "AdamW":
        return torch.optim.AdamW(parameters, lr=lr)
    else:
        raise Exception(f"Unknown optimizer '{name}'...")