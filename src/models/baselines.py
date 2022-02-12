import os

import torch

class LogisticRegression(torch.nn.Module):
      def __init__(self, input_length: int=-1):
         super(LogisticRegression, self).__init__()
         self.input_length = input_length
         self.linear = torch.nn.Linear(input_length, 1) if input_length != -1 else None

      def forward(self, x: torch.tensor):
         y_pred = torch.sigmoid(self.linear(x))
         return y_pred
      
      def save(self, filepath: str):
         assert not os.path.exists(filepath), f"{filepath} already exists!"
         # TODO think about only saving state dict...
         torch.save({"input_length": self.input_length, "state_dict": self.state_dict()}, filepath)

      def load(self, filepath: str):
         assert os.path.exists(filepath), f"{filepath} does not exist!"
         save_dict = torch.load(filepath)
         self.input_length = save_dict["input_length"]
         self.linear = torch.nn.Linear(save_dict["input_length"], 1)
         self.load_state_dict(save_dict["state_dict"])
