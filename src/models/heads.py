import os

import torch

class SoftmaxHead(torch.nn.Module):
      def __init__(self, input_length: int=-1, num_classes: int=-1):
         super(SoftmaxHead, self).__init__()
         self.input_length = input_length
         self.num_classes = num_classes
         self.linear = torch.nn.Linear(input_length, num_classes)
         self.softmax = torch.nn.Softmax(-1)

      def forward(self, x: torch.tensor) -> torch.tensor:
         y_pred = self.softmax(self.linear(x))
         return y_pred
      
      def save(self, filepath: str):
         assert not os.path.exists(filepath), f"{filepath} already exists!"
         torch.save({"input_length": self.input_length, "state_dict": self.state_dict()}, filepath)
         print(f"Saved SoftmaxHead model to: {filepath}")

      def load(self, filepath: str):
         assert os.path.exists(filepath), f"{filepath} does not exist!"
         save_dict = torch.load(filepath)
         self.input_length = save_dict["input_length"]

         self.load_state_dict(save_dict["state_dict"])
