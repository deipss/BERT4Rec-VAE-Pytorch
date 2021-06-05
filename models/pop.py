from .base import BaseModel
import torch.nn as nn

class PopModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.E = nn.Embedding(1, 1)

    @classmethod
    def code(cls):
        return 'pop'

    def forward(self, x):
        return self
