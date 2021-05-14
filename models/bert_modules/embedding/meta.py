import torch.nn as nn
import torch


class MetaEmbedding(nn.Embedding):

    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)

    def forward(self, x):
        batch_size = x.size(0)
        E = super().forward(x)
        return torch.sum(E,dim=-2)
