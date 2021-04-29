from .base import BaseModel
from .bert_modules.bert import BERT

import torch.nn as nn


class BERTCNNModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.bert = BERT(args)
        self.cnn = nn.Conv1d(in_channels=args.bert_max_len,
                             out_channels=args.bert_max_len,
                             kernel_size=args.kernel_size,
                             stride=args.stride)
        lout_size = (args.bert_hidden_units-(args.kernel_size-1)-1)//args.stride+1
        self.out = nn.Linear(lout_size, args.num_items + 1)


    @classmethod
    def code(cls):
        return 'bert_cnn'

    def forward(self, x):
        x = self.bert(x)
        x = self.cnn(x)
        return self.out(x)
