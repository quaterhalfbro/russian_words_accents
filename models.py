import torch
from torch import nn
import os
import json
from typing import Union


class StressNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = json.load(open(os.path.join(os.getcwd(), "params.json")))["BertNet"]
        embedding_dim = self.get_parameter("EMBEDDING_DIM")
        dict_size = self.get_parameter("DICT_SIZE")
        max_length = self.get_parameter("MAX_LENGTH")
        self.embedding_layer = nn.Embedding(num_embeddings=dict_size, embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(0.2)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(embedding_dim, 4, batch_first=True), 2)
        self.extra_dense = nn.Sequential(nn.Linear(embedding_dim * max_length, max_length), nn.ReLU(), nn.Dropout(0.1),
                                         nn.Linear(max_length, max_length))

    def get_parameter(self, key: str) -> Union[int, float]:
        return self.params[key]

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.dropout(x)
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = self.extra_dense(x)
        return x


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = json.load(open(os.path.join(os.getcwd(), "params.json")))["DenseNet"]
        embedding_dim = self.get_parameter("EMBEDDING_DIM")
        dict_size = self.get_parameter("DICT_SIZE")
        max_length = self.get_parameter("MAX_LENGTH")
        self.embedding_layer = nn.Embedding(num_embeddings=dict_size, embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Sequential(nn.Linear(embedding_dim * max_length, 512),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(512, max_length))

    def get_parameter(self, key: str) -> Union[int, float]:
        return self.params[key]

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = self.dense(x)
        return x
