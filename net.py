import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from typing import List
import torch.nn.functional as F
from sentiment_data import WordEmbeddings


class FeedForward(nn.Module):
    """
    Deep average network for sentiment classification
    """

    def __init__(self, word_vectors: WordEmbeddings, d_hidden=256, d_out=2):

        super(FeedForward, self).__init__()

        p_drop = 0.2
        self.d_out = d_out
        self.word_vectors = word_vectors

        # layers of model
        emb_tensor = torch.FloatTensor(word_vectors.vectors)
        d_emb = emb_tensor.shape[1]

        self.embedder = nn.Embedding.from_pretrained(emb_tensor, freeze=True, padding_idx=0)
        self.fc1 = nn.Linear(d_emb, d_hidden)
        self.dropout = nn.Dropout(p_drop)
        self.hidden1 = nn.Linear(d_hidden, d_hidden)
        self.hidden2 = nn.Linear(d_hidden, d_out)
        self.out_layer = nn.LogSoftmax(dim=0)

        # initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.hidden1.weight)
        nn.init.xavier_uniform_(self.hidden2.weight)

    def forward(self, x:List[str]):

        # TODO add batching

        # convert text to embeddings
        indices = list()

        for w in x:
            idx = self.word_vectors.word_indexer.index_of(w)
            if idx == -1:
                idx = self.word_vectors.word_indexer.index_of("UNK")
            indices.append(idx)

        x = torch.LongTensor(indices)

        x = self.embedder(x)

        # average embeddings
        x = x.mean(dim=0)

        # pass through network
        x = self.fc1(x)
        x = F.relu(x)
        x = self.hidden1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = self.out_layer(x)
        x = x.reshape((-1, self.d_out))
        return x


class RNN(nn.Module):

    def __init__(self, word_vectors:WordEmbeddings, seq_len:int, batch_sz:int, h_size=20, d_out=2):

        super(RNN, self).__init__()

        emb_tensor = torch.FloatTensor(word_vectors.vectors)
        self.d_emb = emb_tensor.shape[1]
        self.h_size = h_size
        self.d_out = d_out
        self.num_layers = 2
        self._batch_sz = batch_sz

        # load Glove embeddings
        self.embedder = nn.Embedding.from_pretrained(emb_tensor, freeze=True)
        self.encoder = nn.LSTM(self.d_emb, self.h_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*self.h_size, self.d_out)
        self.out_layer = nn.LogSoftmax(dim=1)

        # initialize weights of network
        nn.init.xavier_uniform_(self.fc.weight)

    @property
    def batch_sz(self):
        return self._batch_sz

    @batch_sz.setter
    def batch_sz(self, batch_sz:int):
        self._batch_sz = batch_sz

    def forward(self, x:Tensor) -> Tensor:
        """
        :param x: indices of words for input sentences of shape batch_sz x max_seq_len
        :returns: log probabilities
        """

        embeddings = self.embedder(x)
        _, (h, c) = self.encoder(embeddings)

        h = h.reshape(self.num_layers, 2, self.batch_sz, self.h_size)

        h_final = h[1, :, :, :].reshape(self.batch_sz, -1)

        x = self.fc(h_final)
        x = self.out_layer(x)
        return x

    @batch_sz.setter
    def batch_sz(self, value):
        self._batch_sz = value
