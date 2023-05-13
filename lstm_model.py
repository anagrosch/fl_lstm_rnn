# PyTorch base model for all nn modules

import torch
import torch.nn as nn
import math
from torch import nn, optim

class Model(nn.Module):
	def __init__(self, n_vocab):
		super(Model, self).__init__()
		self.lstm_size = 128
		self.embedding_dim = 128 
		self.num_layers = 3
		self.n_vocab = n_vocab

		self.embedding = nn.Embedding(
			num_embeddings=self.n_vocab, 
			embedding_dim=self.embedding_dim,
		)
		self.lstm = nn.LSTM(
			input_size=self.lstm_size,
			hidden_size=self.lstm_size,
			num_layers=self.num_layers,
			dropout=0.2,
		)
		self.fc = nn.Linear(self.lstm_size, self.n_vocab)

	def forward(self, x, prev_state):
		embed = self.embedding(x)
		output, state = self.lstm(embed, prev_state)
		logits = self.fc(output)
		return logits, state

	def init_state(self, sequence_length):
		return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
			torch.zeros(self.num_layers, sequence_length, self.lstm_size))
