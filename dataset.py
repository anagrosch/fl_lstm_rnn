# Dataset class for preparing model data

import sys
import math
import torch
import pandas as pd
from collections import Counter

class Dataset(torch.utils.data.Dataset):
	def __init__(self, args, type='full'):
		self.type = type
		self.args = args
		self.words = self.load_words()
		self.uniq_words = self.get_uniq_words()
		self.n_uniq_words = len(self.uniq_words)

		self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
		self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

		self.words_indexes = [self.word_to_index[w] for w in self.words]

	def load_words(self):
		train_df = pd.read_csv(self.args.csv_file)
		num_rows = self.get_num_rows(train_df)
		try:
			if self.type=='train':
				text = train_df[(2*(self.args.round-1))*num_rows:(2*self.args.round-1)*num_rows]['Joke'].str.cat(sep=' ')
			elif self.type=='valid': 
				text = train_df[(2*self.args.round-1)*num_rows:(2*self.args.round)*num_rows]['Joke'].str.cat(sep=' ')
			elif self.type=='full':
				text = train_df[(2*(self.args.round-1))*num_rows:(2*self.args.round)*num_rows]['Joke'].str.cat(sep=' ')
			else: sys.exit("Error: Invalid dataset type.")
		except:
			sys.exit("Error: Round is too high for dataset. Try lower round or a different dataset")
		return text.split(' ')

	def get_num_rows(self, train_df):
		row = 0
		total = 0
		while total < self.args.batch_size*self.args.batch_num:
			total += train_df.iloc[row]['Joke'].count(' ') + 1
			row += 1
		return row

	def get_uniq_words(self):
		word_counts = Counter(self.words)
		return sorted(word_counts, key=word_counts.get, reverse=True)

	def __len__(self):
		return len(self.words_indexes) - self.args.sequence_length

	def __getitem__(self, index):
		return (torch.tensor(self.words_indexes[index:index+self.args.sequence_length]),
			torch.tensor(self.words_indexes[index+1:index+self.args.sequence_length+1]))

