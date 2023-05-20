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
		lower_bound, middle_bound, upper_bound = self.get_row_bounds(train_df)

		if self.type=='train':
			text = train_df[lower_bound:middle_bound]['nlp_data'].str.cat(sep=' ')
		elif self.type=='valid': 
			text = train_df[middle_bound+1:upper_bound]['nlp_data'].str.cat(sep=' ')
		elif self.type=='full':
			text = train_df[lower_bound:upper_bound]['nlp_data'].str.cat(sep=' ')
		else: sys.exit("Error: Invalid dataset type.")

		return text.split(' ')

	def get_row_bounds(self, train_df):
		row = 0
		total = 0
		counter = 0
		limit = self.args.batch_size*self.args.batch_num
		try:
			while total < limit and counter < 2*(self.args.round-1):
				#curr_row = [str(x) for x in train_df.iloc[row]['nlp_data']]
				total += str(train_df.iloc[row]['nlp_data']).count(' ') + 1
				row += 1
				if total >= limit:
					counter += 1
					total = 0
			lower_bound = row

			total = 0
			while total < limit:
				total += str(train_df.iloc[row]['nlp_data']).count(' ') + 1
				row += 1
			middle_bound = row - 1

			total = 0
			while total < limit:
				total += str(train_df.iloc[row]['nlp_data']).count(' ') + 1
				row += 1
			upper_bound = row - 1

		except Exception as e:
			print(e)
			print("Format csv file with prepare_data.py if not done already.")
			raise SystemExit(1)

		return lower_bound, middle_bound, upper_bound

	def get_uniq_words(self):
		word_counts = Counter(self.words)
		return sorted(word_counts, key=word_counts.get, reverse=True)

	def __len__(self):
		return len(self.words_indexes) - self.args.sequence_length

	def __getitem__(self, index):
		return (torch.tensor(self.words_indexes[index:index+self.args.sequence_length]),
			torch.tensor(self.words_indexes[index+1:index+self.args.sequence_length+1]))

