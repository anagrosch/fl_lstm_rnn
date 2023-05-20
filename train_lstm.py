# Train and implement lstm model

import math
import os
import time
import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from lstm_model import Model
from dataset import Dataset
from utils import SaveBestModel, save_final, fine_tune_model, save_plots, save_params, update_params

AGGR_PERIOD = 0 #increment after each aggregation

def valid_file(filename):
	"""
	Function to check that the csv file input is a csv file.
	"""
	ext = os.path.splitext(filename)[1][1:]
	if ext != 'csv':
		parser.error('Invalid file type. Does not end in csv')
	return filename


def train_and_save(model, optimizer, args):
	"""
	Function to run training, validation, and add checkpoints.
	"""
	train_loss, valid_loss = [], []
	train_acc, valid_acc = [], []

	criterion = nn.CrossEntropyLoss()
	save_best = SaveBestModel()

	start_time = time.perf_counter()

	for epoch in range(args.max_epochs):
		train_epoch_loss, train_epoch_acc = train_epoch(criterion, optimizer, model, epoch, args)
		valid_epoch_loss, valid_epoch_acc = validate(criterion, model, args)

		train_loss.append(train_epoch_loss)
		valid_loss.append(valid_epoch_loss)
		
		train_acc.append(train_epoch_acc)
		valid_acc.append(valid_epoch_acc)

		save_best(valid_epoch_loss, epoch, model, optimizer, criterion)

	end_time = time.perf_counter()

	save_final(args.max_epochs, model, optimizer, criterion)
	save_params(model)
	save_plots(train_loss, valid_loss, train_acc, valid_acc, "lstm_rnn", (end_time-start_time)/60, AGGR_PERIOD)

	print('TRAINING COMPLETE')
	print(f"Time Taken: {round((end_time - start_time)/60, 6)} minutes")
	print('----------------------')


def train_epoch(criterion, optimizer, model, epoch, args):
	"""
	Function to train each epoch of the model.
	"""
	train_data = Dataset(args, type='train')
	model.train()
	
	total = 0
	running_loss = 0.0
	running_acc = 0.0
	dataloader = DataLoader(train_data, batch_size=args.batch_size)

	state_h, state_c = model.init_state(args.sequence_length)

	for batch, (x, y) in enumerate(dataloader):
		optimizer.zero_grad()	#clear existing gradients of previous epoch
			
		y_pred, (state_h, state_c) = model(x, (state_h, state_c))
		loss = criterion(y_pred.transpose(1, 2), y)

		state_h = state_h.detach()
		state_c = state_c.detach()

		loss.backward()		#backpropagation & calculate gradients
		optimizer.step()	#update weights

		running_loss += loss.item()
		classes = torch.argmax(y_pred.transpose(1, 2), dim=1)
		acc = np.mean([float(classes.flatten()[label] == y.flatten()[label]) for label in range(torch.numel(y))])
		running_acc += acc

		print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item(), 'accuracy': acc })

	epoch_loss = running_loss/len(dataloader)
	epoch_acc = running_acc/len(dataloader)
	return epoch_loss, epoch_acc


def validate(criterion, model, args):
	"""
	Function to validate the model with the validation dataset.
	"""
	valid_data = Dataset(args, type='valid')
	model.eval()

	running_loss = 0.0
	running_acc = 0.0
	dataloader = DataLoader(valid_data, batch_size=args.batch_size)

	state_h, state_c = model.init_state(args.sequence_length)

	print('\nRunning validation set...\n')

	with torch.no_grad():
		for batch, (x, y) in enumerate(dataloader):
			y_pred, (state_h, state_c) = model(x, (state_h, state_c))
			loss = criterion(y_pred.transpose(1, 2), y)
			running_loss += loss.item()
			classes = torch.argmax(y_pred.transpose(1, 2), dim=1)
			running_acc += np.mean([float(classes.flatten()[label] == y.flatten()[label]) for label in range(torch.numel(y))])

	epoch_loss = running_loss/len(dataloader)
	epoch_acc = running_acc/len(dataloader)
	return epoch_loss, epoch_acc


def predict(dataset, model, optimizer, args):
	"""
	Function to predict next words with trained model.
	"""
	model.eval()

	words = args.predict_text.split(' ')
	state_h, state_c = model.init_state(len(words))

	for i in range(0, args.predict_size):
		x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
		y_pred, (state_h, state_c) = model(x, (state_h, state_c))

		last_word_logits = y_pred[0][-1]
		p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
		word_index = np.random.choice(len(last_word_logits), p=p)
		words.append(dataset.index_to_word[word_index])

	return ' '.join([str(char) for char in words])	#convert list to string

# main function
parser = argparse.ArgumentParser(prog='TRAIN_LSTM', usage='%(prog)s [options]')
parser.add_argument('-me', '--max-epochs', type=int, default=32)
parser.add_argument('-bs', '--batch_size', type=int, default=64)
parser.add_argument('-bn', '--batch_num', type=int, default=64)
parser.add_argument('-sl', '--sequence-length', type=int, default=4)
parser.add_argument('-cf', '--csv-file', type=valid_file, default='data/reddit-cleanjokes.csv')
parser.add_argument('-r', '--round', type=int, default=1)
parser.add_argument('-t', '--train', action='store_true')
parser.add_argument('-p', '--predict', action='store_true')
parser.add_argument('-pt', '--predict-text', type=str, default='Knock knock. Whos there?')
parser.add_argument('-ps', '--predict-size', type=int, default=10)
args = parser.parse_args()

# get full dataset
full_data = Dataset(args)

# create '/outputs/best_model.pth' if dne
if not os.path.exists('outputs'):
	os.mkdir('outputs')
	print('/outputs/ directory created.\n')
	with open(os.path.join('outputs','best_model.pth'), 'w'):
		pass

# get checkpoint from best model and update parameters
try:
	model, optimizer = fine_tune_model('outputs/best_model.pth',full_data)
	print('Continuing from model checkpoint.\n')
except:
	model = Model(full_data.n_uniq_words)
	optimizer = optim.Adam(model.parameters(), lr=0.0001)

	# initialize weights
	for name, weight in model.named_parameters():
		nn.init.uniform_(weight, -1/math.sqrt(model.lstm_size), 1/math.sqrt(model.lstm_size))
	nn.init.xavier_uniform_(model.fc.weight)

	print('Cannot load best model. Continuing with untrained model.\n')

# train model
if args.train:
	train_and_save(model, optimizer, args)

# run model for prediction
if args.predict: 
	print(predict(full_data, model, optimizer, args))

