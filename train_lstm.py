# Train and implement lstm model

import time, argparse, os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from lstm_model import Model
from dataset import Dataset
from utils import SaveBestModel, save_final, get_model, save_plots, save_params, update_params

def valid_file(filename, choices):
	parser = argparse.ArgumentParser()
	parser.add_argument('-me', '--max-epochs', type=int, default=100)
	ext = os.path.splitext(filename)[1][1:]
	if ext not in choices:
		parser.error("Invalid file type. Does not end in {}".format(choices))
	return filename


def train_and_save(train_data, valid_data, model, optimizer, args):
	"""
	Function to run training, validation, and add checkpoints.
	"""
	train_loss, valid_loss = [], []

	criterion = nn.CrossEntropyLoss()
	save_best = SaveBestModel()

	start_time = time.perf_counter()

	for epoch in range(args.max_epochs):
		train_epoch_loss = train_epoch(train_data, criterion, optimizer, model, epoch, args)
		valid_epoch_loss = validate(valid_data, criterion, model, args)

		train_loss.append(train_epoch_loss)
		valid_loss.append(valid_epoch_loss)

		save_best(valid_epoch_loss, epoch, model, optimizer, criterion)

	save_final(args.max_epochs, model, optimizer, criterion)
	save_plots(train_loss, valid_loss)

	end_time = time.perf_counter()

	print('TRAINING COMPLETE')
	print(f"Time Taken: {end_time - start_time}")
	print('----------------------')


def train_epoch(dataset, criterion, optimizer, model, epoch, args):
	"""
	Function to train each epoch of the model.
	"""
	model.train()
	
	counter = 0
	running_loss = 0.0
	dataloader = DataLoader(dataset, batch_size=args.batch_size)

	print(dataloader)

	state_h, state_c = model.init_state(args.sequence_length)

	for batch, (x, y) in enumerate(dataloader):
		counter += 1
		optimizer.zero_grad()	#clear existing gradients of previous epoch
			
		y_pred, (state_h, state_c) = model(x, (state_h, state_c))
		loss = criterion(y_pred.transpose(1, 2), y)
		running_loss += loss.item()

		state_h = state_h.detach()
		state_c = state_c.detach()

		loss.backward()		#backpropagation & calculate gradients
		optimizer.step()	#update weights

		print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })

	epoch_loss = running_loss/counter
	return epoch_loss


def validate(dataset, criterion, model, args):
	"""
	Function to validate the model with the validation dataset.
	"""
	model.eval()

	counter = 0
	running_loss = 0.0
	dataloader = DataLoader(dataset, batch_size=args.batch_size)

	state_h, state_c = model.init_state(args.sequence_length)

	with torch.no_grad():
		for batch, (x, y) in enumerate(dataloader):
			counter += 1
			y_pred, (state_h, state_c) = model(x, (state_h, state_c))
			loss = criterion(y_pred.transpose(1, 2), y)
			running_loss += loss.item()

	epoch_loss = running_loss/counter
	return epoch_loss


def predict(dataset, model, optimizer, text, next_words):
	"""
	Function to predict next words with trained model.
	"""
	model.eval()

	words = text.split(' ')
	state_h, state_c = model.init_state(len(words))

	for i in range(0, next_words):
		x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
		y_pred, (state_h, state_c) = model(x, (state_h, state_c))

		last_word_logits = y_pred[0][-1]
		p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
		word_index = np.random.choice(len(last_word_logits), p=p)
		words.append(dataset.index_to_word[word_index])

	return ' '.join([str(char) for char in words])	#convert list to string

# main function

parser = argparse.ArgumentParser()
parser.add_argument('-me', '--max-epochs', type=int, default=100)
parser.add_argument('-bs', '--batch-size', type=int, default=256)
parser.add_argument('-sl', '--sequence-length', type=int, default=4)
parser.add_argument('-t', '--train', action='store_true')
parser.add_argument('-cf', '--csv-file', type=lambda s:valid_file(s,("csv")), default='data/reddit-cleanjokes.csv')
parser.add_argument('-p', '--predict', action='store_true')
parser.add_argument('-pt', '--predict-text', type=str, default='Knock knock. Whos there?')
parser.add_argument('-ps', '--predict-size', type=int, default=100)
args = parser.parse_args()

# get training and validation datasets
train_data = Dataset(0.4, args)
valid_data = Dataset(0.4, args, train=False)

model = Model(train_data)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
	
# get checkpoint from best model and update parameters
try:
	model, optimizer, epoch, loss = get_model(model, optimizer, 'outputs/best_model.pth')
	model = update_params(model)
except:
	print('Cannot load best model. Continuing with untrained model.')

if args.train:
	train_and_save(train_data, valid_data, model, optimizer, args)
	save_params(model, optimizer)

if args.predict: 
	print(predict(train_data, model, optimizer, text=args.predict_text, next_words=args.predict_size))

