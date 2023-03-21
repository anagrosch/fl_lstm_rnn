# Train and implement lstm model with differential privacy

import os
import time
import argparse
import torch
import numpy as np
from torch import nn, optim
from sklearn.metrics import accuracy_score
from opacus import PrivacyEngine
from torch.utils.data import DataLoader
from lstm_model import Model
from dataset import Dataset
from utils import SaveBestModel, save_final, get_model, save_plots, save_params, update_params

MAX_GRAD_NORM = 0.5
EPSILON = 50.0
DELTA = 1/(2*len(train_data))

def valid_file(filename):
	"""
	Function to check that the csv file input is a csv file.
	"""
	ext = os.path.splitext(filename)[1][1:]
	if ext != 'csv':
		parser.error('Invalid file type. Does not end in csv')
	return filename


def train_and_save(train_data, valid_data, model, optimizer, args):
	"""
	Function to run training, validation, and add checkpoints.
	"""
	train_loss, valid_loss = [], []
	train_acc, valid_acc = [], []

	criterion = nn.CrossEntropyLoss()
	save_best = SaveBestModel()

	start_time = time.perf_counter()

	for epoch in range(args.max_epochs):
		train_epoch_loss, train_epoch_acc = train_epoch(train_data, criterion, optimizer, model, epoch, args)
		valid_epoch_loss, valid_epoch_acc = validate(valid_data, criterion, model, args)

		train_loss.append(train_epoch_loss)
		valid_loss.append(valid_epoch_loss)
		
		train_acc.append(train_epoch_acc)
		valid_acc.append(valid_epoch_acc)

		save_best(valid_epoch_loss, epoch, model, optimizer, criterion)

	end_time = time.perf_counter()

	save_final(args.max_epochs, model, optimizer, criterion)
	save_params(model, optimizer)
	save_plots(train_loss, valid_loss, train_acc, valid_acc, "diff_priv", end_time-start_time)

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
	running_acc = 0.0
	dataloader = DataLoader(dataset, batch_size=args.batch_size)

	privacy_engine = PrivacyEngine()
	model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
		module=model,
		optimizer=optimizer,
		data_loader=dataloader,
		epochs=args.max_epochs,
		target_epsilon=EPSILON,
		target_delta=DELTA,
		max_grad_norm=MAX_GRAD_NORM,
	)

	state_h, state_c = model.init_state(args.sequence_length)

	for batch, (x, y) in enumerate(dataloader):
		counter += 1
		optimizer.zero_grad()	#clear existing gradients of previous epoch
			
		y_pred, (state_h, state_c) = model(x, (state_h, state_c))
		loss = criterion(y_pred.transpose(1, 2), y)
		running_loss += loss.item()
		acc = accuracy_score(y, y_pred.transpose(1, 2))
		running_acc += acc

		state_h = state_h.detach()
		state_c = state_c.detach()

		loss.backward()		#backpropagation & calculate gradients
		optimizer.step()	#update weights

		print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item(), 'accuracy': acc })

	epoch_loss = running_loss/counter
	epoch_acc = running_acc/counter
	return epoch_loss, epoch_acc


def validate(dataset, criterion, model, args):
	"""
	Function to validate the model with the validation dataset.
	"""
	model.eval()

	counter = 0
	running_loss = 0.0
	running_acc = 0.0
	dataloader = DataLoader(dataset, batch_size=args.batch_size)

	state_h, state_c = model.init_state(args.sequence_length)

	with torch.no_grad():
		for batch, (x, y) in enumerate(dataloader):
			counter += 1
			y_pred, (state_h, state_c) = model(x, (state_h, state_c))
			loss = criterion(y_pred.transpose(1, 2), y)
			running_loss += loss.item()
			running_acc += accuracy_score(y, y_pred.transpose(1,2))

	epoch_loss = running_loss/counter
	epoch_acc = running_acc/counter
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
parser.add_argument('-bs', '--batch-size', type=int, default=64)
parser.add_argument('-sl', '--sequence-length', type=int, default=4)
parser.add_argument('-cf', '--csv-file', type=valid_file, default='data/reddit-cleanjokes.csv')
parser.add_argument('-t', '--train', action='store_true')
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

# train model
if args.train:
	train_and_save(train_data, valid_data, model, optimizer, args)

# run model for prediction
if args.predict: 
	print(predict(train_data, model, optimizer, args))

