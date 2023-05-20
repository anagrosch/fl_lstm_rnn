import os
import torch
import pickle
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from os.path import join, exists, getsize
from datetime import date
from lstm_model import Model

plt.style.use('bmh')

class SaveBestModel:
	"""
	Class to save the best model while training to
	avoid using overfit models. If current epoch's
	validation loss is less than previous least,
	save the model state.
	"""
	def __init__(self, best_valid_loss=float('inf')):
		self.best_valid_loss = best_valid_loss

	def __call__(self, current_valid_loss, epoch, model, optimizer, criterion):
		if current_valid_loss < self.best_valid_loss:
			self.best_valid_loss = current_valid_loss
			print(f"Saving best model for epoch: {epoch}\n")
			torch.save({
				'epoch': epoch+1,
				'n_vocab': model.n_vocab,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'loss': criterion,
				}, 'outputs/best_model.pth')

def save_final(epochs, model, optimizer, criterion):
	"""
	Function to save the fully trained model to disk.
	"""
	print(f"Saving final model...")
	torch.save({
		'epoch': epochs,
		'n_vocab': model.n_vocab,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'loss': criterion,
		}, 'outputs/final_model.pth')


def fine_tune_model(path, dataset):
	"""
	Function to load model checkpoint & prepare for fine-tuning.
	"""
	model, optimizer, epoch, loss = get_checkpoint(path)
	if exists(path):
		ft_model = Model(dataset.n_uniq_words)
		ft_model.lstm = model.lstm
		
		# randomly initialize output layer
		torch.nn.init.xavier_uniform_(ft_model.fc.weight)

		return ft_model, optimizer


def get_checkpoint(path):
	"""
	Function to load model checkpoint.
	"""
	if exists(path):
		checkpoint = torch.load(path)
		model = Model(checkpoint['n_vocab'])
		optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		loss = checkpoint['loss']
	else: print("Model checkpoint file path does not exist.")

	return model, optimizer, epoch, loss


def save_params(model):
	"""
	Function to get model layer weights and save each parameter to its own file
	"""
	param_path = join(os.getcwd(), "outputs", "best_model_params.pkl")
	
	param_dict = {}
	for name, param in model.named_parameters():
		param_dict[name] = param

	# write parameters to text file
	with open(param_path, 'wb') as f:
		pickle.dump(param_dict, f)
	

def update_params():
	"""
	Function to update model layer weights with aggregated parameters.
	Updates both model parameters file and model checkpoint.
	"""
	param_path = join(os.getcwd(), "outputs", "best_model_params.pkl")
	model_path = join(os.getcwd(), "outputs", "best_model.pth")

	with open("tmp.pkl", 'rb') as f:
		# get aggregated params
		tmp_dict = pickle.load(f)

	with open(param_path, 'rb') as f:
		# get local file params
		param_dict = pickle.load(f)

	# combine aggregated params with local params
	for weight in param_dict.keys():
		tmp_dict[weight], param_dict[weight] = resize_tensor(tmp_dict[weight], param_dict[weight])
		param_dict[weight] = (param_dict[weight] + tmp_dict[weight])/2

	with open(param_path, 'wb') as f:
		pickle.dump(param_dict, f)

	os.remove("tmp.pkl")

	# update model checkpoint
	model, optimizer, epoch, loss = get_checkpoint(model_path)

	for weight, param in model.named_parameters():
		param = param_dict[weight]

	save_final(epoch, model, optimizer, loss)


def resize_tensor(tensor1, tensor2):
	"""
	Function to set two tensors to the same size.
	Filles smaller tensor (tensor1) with random initializer.
	"""
	if tensor1.shape > tensor2.shape: #ensure tensor1 is smaller
		tensor2, tensor1 = resize_tensor(tensor2, tensor1)
	elif tensor1.shape < tensor2.shape:
		if len(tensor2.shape) == 1:
			tmp = torch.empty(tensor2.shape[0]-tensor1.shape[0])
			torch.nn.init.uniform_(tmp)
		else:
			tmp = torch.empty(tensor2.shape[0]-tensor1.shape[0], tensor2.shape[1])
			torch.nn.init.xavier_uniform_(tmp)
		tensor1 = torch.cat((tensor1,tmp),0)
	return tensor1, tensor2


def save_time_to_csv(csv_file, model_type, time, period):
	"""
	Function to save training times and time plot info to csv_file.
	"""
	color_dict = {'lstm_rnn': '#baf0f5', 'dp_lstm_rnn': '#e1cff4',
		      'smc_lstm_rnn': '#a4bda2', 'he_lstm_rnn': '#f5d1cb'}
	with open(csv_file, 'a', newline='') as f:
		write = csv.writer(f)

		# add headers if csv file is empty
		if getsize(csv_file) == 0:
			write.writerow(["Model Type", "Training Time", "Period", "Color"])
		write.writerow([model_type, time, period, color_dict[model_type]])


def get_csv_data(csv_file):
	"""
	Function to get plot data from csv_file.
	"""
	df = pd.read_csv(csv_file)
	x = df['Period']
	y = df['Training Time']
	c = df['Color']
	mt = df['Model Type']

	xr = []
	for i in range(len(x)):
		if mt[i] == 'lstm_rnn': xr.append(x[i]-0.03)
		elif mt[i] == 'dp_lstm_rnn': xr.append(x[i])
		elif mt[i] == 'smc_lstm_rnn': xr.append(x[i]+0.03)
		elif mt[i] == 'he_lstm_rnn': xr.append(x[i]+0.06)
		else: print("{mt} is an invalid model type".format(mt=mt[i]))

	return xr, y, c


def save_plots(train_loss, valid_loss, train_acc, valid_acc, model_type, time, period):
	"""
	Function to save the loss plots and time plot to disk.
	"""
	today = date.today()
	date_format = today.strftime("%m-%d-%y")

	loss_file = join(os.getcwd(),"outputs",model_type+"_loss_"+date_format+".png")
	acc_file = join(os.getcwd(),"outputs",model_type+"_acc_"+date_format+".png")
	time_file = join(os.getcwd(),"outputs","train_times.png")
	csv_file = join(os.getcwd(),"outputs","times.csv")

	# loss plots
	plt.figure(figsize=(10,7))
	plt.plot(train_loss, color='#a4d5dc', linestyle='-', label='train loss')
	plt.plot(valid_loss, color='#b5b9ff', linestyle='-', label='validation loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Epoch Loss for '+str(model_type)+' Training ('+date_format+')')
	plt.legend()
	plt.savefig(loss_file)

	# accuracy plots
	plt.figure(figsize=(10,7))
	plt.plot(train_acc, color='#a4d5dc', linestyle='-', label='train accuracy')
	plt.plot(valid_acc, color='#b5b9ff', linestyle='-', label='validation accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.title('Epoch Accuracy for '+str(model_type)+' Training ('+date_format+')')
	plt.legend()
	plt.savefig(acc_file)

	# add training time to csv file
	save_time_to_csv(csv_file, model_type, time, period)

	# time plot
	xr, y, c = get_csv_data(csv_file)
	plt.figure(figsize=(10,7))
	plt.scatter(xr, y, c=c, s=100)
	plt.xticks(np.arange(0, period, step=1))
	plt.xlabel('Aggregation Period')
	plt.ylabel('Training Time (min)')
	plt.title('Training Times for Different Model Types')

	# create time plot legend
	lstm = mp.Patch(color='#baf0f5', label='lstm_rnn')
	dp = mp.Patch(color='#e1cff4', label='dp_lstm_rnn')
	smc = mp.Patch(color='#a4bda2', label='smc_lstm_rnn')
	he = mp.Patch(color='#f5d1cb', label='he_lstm_rnn')
	plt.legend(bbox_to_anchor=(1.05,1.0), loc='upper left', handles=[lstm,dp,smc,he])

	plt.savefig(time_file, bbox_inches='tight')
