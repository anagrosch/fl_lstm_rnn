import os
import torch
import pickle
import csv
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join, exists, getsize

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
			print(f"\nSaving best model for epoch: {epoch}\n")
			torch.save({
				'epoch': epoch+1,
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
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'loss': criterion,
		}, 'outputs/final_model.pth')


def save_plots(train_loss, valid_loss, train_acc, valid_acc, model_type, time):
	"""
	Function to save the loss plots and time plot to disk.
	"""
	loss_file = join(os.getcwd(),"outputs",model_type+"_loss.png")
	acc_file = join(os.getcwd(),"outputs",model_type+"_acc.png")
	time_file = join(os.getcwd(),"outputs","train_times.png")
	csv_file = join(os.getcwd(),"outputs","times.csv")

	# loss plots
	plt.figure(figsize=(10,7))
	plt.plot(train_loss, color='#a4d5dc', linestyle='-', label='train loss')
	plt.plot(valid_loss, color='#b5b9ff', linestyle='-', label='validation loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig(loss_file)

	# accuracy plots
	plt.figure(figsize=(10,7))
	plt.plot(train_acc, color='#a4d5dc', linestyle='-', label='train accuracy')
	plt.plot(valid_acc, color='#b5b9ff', linestyle='-', label='validation accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.savefig(acc_file)

	# add training time to csv file
	with open(csv_file, 'a', newline='') as f:
		write = csv.writer(f)

		# add headers if csv file is empty
		if getsize(csv_file) == 0:
			write.writerow(["Model Type", "Training Time"])
		write.writerow([model_type, time])

	# time plot
	df = pd.read_csv(csv_file)
	x = df['Model Type']
	y = df['Training Time']
	plt.scatter(x, y, c='#8fb9a1', s=100, alpha=0.8)
	plt.xlabel('Model Type')
	plt.ylabel('Training Time')
	plt.title('Training Times for Different Model Types')
	plt.savefig(time_file)


def get_model(model, optimizer, path):
	"""
	Function to load model checkpoint.
	"""
	checkpoint = torch.load(path)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']

	return model, optimizer, epoch, loss


def save_params(model, optimizer):
	"""
	Function to get model layer weights and save each parameter to its own file
	"""
	model_path = join(os.getcwd(), "outputs", "best_model.pth")
	param_path = join(os.getcwd(), "outputs", "best_model_params.pkl")
	
	model, optimiser, epoch, loss = get_model(model, optimizer, model_path)
	
	param_dict = {}
	for name, param in model.named_parameters():
		param_dict[name] = param

	# write parameters to text file
	with open(param_path, 'wb') as f:
		pickle.dump(param_dict, f)
	

def update_params(model):
	"""
	Function to update model layer weights with aggregated parameters.
	"""
	param_path = join(os.getcwd(), "outputs", "best_model_params.pkl")

	with open(param_path, 'rb') as f:
		param_dict = pickle.load(f)

	for name, param in model.named_parameters():
		param = param_dict[name]
	print('\nModel parameters updated')

	return model

