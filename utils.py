import os
import torch
import pickle
import matplotlib.pyplot as plt

plt.style.use('ggplot')

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


def save_plots(train_loss, valid_loss):
	"""
	Function to save the loss and accuracy plots to disk.
	"""
	# loss plots
	plt.figure(figsize=(10,7))
	plt.plot(train_loss, color='orange', linestyle='-', label='train loss')
	plt.plot(valid_loss, color='red', linestyle='-', label='validation loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig('outputs/loss.png')


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


def get_params(model, optimizer):
	"""
	Function to get model layer weights, add to dictionary, and save to file
	"""
	model_path = os.path.join(os.getcwd(), "outputs", "best_model.pth")
	param_path = os.path.join(os.getcwd(), "outputs", "best_model_params.pkl")

	model, optimiser, epoch, loss = get_model(model, optimizer, model_path)
	
	param_dict = {}
	for name, param in model.named_parameters():
		param_dict[name] = param

	# write parameters to text file
	with open(param_path, 'wb') as f:
		pickle.dump(param_dict, f)

	

