import os
import pickle
import shutil
import torch
from os.path import join, exists

final_weights = join(os.getcwd(),"outputs","final_weights.pkl")

def aggr_params():
	"""
	Function to get parameters from each client file.
	"""
	print('\n--------------------------')
	print('Aggregating client weights')
	print('--------------------------\n')

	curr_dir = os.getcwd()
	param_dir = join(curr_dir, "client_models")

	# check if client model directory exists
	if not exists(param_dir):
		raise Exception("Client model directory does no exist.")

	file_count = 0
	for file in os.listdir(param_dir):
		if file.endswith('.pkl'):
			file_count += 1

	# get parameters from last aggregation round
	param_dict, file_count = get_dict(final_weights,file_count=file_count)

	tmp_dict = {}
	for file in os.listdir(param_dir):
		if file.endswith('.pkl'):
			tmp_dict, file_count = get_dict(join(param_dir, file),file_count=file_count)

			# aggregate params
			for weight in tmp_dict.keys():
				if weight in param_dict:
					# set weight tensor to equal sizes
					tmp_dict[weight], param_dict[weight] = resize_tensor(tmp_dict[weight], param_dict[weight])
					param_dict[weight] += tmp_dict[weight]
				if weight not in param_dict:
					param_dict[weight] = tmp_dict[weight]
				
			print("Aggregated weights for file: {file}".format(file=file))

	# write aggregated weights to local file
	with open(final_weights, 'wb') as f:
		pickle.dump(param_dict, f)
	print('\nAggregated results saved to /outputs/final_weights.pkl')

	# delete client weight files
	try:
		shutil.rmtree(param_dir)
		print('Client data deleted')
	except:
		raise Exception("Error deleting client_models directory")


def get_dict(file_path, file_count=1):
	"""
	Function to get parameter dictionary from a file and prepare weights for aggregation.
	"""
	if not exists(file_path):
		dict = {}
	else:
		with open(file_path, 'rb') as f:
			dict = pickle.load(f)

		# prepare weights for aggregation
		file_count += 1
		for weight in dict.keys():
			dict[weight] = dict[weight] / file_count
	return dict, file_count


def resize_tensor(tensor1, tensor2):
	"""
	Function to set two tensors to the same size.
	Fills smaller tensor (tensor1) with random initializer.
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


