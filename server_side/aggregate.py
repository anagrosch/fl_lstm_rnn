import os
import pickle
import shutil
from os.path import join, isfile, exists
from server import server_get, server_send

final_weights = join(os.getcwd(), "final_weights.pkl")

def get_params(path):
	"""
	Function to get parameters from each client file.
	"""
	curr_dir = os.getcwd()
	param_dir = join(curr_dir, "client_models")

	# check if client model directory exists
	if not exists(param_dir):
		raise Exception("Client model directory does no exist.")

	# get parameters from last aggregation round
	param_dict = get_dict(final_weights)

	file_count = 0
	for file in os.listdir(param_dir):
		if file.endswith('.pkl'):
			file_count += 1

	tmp_dict = {}
	addr_dict = {}
	for file in os.listdir(param_dir):
		if file.endswith('.pkl'):
			tmp_dict = get_dict(join(param_dir, file))

			# get client address from dictionary
			ip = tmp_dict.keys()[-1]
			addr_dict[ip] = tmp_dict[ip]

			# delete client address from dictionary
			tmp_dict.pop(ip)

			# aggregate params
			for weight in tmp_dict.keys():
				if weight in param_dict:
					param_dict[weight] = param_dict[weight] + tmp_dict[weight]/file_count
				if weight not in param_dict:
					param_dict[weight] = tmp_dict[weight]/file_count

	# write aggregated weights to local file
	with open(final_weights, 'wb') as f:
		pickle.dump(param_dict, f)

	# delete client weight files
	try:
		shutil.rmtree(param_dir)
	except:
		raise Exception("Error deleting client_models directory")

	return addr_dict


def get_dict(file_path):
	"""
	Function to get parameter dictionary from a file.
	"""
	if not is_file(file_path):
		dict = {}
	else:
		with open(file_path, 'rb') as f:
			dict = pickle.load(f)
	return dict
	

# get trained models from clients
server_get()

# aggregate weights
addr_dict = get_params()

# send aggregated weights
server_send(addr_dict, final_weights)
