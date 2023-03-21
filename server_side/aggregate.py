import os
import pickle
import csv
import shutil
from datetime import date
from os.path import join, isfile, exists, getsize

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
	param_dict, file_count = get_dict(final_weights,file_count)

	tmp_dict = {}
	for file in os.listdir(param_dir):
		if file.endswith('.pkl'):
			tmp_dict, file_count = get_dict(join(param_dir, file),file_count)

			# aggregate params
			for weight in tmp_dict.keys():
				if weight in param_dict:
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


def get_dict(file_path, file_count):
	"""
	Function to get parameter dictionary from a file.
	"""
	if not exists(file_path):
		dict = {}
	else:
		with open(file_path, 'rb') as f:
			dict = pickle.load(f)

		# prepare weights for aggregation
		file_count += 1
		for weight in dict.keys():
			dict[weight] /= file_count
	return dict, file_count


def save_times(get_time=0, aggr_time=0, send_time=0):
	"""
	Function to save transfer times to a csv file.
	"""
	csv_file = join(os.getcwd(),"outputs","server_times.csv")
	curr_date = date.today()
	date_format = curr_data.strftime("%m/%d/%y")
	
	with open(csv_file, 'a', newline='') as f:
		write = csv.writer(f)
		
		# add headers if csv file is empty
		if getsize(csv_file) == 0:
			write.writerow(["End Date","Get Weights","Aggregate","Redistribute"])
		write.writerow([date_format, get_time, aggr_time, send_time])


