import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from os.path import join

plt.style.use('bmh')

CSV_FILE = join(os.getcwd(),"outputs","server_times.csv")

def save_times(get_time='', aggr_time='', send_time=''):
	"""
	Function to save transfer times to a csv file.
	Value of -1 means specific function was not applied.
	"""
	curr_date = date.today()
	date_format = curr_date.strftime("%m/%d/%y")

	with open(csv_file, 'a', newline='') as f:
		write = csv.writer(f)

		# add headers if csv file is empty
		if getsize(csv_file) == 0:
			write.writerow(["End Date","Collection (s)","Aggregation (s)","Redistribution (s)"])
		write.writerow([date_format, get_time, aggr_time, send_time])


def save_time_plots():
	"""
	Function to save server's get_time, aggr_time, and send_time.
	"""
	transfer_plot = join(os.getcwd(),"outputs","transfer_times.png")

	df = pd.read_csv(CSV_FILE)
	x = df['End Date']
	yc = df['Collection (s)']
	ya = df['Aggregation (s)']
	yr = df['Redistribution (s)']

	fig, (ax0,ax1,ax2) = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(10,9))
	fig.suptitle('Server/Client Weight Transfer Times')
	fig.tight_layout(pad=4.0)

	# collection plot
	ax0.scatter(x, yc, c='#f55c7a', s=100)
	ax0.set_ylabel('Time Taken (s)')
	ax0.set_title('Collection Times')

	# aggregation plot
	ax1.scatter(x, ya, c='#f68c70', s=100)
	ax1.set_ylabel('Time Taken (s)')
	ax1.set_title('Aggregation Times')

	# redistribution plot
	ax2.scatter(x, yr, c='#f6bc66', s=100)
	ax2.set_xticklabels(x, rotation=45)
	ax2.set_xlabel('Date Run')
	ax2.set_ylabel('Time Taken (s)')
	ax2.set_title('Redistribution Times')

	plt.savefig(transfer_plot, bbox_inches='tight')

