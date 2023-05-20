# Format csv file for dataloader

import os
import sys
import csv
import pandas as pd

def select_column(filename):
	"""
	Function to select column with data.
	"""
	df = pd.read_csv(filename)
	print("csv columns: {c}".format(c=list(df.columns)))
	data_col = input("Enter column with data to rename: ")

	return data_col, df
	

def format_data_file(filename, column, df):
	"""
	Function to set title of column with data to 'text'.
	"""
	df.rename(columns={column:'nlp_data'}, inplace=True)
	df.to_csv(filename)
	print("Data column title set to 'nlp_data'")


def valid_file(filename):
	"""
	Function to check that the file input is a csv file.
	"""
	ext = os.path.splitext(filename)[1][1:]
	if ext != 'csv':
		parser.error('Invalid file type. Does not end in csv')
	return filename


# main function
if len(sys.argv) != 2:
	print("Missing argument")
	print("Run <python3 prepare_data.py [filepath]>")
	raise SystemExit(1)

filename = sys.argv[1]

column, df = select_column(filename)
format_data_file(filename, column, df)
