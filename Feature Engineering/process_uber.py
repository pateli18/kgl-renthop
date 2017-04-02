import pandas as pd
from ast import literal_eval
import sys

def process_uber(dataset_list, train_set):
	train_df = pd.read_csv(train_set)
	print(train_df.shape)
	datasets = dataset_list.split(',')
	names = [dataset.replace("_uber.csv", '').replace('datasets/', '') for dataset in datasets]
	dfs = [pd.read_csv(dataset) for dataset in datasets]
	for i,df in enumerate(dfs):
		df = df.convert_objects(convert_numeric=True)
		df[names[i] + ' cost'] = df['uber_x_low_estimate']
		df[names[i] + ' distance'] = df['uber_x_duration']
		train_df = pd.concat([train_df, df[[names[i] + ' cost', names[i] + ' distance']]], axis = 1)
	print(train_df.shape)
	train_df.to_csv(train_set, index = False)

dataset_list = sys.argv[1]
train_set = sys.argv[2]
process_uber(dataset_list, train_set)