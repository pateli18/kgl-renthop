import pandas as pd
import numpy as np
from ast import literal_eval
import sys
import datetime

def add_features(load_filepath, save_filepath):
	df = pd.read_csv(load_filepath)
	# add feature lengths
	df['description_length'] = df['description'].apply(lambda x: len(literal_eval(x)[0]))
	df['num_photos'] = df['photos'].apply(lambda x: len(x))
	df['num_features'] = df['features'].apply(lambda x: len(x))
	# add non-linear prices
	df['price_sq'] = df['price'].apply(lambda x: x**2)
	df['price_log'] = df['price'].apply(lambda x: np.log(x))
	# add date features
	df['created'] = pd.to_datetime(df['created'],infer_datetime_format = True)
	df['month'] = df['created'].apply(lambda date: date.month)
	df['day'] = df['created'].apply(lambda date: date.day)
	df['dayofweek'] = df['created'].apply(lambda date: date.dayofweek)
	df['hour'] = df['created'].apply(lambda date: date.hour)
	df.to_csv(save_filepath, index = False)
	print("Complete")

load_filepath = sys.argv[1]
save_filepath = sys.argv[2]
add_features(load_filepath, save_filepath)