import pandas as pd
import sys

def clean_data(filename, new_filename):
	df = pd.read_json(filename)
	columns_to_encode = ['description', 'display_address', 'street_address']
	for column in columns_to_encode:
		df[column] = df[column].apply(lambda value: value.encode('ascii', 'ignore').decode('ascii'))
	print("Original Shape: {0} by {1}".format(df.shape[0], df.shape[0]))
	# drop duplicate columns
	duplicate_columns = ['bathrooms', 'bedrooms', 'building_id', 'description', 'display_address', 'latitude', 'longitude', 'manager_id', 'price', 'street_address']
	df_deduped = df.drop_duplicates(duplicate_columns)
	# drop price outliers
	df_deduped = df_deduped[(df_deduped['price'] > 50) & (df_deduped['price'] < 1000000)]
	print("New Shape: {0} by {1}".format(df_deduped.shape[0], df_deduped.shape[0]))
	df_deduped.to_csv(new_filename)
	print("Complete")

filename = sys.argv[1]
new_filename = sys.argv[2]
clean_data(filename, new_filename)