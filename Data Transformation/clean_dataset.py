import pandas as pd
import sys

def clean_data(filename, new_filename):
	df = pd.read_json(filename)
	print("Original Shape: {0}".format(df.shape))
	# drop duplicate columns
	duplicate_columns = ['bathrooms', 'bedrooms', 'building_id', 'description', 'display_address', 'latitude', 'longitude', 'manager_id', 'price', 'street_address']
	df_deduped = df.drop_duplicates(duplicate_columns)
	# drop price outliers
	df_deduped = df_deduped[(df_deduped['price'] > 50) & (df_deduped['price'] < 1000000)]
	columns_to_encode = ['description', 'display_address', 'street_address']
	for column in columns_to_encode:
		df_deduped[column] = df_deduped[column].apply(lambda value: [value.encode('ascii', 'ignore')])
	print("New Shape: {0}".format(df_deduped.shape))
	df_deduped.to_csv(new_filename, index = False)
	print("Complete")

filename = sys.argv[1]
new_filename = sys.argv[2]
clean_data(filename, new_filename)