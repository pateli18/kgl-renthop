import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import sys
from ast import literal_eval

def fill(value, lat, lng, classifier):
	if pd.isnull(value):
		return classifier.predict([lat, lng])
	else:
		return value

def fill_knn(dataset, columns_to_fill):
	df = pd.read_csv(dataset)
	df['std_lat'] = StandardScaler(copy = True).fit_transform(df['latitude'])
	df['std_lng'] = StandardScaler(copy = True).fit_transform(df['longitude'])
	for column in columns_to_fill:
		existing = df[~df[column].isnull()]
		knn_clf = KNeighborsRegressor()
		knn_clf.fit(existing[['std_lat', 'std_lng']], existing[column])
		df[column] = df.apply(lambda row: fill(row[column], row['std_lat'],
			row['std_lng'], knn_clf), axis = 1)
	df = df.drop(['std_lat', 'std_lng'], axis = 1)
	df.to_csv(dataset, index = False)
	for column in columns_to_fill:
		print("{0} : {1}".format(column, df[column].isnull().sum()))
	print("Complete")

dataset = sys.argv[1]
columns_to_fill = sys.argv[2].split(',')
fill_knn(dataset, columns_to_fill)