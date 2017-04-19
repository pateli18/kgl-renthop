
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss,precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from ast import literal_eval
import datetime
import time
import re

RF_PARAMETERS = {'n_estimators': [500],'max_features' : ['auto', 'sqrt', 'log2'], 'min_samples_leaf' : [1, 10, 100],
'random_state': [42], 'class_weight' : [None, 'balanced', 'balanced_subsample']}

object_cols = ['building_id','created','description','display_address','features','interest_level','manager_id','photos','street_address']

def replace(string):
	if isinstance(string,basestring):
		string = re.sub('(\[|\]|\ )','',string)
		return string
	else:
		return string

def split_data(dataset):
	X = dataset.drop(object_cols, axis = 1)
	X = X.applymap(replace)
	y = dataset['interest_level']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
	# print shape of splits
	print("Train Predictor Shape: {0}".format(X_train.shape))
	print("Train Outcome Shape: {0}".format(y_train.shape))
	print("Test Predictor Shape: {0}".format(X_test.shape))
	print("Test Outcome Shape: {0}".format(y_test.shape))
	# print outcome splits
	y_train_split = {key: value*1.0/y_train.shape[0] for key, value in zip(y_train.value_counts().keys(),y_train.value_counts())}
	y_test_split = {key: value*1.0/y_test.shape[0] for key, value in zip(y_test.value_counts().keys(),y_test.value_counts())}
	for key in y_train_split.keys():
		print("{0}: {1:.4f} | {2:.4f}".format(key, y_train_split[key], y_test_split[key]))
	return X_train, X_test, y_train, y_test

def feature_selection(X_train,Y_train):
	k_range = range(1,X_train.shape[1])
	k_feature_test = []
	for k in k_range:
		print k
		selector = SelectKBest(k=k)
		features_train = selector.fit_transform(X_train,Y_train)
		clf = RandomForestClassifier(n_estimators=500,max_features='log2',min_samples_leaf=10,random_state=42,class_weight='balanced_subsample')
		#clf = GridSearchCV(rf,RF_PARAMETERS,scoring="neg_log_loss",cv=5,error_score=100)	
		clf.fit(features_train,Y_train)
		#log_loss = log_loss(Y_train,clf.predict(features_train))
		precision = precision_score(Y_train,clf.predict(features_train),average='weighted')
		recall = recall_score(Y_train,clf.predict(features_train),average='weighted')
		scores = np.array([k,precision,recall])
		k_feature_test.append(scores)
	k_feature_test = np.asarray(k_feature_test)
	df = pd.DataFrame(k_feature_test)
	df.to_csv('k_feature_test.csv',index=False)
	#plt.plot(k_feature_test[:,0],k_feature_test[:,1],label='Log Loss')
	plt.plot(k_feature_test[:,0],k_feature_test[:,2],label='Precision')
	plt.plot(k_feature_test[:,0],k_feature_test[:,3],label='Recall')
	plt.xlabel('Number of K Features')
	plt.ylabel('Scores')
	legend = plt.legend(loc='upper center')
	plt.show()
	plt.savefig('features.pdf')

def run_models(dataset):
	print("Loading data...")	
	df_full = pd.read_csv(dataset)
	print("Splitting data...")
	X_train, X_test, y_train, y_test = split_data(df_full)
	feature_selection(X_train,y_train)
	print("Complete")

dataset = sys.argv[1]
run_models(dataset)
