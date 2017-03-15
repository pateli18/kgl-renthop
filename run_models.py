
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import datetime
import time

TEST_SIZE = .3
RANDOM_SEED = 42

LOG_PARAMETERS = {'solver' : ['newton-cg', 'lbfgs'], 'multi_class' : ['ovr', 'multinomial'], 
'class_weight' : ['balanced', None], 'C' : [10**c for c in range(-3, 4)]}

RF_PARAMETERS = {'n_estimators': [10, 50, 100, 250, 500], 'criterion': ['gini', 'entropy'],
'max_features' : ['auto', 'sqrt', 'log2'], 'min_samples_leaf' : [1, 10, 20, 50, 100],
'random_state': [42], 'class_weight' : [None, 'balanced', 'balanced_subsample']}

def split_data(dataset):
	X = dataset.drop('interest_level', axis = 1)
	y = dataset['interest_level']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE, random_state = RANDOM_SEED)
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

def parse_grid_scores(grid_scores, model_name, predictors):
	values = []
	for model in grid_scores:
		values.append({'timestamp': datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M'), 'model' : model_name,
			'parameters' : str(model[0]), 'score' : model[1], 'score_std' : np.std(model[2]), 'cv_folds' : len(model[2]),
			'predictors' : predictors})
	return pd.DataFrame(values)

def chosen_model_score(model, model_name, model_parameters, X_test, y_test):
	probabilities = model.predict_proba(X_test)
	predictions = model.predict(X_test)
	log_loss_score = log_loss(y_test, probabilities)
	accuracy = accuracy_score(y_test, predictions)
	cm = confusion_matrix(y_test, predictions, labels = ['low', 'medium', 'high'])
	values = [{'timestamp': datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M'), 'model' : model_name,
			'parameters' : model_parameters, 'log_loss' : log_loss_score, 'accuracy' : accuracy, 'confusion_matrix' : cm,
			'predictors' : list(X_test.columns)}]
	return pd.DataFrame(values)

def cross_validate_model(model, parameters, X_train, y_train):
	model_cv = GridSearchCV(model, parameters, scoring = "log_loss", cv = 5, error_score = 100)
	model_cv.fit(X_train, y_train)
	return model_cv

def evaluate_model(model_name, model, parameters, X_train, X_test, y_train, y_test):
	model_cv = cross_validate_model(model, parameters, X_train, y_train)
	model_performance = parse_grid_scores(model_cv.grid_scores_, model_name, list(X_train.columns))
	model_score = chosen_model_score(model_cv.best_estimator_, model_name, str(model_cv.best_params_), X_test, y_test)
	return model_performance, model_score

def run_models(dataset, model_performance_dataset, chosen_models_dataset, models = ['log', 'rf', 'xbg']):
	print("Loading data...")	
	df_full = pd.read_csv(dataset)
	try:
		df_model_performance = pd.read_csv(model_performance_dataset)
	except IOError:
		df_model_performance = pd.DataFrame(columns = ['timestamp', 'model', 'parameters', 'score', 'score_std',
			'cv_folds', 'predictors'])
	try:
		df_chosen_models = pd.read_csv(chosen_models_dataset)
	except IOError:
		df_chosen_models = pd.DataFrame(columns = ['timestamp', 'model', 'parameters', 'log_loss', 'accuracy',
			'confusion_matrix', 'predictors'])
	print("Splitting data...")
	X_train, X_test, y_train, y_test = split_data(df_full)
	if 'log' in models:
		print("Cross-Validating Log Model...")
		log_model = LogisticRegression()
		log_model_performance, log_model_score = evaluate_model('log', log_model, LOG_PARAMETERS, X_train, X_test, y_train, y_test)
		df_model_performance.append(log_model_performance, ignore_index = True).to_csv(model_performance_dataset, index = False)
		df_chosen_models.append(log_model_score, ignore_index = True).to_csv(chosen_models_dataset, index = False)
		print("Log Model Complete!")
	if 'rf' in models:
		print("Cross-Validating Random Forest Model...")
		rf_model = RandomForestClassifier()
		rf_model_performance, rf_model_score = evaluate_model('rf', rf_model, RF_PARAMETERS, X_train, X_test, y_train, y_test)
		df_model_performance.append(rf_model_performance, ignore_index = True).to_csv(model_performance_dataset, index = False)
		df_chosen_models.append(rf_model_score, ignore_index = True).to_csv(chosen_models_dataset, index = False)
	print("Complete")
