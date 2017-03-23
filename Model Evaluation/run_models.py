
import pandas as pd
import numpy as np
import sys
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from ast import literal_eval
import datetime
import time
from xgboost import XGBClassifier
from nltk.stem.snowball import SnowballStemmer

TEST_SIZE = .3
RANDOM_SEED = 42
MAX_FEAT = 1000

LOG_PARAMETERS = {'solver' : ['newton-cg'], 'multi_class' : ['multinomial'], 
'class_weight' : ['balanced', None], 'C' : [10**c for c in range(-3, 4)]}

RF_PARAMETERS = {'n_estimators': [500],'max_features' : ['auto', 'sqrt', 'log2'], 'min_samples_leaf' : [1, 10, 100],
'random_state': [42], 'class_weight' : [None, 'balanced', 'balanced_subsample']}

XGB_PARAMETERS = {'learning_rate': [.01, .05, .1, .2, .5], 'max_depth' : [5, 10], 'n_estimators': [500],
'objective' : ['multi:softprob']}

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

def stem_data(dataset):
	print("Stemming Description Text...")
	def stemming(sentence):
		new_sentence = []
		stemmer = SnowballStemmer('english')
		for word in sentence:
			stem = stemmer.stem(word)
			new_sentence.append(stem)
		stem_sentence = " ".join(new_sentence)
		return stem_sentence
	dataset['stem_description'] = dataset['description'].str.lower().str.split(' ').apply(stemming)
	print("Complete")
	return dataset

def tf_idf(X_train, X_test):
	print("Transforming Description Text...")
	# split dataframes into description and non-desciption sets
	X_train_no_desc = X_train.drop('stem_description', axis = 1).reset_index()
	X_train_desc = X_train['stem_description']
	X_test_no_desc = X_test.drop('stem_description', axis = 1).reset_index()
	X_test_desc = X_test['stem_description']
	# create count vector
	stp_wrds = ENGLISH_STOP_WORDS.union(['york'])
	count_vector = CountVectorizer(max_df = .2, stop_words = stp_wrds, max_features = MAX_FEAT,ngram_range = (1,3))
	X_train_counts = count_vector.fit_transform(X_train_desc)
	# get column names excluding those words that are just numbers
	feature_names = count_vector.get_feature_names()
	non_numeric_counts = [i for i in range(len(feature_names)) if feature_names[i].isdigit() == False]
	column_names = np.array(feature_names)[non_numeric_counts]
	X_train_counts = X_train_counts[:, non_numeric_counts]
	# create tf-idf tranformer
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	# transform test set
	X_test_counts = count_vector.transform(X_test_desc)
	X_test_counts = X_test_counts[:, non_numeric_counts]
	X_test_tfidf = tfidf_transformer.transform(X_test_counts)
	# convert and concatenate dataframes
	X_train_tfidf = pd.DataFrame(X_train_tfidf.toarray(), columns = column_names).reset_index()
	X_test_tfidf = pd.DataFrame(X_test_tfidf.toarray(), columns = column_names).reset_index()
	X_train = X_train_no_desc.join(X_train_tfidf, lsuffix = 'pred', rsuffix = 'word')
	X_test = X_test_no_desc.join(X_test_tfidf, lsuffix = 'pred', rsuffix = 'word')
	print("Train Predictor Shape: {0}".format(X_train.shape))
	print("Test Predictor Shape: {0}".format(X_test.shape))
	return X_train, X_test

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
	feature_importance = model.feature_importances_ if model_name != 'log' else None
	cm = confusion_matrix(y_test, predictions, labels = ['low', 'medium', 'high'])
	values = [{'timestamp': datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M'), 'model' : model_name,
			'parameters' : model_parameters, 'log_loss' : log_loss_score, 'accuracy' : accuracy, 'confusion_matrix' : cm,
			'predictors' : list(X_test.columns), 'feature_importances' : feature_importance}]
	print("Log-Loss: {0:.4f}".format(log_loss_score))
	print("Accuracy: {0:.4f}".format(accuracy))
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

def run_models(dataset, model_performance_dataset, chosen_models_dataset, models):#,tf_idf_parameters):
	print("Loading data...")	
	df_full = pd.read_json(dataset)
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
	df_full = stem_data(df_full)
	print("Splitting data...")
	X_train, X_test, y_train, y_test = split_data(df_full)
	# if tf_idf_parameters[0]:
	X_train, X_test = tf_idf(X_train, X_test)
	if 'log' in models:
		print("Cross-Validating Log Model...")
		log_time_start = time.time()
		log_model = LogisticRegression()
		log_model_performance, log_model_score = evaluate_model('log', log_model, LOG_PARAMETERS, X_train, X_test, y_train, y_test)
		df_model_performance = df_model_performance.append(log_model_performance, ignore_index = True)
		df_chosen_models = df_chosen_models.append(log_model_score, ignore_index = True)
		log_time_end = time.time()
		print("Log Model Complete! Time Elapsed - {0} Seconds".format(log_time_end - log_time_start))
	if 'rf' in models:
		print("Cross-Validating Random Forest Model...")
		rf_time_start = time.time()
		rf_model = RandomForestClassifier()
		rf_model_performance, rf_model_score = evaluate_model('rf', rf_model, RF_PARAMETERS, X_train, X_test, y_train, y_test)
		df_model_performance = df_model_performance.append(rf_model_performance, ignore_index = True)
		df_chosen_models = df_chosen_models.append(rf_model_score, ignore_index = True)
		rf_time_end = time.time()
		print("RF Model Complete! Time Elapsed - {0} Seconds".format(rf_time_end - rf_time_start))
	if 'xgb' in models:
		print("Cross-Validating XGBoost Model...")
		xgb_time_start = time.time()
		xgb_model = XGBClassifier()
		xgb_model_performance, xgb_model_score = evaluate_model('xgb', xgb_model, XGB_PARAMETERS, X_train, X_test, y_train, y_test)
		df_model_performance = df_model_performance.append(xgb_model_performance, ignore_index = True)
		df_chosen_models = df_chosen_models.append(xgb_model_score, ignore_index = True)
		xgb_time_end = time.time()
		print("XGB Model Complete! Time Elapsed - {0} Seconds".format(xgb_time_end - xgb_time_start))
	df_model_performance.to_csv(model_performance_dataset, index = False)
	df_chosen_models.to_csv(chosen_models_dataset, index = False)
	print("Complete")

dataset = sys.argv[1]
model_performance_dataset = sys.argv[2]
chosen_models_dataset = sys.argv[3]
models = sys.argv[4]
#tf_idf_parameters = literal_eval(sys.argv[5])
run_models(dataset, model_performance_dataset, chosen_models_dataset, models)#, tf_idf_parameters)
