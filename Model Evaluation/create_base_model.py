import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import sys
import datetime
import time

TEST_SIZE = .3
RANDOM_SEED = 42

def create_base_model(dataset, model_performance_dataset, chosen_models_dataset):
	df = pd.read_csv(dataset)
	df_model_performance = pd.read_csv(model_performance_dataset)
	df_chosen_models = pd.read_csv(chosen_models_dataset)
	# split dataset
	X = df
	y = df['interest_level']
	_, _, _, y_test = train_test_split(X, y, test_size = TEST_SIZE, random_state = RANDOM_SEED)
	# create array of probabilities equal to train set splits
	probabilities = np.zeros((y_test.shape[0], 3))
	probabilities[:, 0] = .7
	probabilities[:, 0] = .22
	probabilities[:, 0] = .08
	# calculate log loss score
	log_loss_score = log_loss(y_test, probabilities)
	# calculate accuracy score
	predictions = y_test.apply(lambda x: 'low')
	accuracy = accuracy_score(y_test, predictions)
	# create confusion matrix
	cm = confusion_matrix(y_test, predictions, labels = ['low', 'medium', 'high'])
	# store base model in model performance dataset
	model_performance_values = [{'timestamp': datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M'), 'model' : 'base',
			'parameters' : None, 'score' : log_loss_score, 'score_std' : 0, 'cv_folds' : None, 'predictors' : None}]
	df_model_performance.append(pd.DataFrame(model_performance_values), ignore_index = True).to_csv(model_performance_dataset, index = False)
	# store base model in chosen model dataset
	chosen_model_values = [{'timestamp': datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M'), 'model' : 'base',
			'parameters' : None, 'log_loss' : log_loss_score, 'accuracy' : accuracy, 'confusion_matrix' : cm,
			'predictors' : None}]
	df_chosen_models = df_chosen_models.append(pd.DataFrame(chosen_model_values), ignore_index = True)
	df_chosen_models['confusion_matrix'] = df_chosen_models['confusion_matrix'].apply(lambda x: str(x).replace('\r', '\n'))
	df_chosen_models.to_csv(chosen_models_dataset, index = False)
	print("Complete")

dataset = sys.argv[1]
model_performance_dataset = sys.argv[2]
chosen_models_dataset = sys.argv[3]
create_base_model(dataset, model_performance_dataset, chosen_models_dataset)
