import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys

def transform_sentiment(sentiment, confidence, pos_scaler, neg_scaler):
	if sentiment == 'Positive':
		return pos_scaler.transform([confidence])[0]
	elif sentiment == 'Negative':
		return -(neg_scaler.transform([confidence])[0])
	else:
		return 0

def process_sentiment(input_filepath, output_filepath):
	df = pd.read_csv(input_filepath)
	dummies = pd.get_dummies(df['sentiment'])
	df_pos = df[df['sentiment'] == 'Positive']['sentiment_confidence']
	df_neg = df[df['sentiment'] == 'Negative']['sentiment_confidence']
	pos_scaler = MinMaxScaler()
	pos_scaler.fit(df_pos)
	neg_scaler = MinMaxScaler()
	neg_scaler.fit(df_neg)
	df['sentiment_score'] = df.apply(lambda row: transform_sentiment(row['sentiment'], row['sentiment_confidence'], pos_scaler, neg_scaler), axis = 1)
	df = df[['sentiment_score']]
	df = pd.concat([df, dummies], axis = 1)
	print('Sentiment Dataframe Size: {0}'.format(df.shape))
	train_df = pd.read_csv(output_filepath)
	print('Train Dataframe Size: {0}'.format(train_df.shape))
	train_df = pd.concat([train_df, df], axis = 1)
	print('Combined Dataframe Size: {0}'.format(train_df.shape))
	train_df.to_csv(output_filepath, index = False)
	print('Complete')

input_filepath = sys.argv[1]
output_filepath = sys.argv[2]
process_sentiment(input_filepath, output_filepath)