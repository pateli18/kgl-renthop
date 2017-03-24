import requests
import pandas as pd
import sys
import json
from ast import literal_eval

def get_response(index, description, total):
	url = 'https://community-sentiment.p.mashape.com/text/'
	api_token ='17xVhg4lDDmshfCmbV9rKbAr5vCCp1IxbmWjsnv799zJwEUifM'
	description = 'txt={0}'.format(description)
	try:
		response = requests.post(url, headers={'X-Mashape-Key': api_token,
			'Accept':'application/json',
			'Content-Type':'application/x-www-form-urlencoded'}, data = description)
		status_code = response.status_code
		data = json.loads(response.text)
		values = [status_code, data['result']['confidence'], data['result']['sentiment']]
		print(status_code)
	except:
		print("Error")
		values = [None, None, None]
	print("{0} out of {1} | {2:.2f}% Complete".format(index, total, index*1./total * 100))
	return values

def get_data(dataset_filepath, sentiment_filepath):
	try:
		df = pd.read_csv(sentiment_filepath)
	except IOError:
		df = pd.read_csv(dataset_filepath)
		df['listing_id'] = df.index
		df = df[['listing_id','description']]
		df['sentiment_status_code'] = df['description'].apply(lambda x: None)
	ids_pulled = [int(i[0]) for i in df[['listing_id', 'sentiment_status_code']].values if pd.notnull(i[1])]
	ids_to_pull = [int(i[0]) for i in df[['listing_id', 'sentiment_status_code']].values if pd.isnull(i[1])]
	total = len(ids_to_pull)
	counter = 1
	while len(ids_to_pull) > 0:
		listing_id = ids_to_pull[0]
		description = literal_eval(df.iloc[listing_id]['description'])[0]
		values = get_response(counter, description, total)
		df.set_value(listing_id,'sentiment_status_code', values[0])
		df.set_value(listing_id,'sentiment_confidence', values[1])
		df.set_value(listing_id,'sentiment', values[2])
		df.to_csv(sentiment_filepath, index = False)
		ids_to_pull.remove(listing_id)
		counter = counter + 1
	print(df['sentiment_status_code'].value_counts())
	print(df['sentiment_status_code'].isnull().sum())
	print("Complete")

dataset_filepath = sys.argv[1]
sentiment_filepath = sys.argv[2]
get_data(dataset_filepath, sentiment_filepath)