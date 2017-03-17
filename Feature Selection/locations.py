import sys
import json
import requests
import pandas
import sys


def walkscore(lat,lng,address):
	api_key = '67f61200a3b427ee5e01e62c974c4ea7'
	url = 'http://api.walkscore.com/score?'
	params = {
	'format': 'json',
	'lat':str(lat),
	'lon':str(lng),
	'wsapikey':api_key,
	'address':address,
	'transit':str(1),
	'bike':str(1)
	}
	r = requests.get(url.encode('utf-8'),params = params)
	data = r.json()
	return data

def get_response(start_lat, start_lng, end_lat, end_lng):
    url = 'https://api.uber.com/v1.2/estimates/price?start_latitude={0}&start_longitude={1}&end_latitude={2}&end_longitude={3}'.format(start_lat, start_lng, end_lat, end_lng)
    api_token ='Rmah_eOq6rBjc-UjQZtszGJX3JAaqhf620WaHYgM'
    authorization_token = 'Token ' + api_token
    response = requests.get(url, headers={'Authorization': authorization_token, 'Accept-Language':'en_US','Content-Type':'application/json'})
    return response

start_time = time.time()
x = get_response(start_lat, start_lng, end_lat, end_lng)
status_code = x.status_code
if status_code == 200:
    y = json.loads(x.text)
    uber_pool = y['prices'][0]
    uber_x = y['prices'][1]
    z = [status_code, uber_pool['distance'], uber_pool['duration'], uber_pool['high_estimate'], uber_pool['low_estimate'],
    uber_x['distance'], uber_x['duration'], uber_x['high_estimate'], uber_x['low_estimate']]
else:
    z = [status_code, None, None, None, None, None, None, None, None]
end_time = time.time()
time.sleep(min(2.5 - (end_time - start_time), 0))

if __name__ == '__main__':
	    	