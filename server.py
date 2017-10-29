from flask import Flask, Response, request, jsonify
import parser, ml, keys
from flask_cors import CORS
from flask_csp.csp import csp_header
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.externals import joblib
import botometer
app = Flask(__name__)

mashape_key = keys.get_key()
twitter_app_auth = keys.get_twitter_auth()

bom = botometer.Botometer(wait_on_ratelimit=True,
                          mashape_key=mashape_key,
                          **twitter_app_auth)

# # Initialize the `count_vectorizer` 
count_vectorizer = joblib.load('vectorizer.pkl')

# initialize classifier
clf = joblib.load('classifier.pkl') 

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response

@app.route("/")
def hello():
	return "Hello world!"

@app.route("/check_url", methods=['POST'])
def check_url():
	request.get_data()
	rating = {}
	json = request.json
	print(json)
	url = json['link']
	tweet = json['id']
	try:
		user = json['user']
		userat = '@' + user
		user_score = test_bot(userat)
		rating['user_score'] = user_score
		rating['user'] = user
	except:
		pass
	rating['score'] = ml.test(clf, parser.parse(url), count_vectorizer)[0,1]
	rating['id'] = tweet
	rating['link'] = url
	print(rating)
	return jsonify(rating)

@app.route("/vanko_mock", methods=['POST'])
@csp_header()
def vanko_mock():
	request.get_data()
	json = request.json
	url = json['link']
	tweet = json['id']
	print(url,tweet)
	return jsonify({'id':tweet, 'link': url, 'score': 0.05})

def test_bot(user):
	result = bom.check_account('user')
	result = result['scores']['english']
	print(result)
	return "hui"

if __name__ == "__main__":
	app.run(threaded=True)