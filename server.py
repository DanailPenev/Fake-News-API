from flask import Flask, Response, request, jsonify
import parser, ml, keys, bot
from flask_cors import CORS
from flask_csp.csp import csp_header
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.externals import joblib
import botometer
import random
app = Flask(__name__)

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
	print("Request received")
	request.get_data()
	rating = {}
	json = request.json
	url = json['link']
	tweet = json['id']
	try:
		user = json['user']
		userat = '@' + user
		user_score = bot.test_user(userat)['scores']['english']
		rating['user_score'] = user_score
		rating['user'] = user
	except Exception as e:
		print(e)
	try:
		rating['score'] = ml.test(clf, parser.parse(url), count_vectorizer)[0,0]
	except:
		rating['score'] = 0.5
	rating['id'] = tweet
	rating['link'] = url
	print(json, rating)
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

@app.route("/test_bot", methods=['POST'])
def test():
	request.get_data()
	rating = {}
	json = request.json
	print(json)
	url = json['link']
	tweet = json['id']
	try:
		user = json['user']
		print(user)
		userat = '@' + user
		print(userat)
		user_score = bot.test_user(userat)['scores']['english']
		print(user_score)
		rating['user_score'] = user_score
		rating['user'] = user
		print(rating)
	except Exception as e:
		print(e)
	try:
		rating['score'] = ml.test(clf, parser.parse(url), count_vectorizer)[0,1]
	except:
		rating['score'] = 0.5
	if rating['score'] > 0.95:
		rating['score'] = random.uniform(0.65,0.85)
	rating['id'] = tweet
	rating['link'] = url
	print(rating)
	return jsonify(rating)

def is_url_whitelisted(url):
	url = url.split['/'][2]
	print(url)

if __name__ == "__main__":
	app.run(threaded=True)