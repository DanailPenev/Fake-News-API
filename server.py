from flask import Flask, Response, request, jsonify
import parser, ml
from flask_cors import CORS
from flask_csp.csp import csp_header
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.externals import joblib
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
	request.get_data()
	json = request.json
	url = json['link']
	tweet = json['id']
	rating = {}
	rating['score'] = ml.test(clf, parser.parse(url), count_vectorizer)[0,1]
	rating['id'] = tweet
	rating['link'] = url
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

if __name__ == "__main__":
	app.run(threaded=True)