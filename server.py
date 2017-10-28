from flask import Flask, Response, request, jsonify
import parser, ml
from flask_cors import CORS
from flask_csp.csp import csp_header
app = Flask(__name__)

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response

# data = np.loadtxt('data.csv', delimiter=",", dtype='str')

# bad_websites = {}
# bad_websites = ml.buildWebsiteSet(bad_websites, data)

# data = data[:, 1:]

# clf = KNeighborsClassifier(n_neighbors=2)
# ml.train(clf, data)

@app.route("/")
def hello():
	return "Hello world!"

@app.route("/check_url", methods=['POST'])
def check_url():
	request.get_data()
	json = request.json
	url = json['link']
	tweet = json['id']
	rating = ml.getScore(clf, bad_websites, ml.makeArray(parser.parse(url)))
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

# fakenews = np.loadtxt("dataset.csv", delimiter=",", dtype='str')
# col = np.zeros((fakenews.shape[0],1))
# fakenews = np.append(fakenews,col,axis=1)
# for i in range(fakenews.shape[0]):
# 	title = fakenews[i,1]
# 	fakenews[i,-1] = sum(1.0 for c in title if c.isupper())/sum(1 for c in title if c.isalpha())	
# X = fakenews[:, -1].reshape(6,1)
# y = fakenews[:,-2]
# print(X)
# print(y)
# clf = KNeighborsClassifier(n_neighbors=2)
# clf.fit(X, y)
# print(clf.predict(np.array([[0.2]])))

if __name__ == "__main__":
	app.run(threaded=True)