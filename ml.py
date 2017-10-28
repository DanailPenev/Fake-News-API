import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

emotional_words = {"breaking", "prayers", "hate", "hates", "love", "loves", "totally", "fantastic", "amazing", "busted", "fake", "false", "absolutely", "definitely", "incredible", "exceptional", "exceptionally", "outstanding", "fall", "falls", "shock", "shocks", "shame", "tremendous", "won’t believe", "will not believe", "can’t believe", "what happens next", "tears", "hot", "regrets", "stupid", "stupidity", "don’t", "simple", "resist", "experts", "scientifically", "stuff", "sums up", "perfectly", "perfect", "blow your minds", "blow your mind", "upset", "bummed", "sad", "must know", "safe your", "hits", "biggest", "best", "worst", "terrible", "sensation", "sensational", "secretly", "nuts", "nobody tells you", "won’t tell you", "change your life", "not ready", "cost you", "top", "exposed", "exposes", "expose", "sex", "sexy", "hoax", "whooping", "blow up", "blew up"}

def testClassifiers(dataset):
	kneighbors = KNeighborsClassifier(n_neighbors=2)
	gaussian = GaussianNB()
	logistic_reg = LogisticRegression(solver='sag', max_iter=100, random_state=42, multi_class='ovr')
	svm = svm.SVC()

	classifiers = {}
	classifiers['kneighbors'] = getMeanAuc(kneighbors, dataset)
	classifiers['gaussian'] = getMeanAuc(gaussian, dataset)
	classifiers['logistic_reg'] = getMeanAuc(logistic_reg, dataset)
	classifiers['svm'] = getMeanAuc(svm, dataset)

	return classifiers

def getMeanAuc(cls, dataset):
	datax = dataset[:, :-1]
	datay = dataset[:,-1]

	cv = StratifiedKFold(n_splits=6)
	roc_auc_scores = []

	for (train, test) in cv.split(datax, datay):
	    #make predictions for each split
	    probas_ = cls.fit(datax[train], datay[train]).predict(datax[test])
	    #add to the list of scores
	    roc_auc_scores.append(roc_auc_score(temp, probas_))
	#get the mean AUC
	return np.mean(roc_auc_scores)

def train(cls, dataset):
	datax = dataset[:, :-1]
	datay = dataset[:,-1]

	cls.fit(datax, datay)

def test(cls, data):
	return cls.predict(data)

def buildWebsiteSet(sites, data):
	for i in data.shape[0]:
		if data[i,-1]==0:
			sites.add(data[i,0])
	return sites

def makeArray(data):
	website = data[0]
	title = data[1]
	capitalRatio = sum(1.0 for c in title if c.isupper())/sum(1 for c in title if c.isalpha())
	punctRatio = sum(1.0 for c in title if c=="?" or c=="!")/sum(1 for c in title if not c.isspace())
	emotionalRatio = sum(1 for word in title.tolower().split(' ') if word in emotional_words)/sum(1 for word in title.split(' '))
	return np.array(website, capitalRatio, puncRatio, emotionalRatio)

def getScore(cls, sites, data):
	resp = {}
	title = data[0]
	data = data[1:]
	resp['site'] = 0
	if title in sites:
		resp['site'] = 1
	resp['score'] = test(cls, data)
	return resp