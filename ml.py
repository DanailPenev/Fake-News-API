#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import time
import pandas as pd
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


def train(cls, df, vectorizer):
	y = df.real 
	# # Drop the `label` column
	df = df.drop("real", axis=1)
	df = df['title']
	# # Fit and transform the training data 
	X = vectorizer.fit_transform(df) 

	cls.fit(X, y)
	return cls

def test(cls, data, vectorizer):
	data = pd.DataFrame([{"title": data}])
	data = data['title']
	return cls.predict_proba(vectorizer.transform(data))

if __name__ == "__main__":
	count_vectorizer = CountVectorizer(stop_words='english')

	# initialize classifier
	clf = MultinomialNB()

	df = pd.read_csv("BetterNewDataForDani.csv")
	df = df.set_index("Unnamed: 0")

	clf = train(clf, df, count_vectorizer)
	joblib.dump(count_vectorizer, 'vectorizer.pkl')
	joblib.dump(clf, 'classifier.pkl')