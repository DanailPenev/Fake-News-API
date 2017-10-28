import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score

df = pd.read_csv("BetterNewDataForDani.csv", low_memory=False)

df = df.set_index("Unnamed: 0")
df = df.reset_index(drop=True)
y = df.real 
# # Drop the `label` column
df = df.drop("real", axis=1)
# # Make training and test sets 
X_train, X_test, y_train, y_test = train_test_split(df['title'], y, test_size=0.33, random_state=53)
print(X_train, X_test)
# # Initialize the `count_vectorizer` 
count_vectorizer = CountVectorizer(stop_words='english')

# # Fit and transform the training data 
count_train = count_vectorizer.fit_transform(X_train) 

# # Transform the test set 
count_test = count_vectorizer.transform(X_test)

clf = MultinomialNB()
clf.fit(count_train, y_train)
print(count_train)
pred = clf.predict(count_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)

start_time = time.time()
cv = StratifiedKFold(n_splits=6)
roc_auc_scores = []

for (train_index, test_index) in cv.split(df, y):
	#make predictions for each split
	X_train, X_test = df.loc[train_index], df.loc[test_index]
	X_train, X_test = X_train['title'], X_test['title']
	y_train, y_test = y.loc[train_index], y.loc[test_index]
	count_train = count_vectorizer.fit_transform(X_train) 
	count_test = count_vectorizer.transform(X_test)
	# print(count_train)
	# print(count_test)
	probas_ = clf.fit(count_train, y_train).predict(count_test)
	temp = [x for x in y_test] 
	#add to the list of scores
	roc_auc_scores.append(roc_auc_score(temp, probas_))
#get the mean AUC
elapsed_time = time.time() - start_time
print(elapsed_time)
print(np.mean(roc_auc_scores))

# # Initialize the `tfidf_vectorizer` 
# tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7) 

# # Fit and transform the training data 
# tfidf_train = tfidf_vectorizer.fit_transform(X_train) 

# # Transform the test set 
# tfidf_test = tfidf_vectorizer.transform(X_test)

# # Get the feature names of `tfidf_vectorizer` 
# print(tfidf_vectorizer.get_feature_names()[-10:])

# # Get the feature names of `count_vectorizer` 

# clf = MultinomialNB() 

# clf.fit(tfidf_train, y_train)
# pred = clf.predict(tfidf_test)
# score = metrics.accuracy_score(y_test, pred)
# print("accuracy:   %0.3f" % score)

# linear_clf = PassiveAggressiveClassifier(max_iter=50)
# linear_clf.fit(tfidf_train, y_train)
# pred = linear_clf.predict(tfidf_test)
# score = metrics.accuracy_score(y_test, pred)
# print("accuracy:   %0.3f" % score)



