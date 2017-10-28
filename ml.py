import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

fakenews = np.loadtxt("dataset.csv", delimiter=",", dtype='str')
col = np.zeros((fakenews.shape[0],1))
fakenews = np.append(fakenews,col,axis=1)
for i in range(fakenews.shape[0]):
	title = fakenews[i,1]
	fakenews[i,-1] = sum(1.0 for c in title if c.isupper())/sum(1 for c in title if c.isalpha())	
X = fakenews[:, -1].reshape(6,1)
y = fakenews[:,-2]
print(X)
print(y)
clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(X, y)
print(clf.predict(np.array([[0.2]])))