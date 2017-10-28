import ml

data = np.loadtxt('data.csv', delimiter=",", dtype='str')

classifiers = ml.testClassifiers(data)

for i in classifiers:
	print(i + ": " + classifiers[i])