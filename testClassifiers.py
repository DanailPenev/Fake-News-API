import ml
import numpy as np
import pandas as pd

data = pd.read_csv('data.csv', keep_default_na=False, low_memory=False)
data = data.replace('NA', np.NaN)
data = data.as_matrix()
titles = data[:,1]
data1 = data[:12000, :]
data2 = data[410000:, :]
data = np.vstack((data1,data2))
data = data[:,2:]

data = data.astype('float64')
data = np.nan_to_num(data)

# data = np.loadtxt('data.csv', delimiter=",", dtype='str')

classifiers = ml.testClassifiers(data)

for i in classifiers:
	print(classifiers[i])