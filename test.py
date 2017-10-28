import numpy as np
import codecs

with open('sample.csv', encoding='utf-8') as fh:
    data = np.loadtxt(fh, dtype="str_", delimiter=',')
print(data)
print(data.shape)