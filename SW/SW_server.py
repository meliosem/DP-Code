import numpy as np
from scipy.spatial.distance import jensenshannon

def Gen_Trans_Matrix(b, bins, eps):
	p = np.exp(eps) / ((2*b + 1) * np.exp(eps) + bins - 1)
	q = 1 / ((2*b + 1) * np.exp(eps) + bins - 1)
	m = np.zeros([bins, bins])
	unit_length = (2*b + 1) / bins
	for row in range(m.shape[0]):
		for column in range(m.shape[1]):
			if np.abs(column - row) * unit_length > b:
				m[row][column] = q
			else:
				m[row][column] = p
	return m

data = np.random.random(size = 1000)	# can replace this with your own data files

eps = 1
bins = 10
b = ((eps - 1) * np.exp(eps) + 1) / (2 * np.exp(eps) * (np.exp(eps) - 1 - eps))
unit_length = (2*b + 1) / bins
M = Gen_Trans_Matrix(b, bins, eps)

n = np.zeros([bins])
for x in data:
	n[int(x // unit_length)] += 1


t = np.exp(eps) * 0.001
estimator = np.array([1 / bins] * bins)
last_estimator = estimator + t 	# plusing t makes the program can go into the while loop
p = np.zeros([bins])

while np.sum(last_estimator - estimator) > t:
	last_estimator = estimator
	for i in range(np.size(p)):
		s = 0
		for j in range(bins):
			s += n[j] * M[i][j] / np.dot(M[:, j], estimator)
		p[i] = estimator[i] * s

	for i in range(np.size(p)):
		estimator[i] = p[i] / np.sum(p)


print(estimator)