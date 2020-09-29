import numpy as np


data_path = ""
v = np.genfromtxt(data_path)
v /= np.sum(v)

eps = 1
b = ((eps - 1) * np.exp(eps) + 1) / (2 * np.exp(eps) * (np.exp(eps) - 1 - eps))


p = np.exp(eps) / (2 * b * np.exp(eps) + 1)
q = 1 / (2 * b * np.exp(eps) + 1)

output_v = []
for i in range(np.size(v)):
	x = np.random.binomial(n = 1, p = 2 * b * p)

	if x == 1:
		output_v.append(np.random.uniform(v[i] - b, v[i] + b))
	if x != 1:
		indicator = np.random.randint(1, 2)
		if indicator == 1:
			output_v.append( np.random.uniform(-b, v[i] - b) )
		if indicator == 2:
			output_v.append( np.random.uniform(v[i] + b, 1 + b) )
