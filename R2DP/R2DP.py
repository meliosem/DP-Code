import numpy as np
from scipy.optimize import minimize

def Constraint(x, delta_q, eps):
	return (x[0] ** 2 * x[3] * x[4] + x[1] ** 2 * (x[5] + x[6]) / 2 + x[2] ** 2 * x[7]) /\
	(\
		(\
		(1 / x[0] * x[3] * x[4] * (1 + x[0] * delta_q * x[4])) ** (x[3] + 1) * \
		(np.exp(-x[0] * delta_q * x[6]) - np.exp(-x[0] * delta_q * x[5]) / ( -x[0] * delta_q * (x[6] - x[5]))) * \
		np.exp(-x[0] * delta_q * x[7]) * np.exp(.5 * (x[8] * x[0] * delta_q) ** 2) \
		) + \
		(\
		(1 / (x[4] * -x[1] * delta_q)) ** x[3] * \
		( (x[6] * np.exp(-x[1] * delta_q * x[6]) - x[5] * np.exp(-x[1] * delta_q * x[5])) * (-x[1] * delta_q) - np.exp(-x[1] * delta_q * x[6]) + np.exp(-x[1] * delta_q * x[5])) / (x[1] * delta_q) ** 2 * (x[6] - x[5]) *\
		np.exp(-x[1] * delta_q * x[7]) * np.exp(.5 * (x[8] * x[1] * delta_q) ** 2)
		) + \
		(\
		(1 / (x[4] * -x[2] * delta_q)) ** x[3] * \
		(np.exp(-x[2] * delta_q * x[6]) - np.exp(-x[2] * delta_q * x[5]) / ( -x[2] * delta_q * (x[6] - x[5]))) * \
		(x[7] + x[8] ** 2 * -x[2] * delta_q) * np.exp(x[7] * -x[2] * delta_q + .5 * (x[8] * -x[2] * delta_q) ** 2)
		)
	) - eps

usefulness_bound = 5
eps = 1
delta_q = 1
# x[0]:a_1, x[1]:a_2, x[2]:a_3, x[3]:k, x[4]:theta, x[5]:a, x[6]:b, 
# x[7]:mu, x[8]:sigma

fun = lambda x : (1 / (1 + x[0] * x[4] * usefulness_bound)) ** x[3] * \
	(np.exp(-x[1] * x[6] * usefulness_bound) - np.exp(-x[1] * x[5] * usefulness_bound) / (-x[1] * (x[6] - x[5]) * usefulness_bound)) * \
	(np.exp(-x[7] * x[2] * usefulness_bound) * np.exp(.5 * (x[8] * x[2] * usefulness_bound) ** 2))

cons = ({"type": "eq", "fun": lambda x : (x[0] ** 2 * x[3] * x[4] + x[1] ** 2 * (x[5] + x[6]) / 2 + x[2] ** 2 * x[7]) /\
	(\
		(\
		(1 / x[0] * x[3] * x[4] * (1 + x[0] * delta_q * x[4])) ** (x[3] + 1) * \
		(np.exp(-x[0] * delta_q * x[6]) - np.exp(-x[0] * delta_q * x[5]) / ( -x[0] * delta_q * (x[6] - x[5]))) * \
		np.exp(-x[0] * delta_q * x[7]) * np.exp(.5 * (x[8] * x[0] * delta_q) ** 2) \
		) + \
		(\
		(1 / (x[4] * -x[1] * delta_q)) ** x[3] * \
		( (x[6] * np.exp(-x[1] * delta_q * x[6]) - x[5] * np.exp(-x[1] * delta_q * x[5])) * (-x[1] * delta_q) - np.exp(-x[1] * delta_q * x[6]) + np.exp(-x[1] * delta_q * x[5])) / (x[1] * delta_q) ** 2 * (x[6] - x[5]) *\
		np.exp(-x[1] * delta_q * x[7]) * np.exp(.5 * (x[8] * x[1] * delta_q) ** 2)
		) + \
		(\
		(1 / (x[4] * -x[2] * delta_q)) ** x[3] * \
		(np.exp(-x[2] * delta_q * x[6]) - np.exp(-x[2] * delta_q * x[5]) / ( -x[2] * delta_q * (x[6] - x[5]))) * \
		(x[7] + x[8] ** 2 * -x[2] * delta_q) * np.exp(x[7] * -x[2] * delta_q + .5 * (x[8] * -x[2] * delta_q) ** 2)
		)
	) - eps})


x_0 = np.array([10, 10, 10, 10, 10, 1, 10, 0, 1])
x_0 = np.random.randint(0, 10, size = 9)

result = minimize(fun, x_0, method = "SLSQP", constraints = cons)

# print(result.x)

a_1 = result.x[0]
a_2 = result.x[1]
a_3 = result.x[2]
k = result.x[3]
theta = result.x[4]
a = result.x[5]
b = result.x[6]
mu = result.x[7]
sigma = result.x[8]

x_1 = np.random.gamma(k, theta)
x_2 = np.random.uniform(a, b)
x_3 = np.random.normal(mu, sigma)

b = 1 / (a_1 * x_1 + a_2 * x_2 + a_3 * x_3)

print(np.random.laplace(0, b))