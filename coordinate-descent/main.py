#As provided on stackoverflow, we can use the dataset to test out model
import matplotlib as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import numpy as np

#  boston house-prices dataset
data = load_boston()
X, y = data.data, data.target

X = StandardScaler().fit_transform(X)  # for easy convergence
X = np.hstack([X, np.ones((X.shape[0], 1))])

param = np.zeros(X.shape[1])


#Here the function "coordinate_descent gets called that we need to refer to

cret, cxret = coordinate_descent(X, y, param.copy())

plt.plot(range(len(xret)), xret, label="GD")
plt.plot(range(len(cxret)), cxret, label="CD")
plt.legend()