#As provided on stackoverflow, we can use the dataset to test out model
import matplotlib.pyplot as plt
import CoordinateDescent as cd
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas
import ElasticNetCD as encd

#  california house-prices dataset
data = fetch_california_housing(as_frame=True)

X, y = data.data, data.target

X = StandardScaler().fit_transform(X)  # for easy convergence
X = np.hstack([X, np.ones((X.shape[0], 1))])

param = np.zeros(X.shape[1])


#Here the function "coordinate_descent gets called that we need to refer to
cret, cxret = cd.CoordinateDescent.coordinate_descent(X, y, param.copy())
print(cret)

plt.plot(range(len(cxret)), cxret, label="CD")

#Elastic net using coordinate descent
bstar = encd.ElasticNetCD.elastic_net(X, y, 0.2, 0.3)
print(bstar)



plt.show()
