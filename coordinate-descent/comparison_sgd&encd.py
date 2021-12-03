import numpy as np
import matplotlib.pyplot as plt
import random
import time
import ElasticNetCD as encd
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

def sgd(samples, y, step_size=0.005, max_iteration_count=100):
    sample_num, dimension = samples.shape
    w = np.ones((dimension,1), dtype=np.float32)
    loss_collection = []
    loss = 1
    iteration_count = 0
    while loss > 0.001 and iteration_count < max_iteration_count:
        loss = 0
        gradient = np.zeros((dimension,1), dtype=np.float32)
        
        #Randomly choose a sample to update the weights
        sample_index = random.randint(0, sample_num-1)
        predict_y = np.dot(w.T, samples[sample_index])
        for j in range(dimension):
            gradient[j] += (predict_y - y[sample_index]) * samples[sample_index][j]
            w[j] -= step_size * gradient[j]

        for i in range(sample_num):
            predict_y = np.dot(w.T, samples[i])
            loss += np.power((predict_y - y[i]), 2)

        loss_collection.append(loss / (2 * len(y)))
        iteration_count += 1
    return w,loss_collection

#  california house-prices dataset
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
X = StandardScaler().fit_transform(X)

#Application of EN
start1 = time.time()
B_hat, cost_history, objective = encd.ElasticNetCD.elastic_net(X, y, 0.8, 0.3, 1e-4, 100)
end1 = time.time()

#Application of SGD
start2 = time.time()
bret, bxret = sgd(X, y, step_size=0.005, max_iteration_count=100)
end2 = time.time()

#Print the running time for each algorithm
print ("The running time of EN is " + str(end1-start1))
print ("The running time of SGD is " + str(end2-start2))

#plot the loss function
plt.plot(range(len(cost_history)), cost_history, label="EN", color='r')
plt.title('SGD & EN Loss')
plt.xlabel('Iterations (Path length)')
plt.ylabel('Loss function')
plt.plot(range(len(bxret)), bxret, label="SGD")
plt.legend()
plt.show()
