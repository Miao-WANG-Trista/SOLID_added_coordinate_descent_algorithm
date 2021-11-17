import numpy as np

class CoordinateDescent:

    def __init__(self, PARAMETERS):

    #Import from stackexchange
    #https://stackoverflow.com/questions/51977418/coordinate-descent-in-python
    def coordinate_descent(X, y, param, iter=300):
        cost_history = [0] * (iter+1)
        cost_history[0] = costf(X, y, param)

        for iteration in range(iter):
            for i in range(len(param)):
                dele = np.dot(np.delete(X, i, axis=1), np.delete(param, i, axis=0))
                param[i] = np.dot(X[:,i].T, (y.ravel() - dele))/np.sum(np.square(X[:,i]))
                cost = costf(X, y, param)
                cost_history[iteration+1] = cost

        return param, cost_history


