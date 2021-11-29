import numpy as np

class CoordinateDescent:

    def costf(X, y, param):
        return np.sum((X.dot(param) - y) ** 2) / (2 * len(y))

    #Import from stackexchange
    #https://stackoverflow.com/questions/51977418/coordinate-descent-in-python

    def coordinate_descent(X, y, param, iter=300):


        cost_history = [0] * (iter+1)
        cost_history[0] = CoordinateDescent.costf(X, y, param)

        for iteration in range(iter):
            for i in range(len(param)):
                dele = np.dot(np.delete(X, i, axis=1), np.delete(param, i, axis=0))
                param[i] = np.dot(X[:,i].T, (y.ravel() - dele))/np.sum(np.square(X[:,i]))
                cost = CoordinateDescent.costf(X, y, param)
                cost_history[iteration+1] = cost

        return param, cost_history

#this is a test 

