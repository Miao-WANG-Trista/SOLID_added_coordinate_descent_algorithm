import numpy as np

class ElasticNetCD:

    #https://towardsdatascience.com/regularized-linear-regression-models-dcf5aa662ab9
    def elastic_net(X, y, l, alpha, tol=1e-4, path_length=100, return_path=False):
        
        X = np.hstack((np.ones((len(X), 1)), X))
        m, n = np.shape(X)
        B_star = np.zeros((n))
        if alpha == 0:
            l2 = 1e-15
        l_max = max(list(abs(np.dot(np.transpose(X), y)))) / m / alpha
        if l >= l_max:
            return np.append(np.mean(y), np.zeros((n - 1)))
        l_path = np.geomspace(l_max, l, path_length)
        for i in range(path_length):
            while True:
                B_s = B_star
                for j in range(n):
                    k = np.where(B_s != 0)[0]
                    update = (1/m)*((np.dot(X[:,j], y)- \
                                    np.dot(np.dot(X[:,j], X[:,k]), B_s[k]))) + \
                                    B_s[j]
                    B_star[j] = (np.sign(update) * max(
                        abs(update) - l_path[i] * alpha, 0)) / (1 + (l_path[i] * (1 - alpha)))
                if np.all(abs(B_s - B_star) < tol):
                    break
                
        return B_star
