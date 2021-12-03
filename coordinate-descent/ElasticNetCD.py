import numpy as np
from numpy import linalg as LA

class ElasticNetCD:

    #a 1/(2m) term is added for mathematical completeness
    def cost_elastic(X, y, param, l, a):
        return (np.sum((X.dot(param) - y) ** 2) / (2 * len(y)))+l*((1-a)*(1/2)*(LA.norm(param,2) ** 2) + a*LA.norm(param,1))

    def objective_function(X, param):
        return X.dot(param)

    def elastic_net(X, y, lambdaa, alpha, tol=1e-4, path_length=100):
        
        #Matrix for X with the independent variables. It includes a column of 1s to take into account the intercept
        X = np.hstack((np.ones((len(X), 1)), X))
        m, n = np.shape(X)
        
        #B hat will be the estimated parameters for the model
        #It is initialized with zeros
        B_hat = np.zeros((n))
        
        #alpha will determine the weight given to the L1 penalty term. 
        #(1-alpha) will be the weight for the L2 penalty term
        if alpha == 0:
            l2 = 1e-15
            
        #lambda will determine the intensity of the regularization applied
        #We initialize with a max value for lambda which is the minimum value that will bring the estimates for all coefficients to zero.
        lambda_max = max(list(abs(np.dot(np.transpose(X), y)))) / m / alpha
        
        #Any values for lambda above this max value will result in total sparsity of the coefficient vector
        if lambdaa >= lambda_max:
            return np.append(np.mean(y), np.zeros((n - 1)))
        
        #The search space for the tuning parameter lambda will be bounded by 
        #the user-defined lambda and the max value calculated for lambda
        #The number of search values will be determined by the path length parameter defined as input
        lambda_path = np.geomspace(lambda_max, lambdaa, path_length)
        
        cost_history=[]
        objective_history=[]
        
        #We iterate over the number of tuning parameter (lambda) values included in the path
        for i in range(path_length):
            while True:
                B_s = B_hat
                
                #We iterate over each coordinate (X variables) to update it: Principle of Coordinate descent
                for j in range(n):
                    k = np.where(B_s != 0)[0]

                    #Sum of current model residuals times x values for the coordinate to be updated
                    covariance_updates=(np.dot(X[:,j], y)- np.dot(np.dot(X[:,j], X[:,k]), B_s[k]))
                    
                    #We use Naive updates on the value of the coefficient for the current coordinate
                    #We calculate the average of covariance updates on all samples and we add the estimate from the previous iteration
                    naive_update = (1/m)*covariance_updates + B_s[j]
                    
                    #We calculate the model coefficient estimates
                    #The numerator is the soft thresholding operator, which sets small values to zero and shrinks large values toward zero.
                    #The denominator takes into account the dual penalties
                    soft_thresholding=(np.sign(naive_update) * max(abs(naive_update) - lambda_path[i] * alpha, 0))
                    B_hat[j] = soft_thresholding / (1 + (lambda_path[i] * (1 - alpha)))
                
               
                #We calculate the loss function for Elastic Net
                cost = ElasticNetCD.cost_elastic(X, y, B_hat, lambdaa, alpha)
                cost_history.append(cost)

                #We calculate the objective function
                objective = ElasticNetCD.objective_function(X, B_hat)
                objective_history.append(objective)
                
                #If the change in the coefficients' values is lower than the tolerance defined as input, we exit the algorithm
                if np.all(abs(B_s - B_hat) < tol):
                    break
                                   
        return B_hat, cost_history, objective
