import random
from abc import ABCMeta, abstractmethod

def cal_d(xu, xl):
    """
    :param xu: upper bound of the interval
    :param xl: lower bound of the interval
    :return: the distance of the interval 
    """
    return 0.618 * (xu - xl) # 0.618 is the golden ratio


def check_e(xu, xl, xopt, absolutePrecision=1e-10):
    """
    :param xu: upper bound of the interval
    :param xl: lower bound of the interval
    :param xopt: minimal values of x
    :param absolutePrecision: the stopping precision,tolerance. Default = 1e-10
    :return: whether the stopping precision is reached 
    """
    ea = (1 - 0.618) * abs((xu - xl) / xopt)
    if ea < absolutePrecision:
        return 1
    return 0


def gold_search(func, xu, xl, absolutePrecision=1e-10, itr=1000):
    """
    :param func: objective function
    :param xu: upper bound of the interval
    :param xl: lower bound of the interval
    :param absolutePrecision: the stopping precision, tolerance. Default = 1e-10
    :param itr: maximum iteration times
    :return: optimal values of x in a minimum problem
    """
    x = [0, 0]
    f = [0, 0]
    result = 0
    for i in range(itr):
        d = cal_d(xu, xl)
        x[0] = xl + d
        x[1] = xu - d
      
        f[0] = func(x[0])
        f[1] = func(x[1])
        if f[0] > f[1]:
            xu = x[0]
            xopt = x[1]
            if check_e(xu,xl,xopt, absolutePrecision):
                break
        else:
            xl = x[1]
            xopt = x[0]
            if check_e(xu, xl, xopt, absolutePrecision):
                break
    return xopt
    
    
class CDGoldenSearch:
    
    def __init__ (self, func, n, lower_bound, higher_bound, max_iter=1000):
        """
        :param func: objective function
        :param n: degree of objective function
        :param lower_bound: lower_bound of search interval
        :param higher_bound: higher_bound of search interval
        :param max_iter: times of iterations to stop algorithm once reached
        """
        self.func = func
        self.n = n
        self.lower_bound = lower_bound
        self.higher_bound = higher_bound
        self.max_iter = max_iter
    
    @abstractmethod
    def func(p, index_p, list_others, n):
        """
        Defines objective function used in golden search
        """
        pass
    
    
    @abstractmethod
    def func_val(x):
        """
        Defines the objective function
        """
        pass
    
    
    def run(self):
        """
        Conducts coordinate descent 
        """
        # random initialization
        # here, we use a list to store values in different coordinates
        init_X = list(range(self.n))

        X = [round(random.uniform(self.lower_bound,self.higher_bound),2) 
             for _ in range(self.n) ]
        for i in range(self.n):
            init_X[i] = X[i]
        
        opt_X = init_X
       
        for i in range(self.max_iter):
            for k in range(self.n):
                # optimize from x[0] and then x[1], x[2]...
                list_p = opt_X[0:k] + opt_X[k+1:self.n]
                opt_X[k] = gold_search(lambda x: self.func(x, k, list_p, self.n),
                                     self.lower_bound,self.higher_bound)
        
        return opt_X
        



    
