import CDGoldenSearch

class Algorithm(CDGoldenSearch.CDGoldenSearch):
    
    """
    Minimze function X1 + X2**2 + 2*X3**2 + 3*X4**3
    """
    
    
    def func_val(x): # x is a list, function is given, so the format of return is given
        return x[0]+pow(x[1],2)+2*pow(x[2],2)+3*pow(x[3],3)


    def func(p, index_p, list_others, n): 
        x = list(range(n))
        x[index_p] = p
        
        for i in range(index_p):
            x[i]=list_others[i]
        for i in range(index_p+1,n):
            x[i]=list_others[i-1]
        return x[0]+pow(x[1],2)+2*pow(x[2],2)+3*pow(x[3],3)
    
def test_algorithm():
    test = Algorithm(Algorithm.func, 4, -1.5, 1.5)
    x_min = test.run()
    y_min = Algorithm.func_val(x_min)


