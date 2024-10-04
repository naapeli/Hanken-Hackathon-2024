import numpy as np 
from MeanAndCovarianceForming import form_optimisation_matricies
"""
Function for performing transformations required for quantum optimization

INPUT:
mean
covariance
price
budget

"""


def transform(mean, covariance, prices, budget):
    P_ = prices.values[-1,:] / budget
    mu_ = P_ * mean
    covariance_ = (P_ * covariance).T * P_ 

    n_max = [int(budget/i) for i in P_]
    d = [int(np.log2(i)) for i in n_max]
    
    def create_matrix(d):
        n = len(d)
    
        total_columns = sum(d_i + 1 for d_i in d)
        
        C = np.zeros((n, total_columns), dtype=int)
        
        start_col = 0
        
        for i in range(n):
            
            for j in range(d[i] + 1):
                C[i][start_col + j] = 2 ** j
            start_col += d[i] + 1
        
        return C
 
    C = create_matrix(d)
    
    mu__ = C.T * mu_
    covariance__ = (C.T @ covariance_) @ C
    P__ = C.T * P_

    return mu__, covariance__, P__, C
    

if __name__ == "__main__":
    mean, covariance, constraint_matrix, constraint_vector, tickers, prices = form_optimisation_matricies(["AMZN", "PM", "CVX", "PFE", "TSLA", "JPM", "V", "GOOGL", "NFLX", "XOM"], "2022-01-01", '2023-10-01', 0.5, 0.5, 0.5, 0.5, 4.0)
    mean_transformed, covariance_transformed, price_transformed, C = transform(mean, covariance, prices, 2000)
    print(mean_transformed)
    print(covariance_transformed)