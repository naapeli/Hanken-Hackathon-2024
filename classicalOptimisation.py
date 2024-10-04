from docplex.mp.model import Model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from MeanAndCovarianceForming import form_optimisation_matricies

"""
Branc-and-Bound algorithm for Portfolio optimisation

INPUTS
means: means of stock prices
covariance_matrix: covariance between stock returns
tickers: Stock names
object: Objective function maximization e.g. 'Returns' as 'R'  and 'Risk Adjusted Returns' as 'RAR'
"""

def branch_and_bound(means: np.ndarray, covariance_matrix: np.ndarray, tickers: list[str], object: str):

    mdl = Model(name='Portfolio_Optimization')
    
    stock_data = {
        'Stock': tickers,
        'Return': means
    }
    
    df_stock = pd.DataFrame(stock_data, columns = ['Stock', 'Return'])
    df_stock.set_index(['Stock'], inplace=True)
    
    stocks = df_stock.index
    
    df_stock['frac'] = mdl.continuous_var_list(stocks, name='frac', ub=1)
    
    mdl.add_constraint(mdl.sum(df_stock.frac) == 1)
    
    actual_return = mdl.dot(df_stock.frac, df_stock['Return'])
    mdl.add_kpi(actual_return, 'ROI')
    
    fracs = df_stock.frac
    
    df_variance = pd.DataFrame(covariance_matrix, index = stocks, columns = stocks)
    
    variance = mdl.sum(float(df_variance[i][j]) * fracs[i] * fracs[j] for i in stocks for j in stocks)
    
    mdl.add_kpi(variance, 'Variance')
    
    if object == 'R':
        mdl.maximize(actual_return)
    elif object == 'RAR':
        mdl.maximize(actual_return - variance)
    else: 
        raise ValueError(f"Invalid objective '{object}'. Expected 'R' or 'RAR'.")
    
    
    assert mdl.solve(), 'Solve failed'
    mdl.report()
    
    return df_stock


"""
Function for visualizing optimisation results
"""
def display_pie(pie_values, pie_labels, colors=None,title=''):
    plt.axis("equal")
    plt.pie(pie_values, labels=pie_labels, colors=colors, autopct="%1.1f%%")
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    mean, covariance, constraint_matrix, constraint_vector, tickers, prices = form_optimisation_matricies(["AMZN", "PM", "CVX", "PFE", "TSLA", "JPM", "V", "GOOGL", "NFLX", "XOM"], "2022-01-01", '2023-10-01', 0.5, 0.5, 0.5, 0.5, 4.0)

    df_stocks = branch_and_bound(mean, covariance, tickers, 'RAR')
        
    all_fracs = {}
    for row in df_stocks.itertuples():
        pct = 100 * row.frac.solution_value
        all_fracs[row[0]] = pct
        print('Portfolio allocation in: {0:<12}: {1:.2f}%'.format(row[0], pct))
                    
    display_pie( list(all_fracs.values()), list(all_fracs),title='Portfolio Allocation')