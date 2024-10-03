import yfinance as yf
import numpy as np


"""
eg.
tickers = ['AMZN', 'GOOGL'] (of length n)
start_time = '2020-01-01'
end_time = '2023-10-01'
esg_factor = 0.5 >= 0
risk_factor = 0.5 >= 0

RETURNS:
mean vector of length n  (added with the ), covariance matrix of shape (n, n), eq_constraint matrix (m, n), eq_constraint vector (m,)
"""
def form_optimisation_matricies(tickers: list[str], start_time: str, end_time: str, esg_factor: float, risk_factor: float):
    stock_data = yf.download(tickers, start=start_time, end=end_time)
    esg_values = np.zeros(len(tickers))
    for i, ticker in enumerate(tickers):
        stock = yf.Ticker(ticker)
        esg_values[i] = stock.sustainability.loc["totalEsg"].values[0]
    prices = stock_data["Close"]
    prices = prices[tickers]
    price_changes = prices.pct_change().dropna()
    mean_vector = price_changes.mean().to_numpy()
    covariance_matrix = price_changes.cov().to_numpy()
    constraint_matrix = np.ones((1, len(tickers)))
    constraint_vector = np.array([[1]])
    return mean_vector + 0.0001 * esg_factor * esg_values, risk_factor * covariance_matrix, constraint_matrix, constraint_vector


if __name__ == "__main__":
    mean, covariance, constraint_matrix, constraint_vector = form_optimisation_matricies(["MSFT", "NVDA", "TSLA", "CVX"], "2020-01-01", '2023-10-01', 0.1, 4.0)
    print(mean)
    print(covariance)
    print(constraint_matrix.shape)
    print(constraint_vector.shape)
