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
def form_optimisation_matricies(tickers: list[str], start_time: str, end_time: str, society_factor, knowledge_factor, health_factor, environment_factor, risk_factor: float):
    for ticker in tickers:
        assert ticker in ["AMZN", "PM", "CVX", "PFE", "TSLA", "JPM", "V", "GOOGL", "NFLX", "XOM"], "Only spesific stocks can be used"
    ENV_VALUES = {"AMZN": [0.102, -0.034, 0.005, -0.85],
                  "PM": [0.082, -0.019, -1.481, -0.095],
                  "CVX": [0.168, -0.019, -0.052, -0.34],
                  "PFE": [0.067, -0.037, 0.547, -0.1],
                  "TSLA": [0.22, -0.028, 0.018, -0.048],
                  "JPM": [0.076, -0.05, 0.004, -0.29],
                  "V": [0.087, -0.044, 0.002, -0.023],
                  "GOOGL": [0.072, 0.127, -0.15, -0.035],
                  "NFLX": [0.08, -0.035, 0.012, -0.018],
                  "XOM": [0.175, -0.022, -0.053, -0.323]}
    stock_data = yf.download(tickers, start=start_time, end=end_time)
    # esg_values = np.zeros(len(tickers))
    # for i, ticker in enumerate(tickers):
    #     stock = yf.Ticker(ticker)
    #     esg_values[i] = stock.sustainability.loc["totalEsg"].values[0]
    society_values = np.zeros(len(tickers))
    knowledge_values = np.zeros(len(tickers))
    health_values = np.zeros(len(tickers))
    environment_values = np.zeros(len(tickers))
    for i, ticker in enumerate(tickers):
        society_values[i] = ENV_VALUES[ticker][0]
        knowledge_values[i] = ENV_VALUES[ticker][1]
        health_values[i] = ENV_VALUES[ticker][2]
        environment_values[i] = ENV_VALUES[ticker][3]
    esg_values = 1 + society_factor * society_values + knowledge_factor * knowledge_values + health_factor * health_values + environment_factor * environment_values
    prices = stock_data["Close"]
    prices = prices[tickers]
    price_changes = prices.pct_change().dropna()
    mean_vector = price_changes.mean().to_numpy()
    covariance_matrix = price_changes.cov().to_numpy()
    constraint_matrix = np.ones((1, len(tickers)))
    constraint_vector = np.array([[1]])
    return mean_vector * esg_values, risk_factor * covariance_matrix, constraint_matrix, constraint_vector


if __name__ == "__main__":
    mean, covariance, constraint_matrix, constraint_vector = form_optimisation_matricies(["AMZN", "PM", "CVX", "PFE", "TSLA", "JPM", "V", "GOOGL", "NFLX", "XOM"], "2022-01-01", '2023-10-01', 0.5, 0.5, 0.5, 0.5, 4.0)
    print(mean)
    print(covariance)
