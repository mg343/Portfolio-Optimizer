import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize



def mean_variance_optimization(data, allocations):
    # Calculate expected returns

    data = data.fillna(0)

    returns = data.pct_change().mean().values

    # Calculate covariance matrix
    cov_matrix = data.pct_change().cov().values

    # Define the risk tolerance level
    risk_tolerance = 0.1

    n_assets = len(returns)

    # Define the objective function for optimization
    def objective(weights):
        portfolio_return = np.dot(returns, weights)
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        return portfolio_volatility

    # Define the constraint for portfolio weights summing to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Define the bounds for portfolio weights (0 <= weight <= 1)
    bounds = tuple((0, 1) for _ in range(n_assets))

    # Set an initial equal-weight allocation as a starting point
    initial_weights = np.array([1 / n_assets] * n_assets)

    # Perform mean-variance optimization using scipy's minimize function
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    # Get the optimized portfolio weights
    optimized_weights = result.x

    # Calculate portfolio metrics (e.g., expected return, volatility)
    portfolio_return = np.dot(returns, optimized_weights)
    portfolio_volatility = np.sqrt(np.dot(optimized_weights, np.dot(cov_matrix, optimized_weights)))

    return optimized_weights, portfolio_return, portfolio_volatility



def calculate_cvar(data, allocations):
    returns = data.mean(axis=0).values
    portfolio_return = np.dot(returns, allocations)
    portfolio_returns_sorted = np.sort(portfolio_return.flatten())
    p = int(0.1 * len(portfolio_returns_sorted))
    cvar = -portfolio_returns_sorted[:p].mean()
    return cvar



def conditional_value_at_risk_optimization(data, allocations):
    returns = data.pct_change().mean().values
    cvar = calculate_cvar(data, allocations)
    n_assets = len(returns)

    def objective(weights):
        portfolio_return = np.dot(returns, weights)
        portfolio_cvar = np.dot(weights, cvar)
        return portfolio_cvar

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.array([1 / n_assets] * n_assets)

    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    optimized_weights = result.x

    portfolio_return = np.dot(returns, optimized_weights)
    portfolio_cvar = np.dot(optimized_weights, cvar)

    return optimized_weights, portfolio_return, portfolio_cvar



