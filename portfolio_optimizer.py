import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize





def mean_variance_optimization(data):
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

    name = "N/A"
    null = 0

    return optimized_weights, portfolio_return, portfolio_volatility, null, name





def conditional_value_at_risk_optimization(data, risk_tolerance=0.1):
    # Calculate expected returns
    returns = data.pct_change().mean().values

    # Calculate covariance matrix
    cov_matrix = data.pct_change().cov().values

    n_assets = len(returns)

    # Define the objective function for optimization
    def objective(weights):
        portfolio_return = np.dot(returns, weights)
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        cvar = calculate_cvar(data, weights, risk_tolerance)
        return cvar

    # Define the constraint for portfolio weights summing to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Define the bounds for portfolio weights (0 <= weight <= 1)
    bounds = tuple((0, 1) for _ in range(n_assets))

    # Set an initial equal-weight allocation as a starting point
    initial_weights = np.array([1 / n_assets] * n_assets)

    # Perform optimization using scipy's minimize function
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    # Get the optimized portfolio weights
    optimized_weights = result.x

    # Calculate portfolio metrics
    portfolio_return = np.dot(returns, optimized_weights)
    portfolio_volatility = np.sqrt(np.dot(optimized_weights, np.dot(cov_matrix, optimized_weights)))
    cvar = calculate_cvar(data, optimized_weights, risk_tolerance)

    name= 'Conditional Value at Risk Metric'

    return optimized_weights, portfolio_return, portfolio_volatility, cvar, name

def calculate_cvar(data, weights, risk_tolerance):
    portfolio_returns = np.dot(data.pct_change().values, weights)
    cvar = -np.percentile(portfolio_returns, risk_tolerance * 100)
    return cvar





def risk_parity_optimization(data):
    # Calculate covariance matrix
    cov_matrix = data.pct_change().cov().values

    n_assets = len(cov_matrix)

    # Define the objective function for optimization
    def objective(weights):
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        risk_contribution = np.dot(cov_matrix, weights) / np.sqrt(portfolio_variance)
        risk_parity = np.sum((risk_contribution - np.mean(risk_contribution))**2)
        return risk_parity

    # Define the constraint for portfolio weights summing to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Define the bounds for portfolio weights (0 <= weight <= 1)
    bounds = tuple((0, 1) for _ in range(n_assets))

    # Set an initial equal-weight allocation as a starting point
    initial_weights = np.array([1 / n_assets] * n_assets)

    # Perform risk parity optimization using scipy's minimize function
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    # Get the optimized portfolio weights
    optimized_weights = result.x

    # Calculate portfolio metrics (e.g., expected return, volatility)
    portfolio_return = np.dot(data.pct_change().mean().values, optimized_weights)
    portfolio_volatility = np.sqrt(np.dot(optimized_weights, np.dot(cov_matrix, optimized_weights)))

    name = "N/A"
    null = 0

    return optimized_weights, portfolio_return, portfolio_volatility, null, name






def tracking_error_optimization(data):
    # Step 1: Fetch the benchmark data (S&P 500) from Yahoo Finance
    benchmark_ticker = "^GSPC"  # Ticker symbol for S&P 500
    start_date = "2010-01-01"

    benchmark_data = yf.download(benchmark_ticker, start=start_date)["Adj Close"]
    benchmark_returns = benchmark_data.pct_change().dropna()

    # Step 2: Prepare your portfolio data
    # Assuming you have your portfolio data in a DataFrame called 'data'

    # Calculate expected returns and covariance matrix
    returns = data.pct_change().mean().values
    cov_matrix = data.pct_change().cov().values

    # Step 3: Define the objective function and constraints
    def objective(weights):
        portfolio_returns = np.dot(returns, weights)
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        tracking_error = np.sqrt(np.mean((portfolio_returns - benchmark_returns) ** 2))
        return tracking_error

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(returns)))
    initial_weights = np.array([1 / len(returns)] * len(returns))

    # Step 4: Perform the optimization
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    optimized_weights = result.x
    portfolio_return = np.dot(returns, optimized_weights)
    portfolio_volatility = np.sqrt(np.dot(optimized_weights, np.dot(cov_matrix, optimized_weights)))
    tracking_error = np.sqrt(np.mean((portfolio_return - benchmark_returns) ** 2))

    name = 'Tracking Error'
    return optimized_weights, portfolio_return, portfolio_volatility, tracking_error, name





def information_ratio_optimization(data):
    # Step 1: Fetch the benchmark data (S&P 500) from Yahoo Finance
    benchmark_ticker = "^GSPC"  # Ticker symbol for S&P 500
    start_date = "2010-01-01"

    benchmark_data = yf.download(benchmark_ticker, start=start_date)["Adj Close"]
    benchmark_returns = benchmark_data.pct_change().dropna()

    # Calculate expected returns
    returns = data.pct_change().mean().values

    # Calculate covariance matrix
    cov_matrix = data.pct_change().cov().values

    n_assets = len(returns)

    # Define the objective function for optimization
    def objective(weights):
        portfolio_return = np.dot(returns, weights)
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        tracking_error = np.std(portfolio_return - benchmark_returns)
        information_ratio = portfolio_return / tracking_error
        return -information_ratio  # Objective is to maximize information ratio

    # Define the constraint for portfolio weights summing to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Define the bounds for portfolio weights (0 <= weight <= 1)
    bounds = tuple((0, 1) for _ in range(n_assets))

    # Set an initial equal-weight allocation as a starting point
    initial_weights = np.array([1 / n_assets] * n_assets)

    # Perform optimization using scipy's minimize function
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    # Get the optimized portfolio weights
    optimized_weights = result.x

    # Calculate portfolio metrics
    portfolio_return = np.dot(returns, optimized_weights)
    portfolio_volatility = np.sqrt(np.dot(optimized_weights, np.dot(cov_matrix, optimized_weights)))
    tracking_error = np.std(portfolio_return - benchmark_returns)
    information_ratio = portfolio_return / tracking_error

    name = 'Information Ratio'

    return optimized_weights, portfolio_return, portfolio_volatility, information_ratio, name




def kelly_criterion_optimization(data):
    # Calculate expected returns
    returns = data.pct_change().mean().values

    # Calculate covariance matrix
    cov_matrix = data.pct_change().cov().values

    n_assets = len(returns)

    # Define the objective function for optimization
    def objective(weights):
        portfolio_return = np.dot(returns, weights)
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        kelly_criterion = portfolio_return / portfolio_volatility
        return -kelly_criterion  # Objective is to maximize Kelly Criterion

    # Define the constraint for portfolio weights summing to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Define the bounds for portfolio weights (0 <= weight <= 1)
    bounds = tuple((0, 1) for _ in range(n_assets))

    # Set an initial equal-weight allocation as a starting point
    initial_weights = np.array([1 / n_assets] * n_assets)

    # Perform optimization using scipy's minimize function
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    # Get the optimized portfolio weights
    optimized_weights = result.x

    # Calculate portfolio metrics
    portfolio_return = np.dot(returns, optimized_weights)
    portfolio_volatility = np.sqrt(np.dot(optimized_weights, np.dot(cov_matrix, optimized_weights)))
    kelly_criterion = portfolio_return / portfolio_volatility

    name = 'Kelly Criterion'

    return optimized_weights, portfolio_return, portfolio_volatility, kelly_criterion, name





def sortino_ratio_optimization(data, min_acceptable_return=0.10):
    # Calculate expected returns
    returns = data.pct_change().mean().values

    # Calculate downside deviation (negative returns below the minimum acceptable return)
    downside_returns = np.minimum(returns - min_acceptable_return, 0)
    downside_deviation = np.std(downside_returns)

    n_assets = len(returns)

    # Define the objective function for optimization
    def objective(weights):
        portfolio_return = np.dot(returns, weights)
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        sortino_ratio = (portfolio_return - min_acceptable_return) / downside_deviation
        return -sortino_ratio  # Negate the sortino_ratio for minimization

    # Calculate covariance matrix
    cov_matrix = data.pct_change().cov().values

    # Define the constraint for portfolio weights summing to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Define the bounds for portfolio weights (0 <= weight <= 1)
    bounds = tuple((0, 1) for _ in range(n_assets))

    # Set an initial equal-weight allocation as a starting point
    initial_weights = np.array([1 / n_assets] * n_assets)

    # Perform optimization using scipy's minimize function
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    # Get the optimized portfolio weights
    optimized_weights = result.x

    # Calculate portfolio metrics
    portfolio_return = np.dot(returns, optimized_weights)
    portfolio_volatility = np.sqrt(np.dot(optimized_weights, np.dot(cov_matrix, optimized_weights)))
    sortino_ratio = (portfolio_return - min_acceptable_return) / downside_deviation

    name = 'Sortino Ratio'

    return optimized_weights, portfolio_return, portfolio_volatility, sortino_ratio, name





def omega_ratio_optimization(data, min_acceptable_return=0.10):
    # Calculate expected returns
    returns = data.pct_change().mean().values

    # Calculate excess returns above the minimum acceptable return
    excess_returns = returns - min_acceptable_return

    n_assets = len(returns)

    # Define the objective function for optimization
    def objective(weights):
        portfolio_return = np.dot(returns, weights)
        omega_ratio = portfolio_return / np.percentile(excess_returns, 5)  # Omega Ratio at the 5th percentile
        return -omega_ratio  # Negate the omega_ratio for minimization

    # Calculate covariance matrix
    cov_matrix = data.pct_change().cov().values

    # Define the constraint for portfolio weights summing to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Define the bounds for portfolio weights (0 <= weight <= 1)
    bounds = tuple((0, 1) for _ in range(n_assets))

    # Set an initial equal-weight allocation as a starting point
    initial_weights = np.array([1 / n_assets] * n_assets)

    # Perform optimization using scipy's minimize function
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    # Get the optimized portfolio weights
    optimized_weights = result.x

    # Calculate portfolio metrics
    portfolio_return = np.dot(returns, optimized_weights)
    portfolio_volatility = np.sqrt(np.dot(optimized_weights, np.dot(cov_matrix, optimized_weights)))
    omega_ratio = portfolio_return / np.percentile(excess_returns, 5)  # Omega Ratio at the 5th percentile

    name = 'Omega Ratio'

    return optimized_weights, portfolio_return, portfolio_volatility, omega_ratio, name





def maximum_drawdown_optimization(data, allocations, min_acceptable_return=None):
    # Calculate the cumulative returns
    cumulative_returns = np.cumprod(1 + data.pct_change())

    # Calculate the drawdowns
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak

    # Set the minimum acceptable return if provided
    if min_acceptable_return is None:
        min_acceptable_return = np.min(drawdown)

    # Define the objective function for optimization
    def objective(weights):
        portfolio_return = np.dot(data.pct_change().mean(), weights)
        portfolio_drawdown = np.dot(weights, drawdown)
        return portfolio_drawdown
    
    cov_matrix = data.pct_change().cov().values

    # Define the constraint for portfolio weights summing to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Define the bounds for portfolio weights (0 <= weight <= 1)
    bounds = tuple((0, 1) for _ in range(len(allocations)))

    # Set an initial equal-weight allocation as a starting point
    initial_weights = np.array([1 / len(allocations)] * len(allocations))

    # Perform the optimization using scipy's minimize function
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    # Get the optimized portfolio weights
    optimized_weights = result.x

    # Calculate portfolio metrics
    portfolio_return = np.dot(data.pct_change().mean(), optimized_weights)
    portfolio_volatility = np.sqrt(np.dot(optimized_weights, np.dot(cov_matrix, optimized_weights)))
    portfolio_drawdown = np.dot(optimized_weights, drawdown)

    name= 'Portfolio Drawdown'

    return optimized_weights, portfolio_return, portfolio_volatility, portfolio_drawdown, name





def risk_adjusted_return_optimization(data, allocations, risk_free_rate=0.035):
    returns = data.pct_change().mean()
    cov_matrix = data.pct_change().cov()

    n_assets = len(returns)

    # Calculate portfolio return and volatility
    portfolio_return = np.dot(allocations, returns)
    portfolio_volatility = np.sqrt(np.dot(allocations, np.dot(cov_matrix, allocations)))

    # Calculate risk-adjusted return (Sharpe ratio)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    # Define the objective function for optimization
    def objective(weights):
        portfolio_return = np.dot(returns, weights)
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio  # Minimize the negative Sharpe ratio

    # Define the constraint for portfolio weights summing to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Define the bounds for portfolio weights (0 <= weight <= 1)
    bounds = tuple((0, 1) for _ in range(n_assets))

    # Set an initial equal-weight allocation as a starting point
    initial_weights = np.array([1 / n_assets] * n_assets)

    # Perform the optimization using scipy's minimize function
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    # Get the optimized portfolio weights
    optimized_weights = result.x

    # Calculate portfolio metrics
    portfolio_return = np.dot(returns, optimized_weights)
    portfolio_volatility = np.sqrt(np.dot(optimized_weights, np.dot(cov_matrix, optimized_weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    name = 'Sharpe Ratio'

    return optimized_weights, portfolio_return, portfolio_volatility, sharpe_ratio, name