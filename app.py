from flask import Flask, render_template, request
import yfinance as yf
from flask import Flask, render_template, request
import pandas as pd

# Import the optimization functions
from portfolio_optimizer import *
from cumsum import *
from graphs import *


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    ticker_symbols = request.form.getlist('ticker')
    allocations = request.form.getlist('allocation')

    # Retrieve the data for the selected ticker symbols from Yahoo Finance
    data = pd.DataFrame()
    for ticker in ticker_symbols:
        stock_data = yf.download(ticker, start='2010-01-01', end='2023-05-28')
        stock_returns = stock_data['Adj Close'].pct_change()
        data[ticker] = stock_returns

    # Convert allocation percentages to floats
    allocations = [float(alloc) for alloc in allocations]

    if request.form['technique'] == 'mean_variance':
        weights, return1, volatility, value1, name = mean_variance_optimization(data)
    elif request.form['technique'] == 'cvar':
        weights, return1, volatility, value1, name= conditional_value_at_risk_optimization(data)
    elif request.form['technique'] == 'risk_parity':
        weights, return1, volatility, value1, name= risk_parity_optimization(data)
    elif request.form['technique'] == 'tracking_error':
        weights, return1, volatility, value1, name= tracking_error_optimization(data)
    elif request.form['technique'] == 'information_ratio':
        weights, return1, volatility, value1, name= information_ratio_optimization(data)
    elif request.form['technique'] == 'kelly_criterion':
        weights, return1, volatility, value1, name= kelly_criterion_optimization(data)
    elif request.form['technique'] == 'sortino_ratio':
        weights, return1, volatility, value1, name= sortino_ratio_optimization(data)
    elif request.form['technique'] == 'omega_ratio':
        weights, return1, volatility, value1, name= omega_ratio_optimization(data)
    elif request.form['technique'] == 'maximum_drawdown':
        weights, return1, volatility, value1, name= maximum_drawdown_optimization(data, allocations)
    elif request.form['technique'] == 'sharpe_ratio':
        weights, return1, volatility, value1, name= risk_adjusted_return_optimization(data, allocations)


    stockolddf = pd.DataFrame(
    {'Ticker': ticker_symbols,
     'Allocation': allocations
    })

    stocknewdf = pd.DataFrame(
    {'Ticker': ticker_symbols,
     'Allocation': weights
    })

    currentreturn1 = calculate_portfolio_returns(stockolddf)
    currentreturn = round(currentreturn1[-1], 3)

    optimizedreturn1 = calculate_portfolio_returns(stocknewdf)
    optimizedreturn = round(optimizedreturn1[-1], 3)

    returnchart = return_graph(optimizedreturn1, currentreturn1)

    # Pass the optimization results to the template
    return render_template('result.html',
                           ticker_symbols=ticker_symbols,
                           allocations=allocations,
                           weights=weights,
                           currentreturn=currentreturn,
                           optimizedreturn=optimizedreturn,
                           volatility=volatility,
                           returnchart = returnchart, name=name, value1=value1)



if __name__ == '__main__':
    app.run(debug=True)