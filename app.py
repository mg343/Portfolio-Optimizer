from flask import Flask, render_template, request
import yfinance as yf
from flask import Flask, render_template, request
import pandas as pd

# Import the optimization functions
from portfolio_optimizer import mean_variance_optimization, conditional_value_at_risk_optimization
from cumsum import *


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
        weights, return1, volatility = mean_variance_optimization(data, allocations)

    elif request.form['technique'] == 'cvar':
        weights, return1, volatility = conditional_value_at_risk_optimization(data, allocations)

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

    plt.figure(figsize=(12, 6))
    plt.plot(currentreturn1, label = "Current Portfolio")
    plt.plot(optimizedreturn1, label = "Optimized Portfolio")
    plt.title('Portfolio Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns (%)')
    plt.grid(True)
    plt.show()

    # Pass the optimization results to the template
    return render_template('result.html',
                           ticker_symbols=ticker_symbols,
                           allocations=allocations,
                           weights=weights,
                           currentreturn=currentreturn,
                           optimizedreturn=optimizedreturn,
                           volatility=volatility)



if __name__ == '__main__':
    app.run(debug=True)