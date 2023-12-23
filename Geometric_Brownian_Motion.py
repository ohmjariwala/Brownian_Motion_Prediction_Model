from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from sklearn.linear_model import LinearRegression
import pandas as pd

class Geometric_Brownian_Motion:
    start_date = dt.datetime(2018, 1, 1)
    end_date=dt.datetime.now()
    stock_ticker= input("Stock Ticker?")
    market_index_ticker = '^GSPC'  # S&P 500 index ticker
    
    # Download data for the stock and market
    stock_data = yf.download(stock_ticker, start_date, end_date)['Adj Close']
    market_data = yf.download(market_index_ticker, start_date, end_date)['Adj Close']
    
    #| Beta
    stock_returns = stock_data.pct_change().dropna()
    market_returns = market_data.pct_change().dropna()

    # Organize Returns data into pandas dataframe
    data = pd.DataFrame({'Stock_Returns': stock_returns, 'Market_Returns': market_returns})
    data = data.dropna()

    # Extract X and Y variables for regression
    X = data['Market_Returns'].values.reshape(-1, 1)
    Y = data['Stock_Returns'].values.reshape(-1, 1)

    # Fit linear regression model and compute Beta
    model = LinearRegression()
    model.fit(X,Y)
    beta = model.coef_[0][0]

    #| Covariance
    cov_data = pd.concat([stock_data, market_data], axis=1)
    cov_data.columns = ['Stock', 'Market']

    # Calculate daily returns
    returns = cov_data.pct_change().dropna()

    # Calculate covariance between stock and market returns
    covariance = returns['Stock'].cov(returns['Market'])


    #| GBM
      
    # drift coefficient -> mu
    mu = (stock_data.pct_change().dropna().mean().round(5))
    # number of steps -> n
    n = 1000
    # time in years -> t
    t = 1/12
    # number of sims -> simulations
    simulations = 1000
    # initial stock price -> So
    So = stock_data.iloc[len(stock_data)-1].round(4)
    # Volatility -> Sigma
    sigma = round(sqrt(covariance/ beta) *sqrt(21) , 5)
    

    # Calculate each time step
    dt = t/n

    # Simulation using numpy arrays
    St = np.exp( (mu - sigma ** 2 / 2) * dt + sigma * np.random.normal(0, np.sqrt(dt), size=(simulations,n)).T)

    # Include array of 1's
    St = np.vstack([np.ones(simulations), St])

    # Multiply through by initial stock price and return the cumulative product of elements along a given simulation path. 
    St = So * St.cumprod(axis=0)

    # Time interval
    time = np.linspace(0,t,n+1)

    # Require numpy array that is the same shape as St
    tt = np.full(shape=(simulations,n+1), fill_value=time).T

    
    # Generate a standard normal random variable
    Z = np.random.normal(0, 1, simulations)

    # Find the end values of each simulation and add to an array
    expected_stock_price_array=[]
    for i in range(simulations):
        final_value = St[:, i][-1]  # Extract the final value for the i-th simulation
        expected_stock_price_array.append(final_value) 
    
    # Calculate expected stock price by taking the mean of the ending prices of all simulations
    expected_stock_price= np.mean(expected_stock_price_array)
    
    # Find the maximum value of the stock at the end of the simulation
    max_stock_value = np.max(St[-1])
    
    # Find the minimum value of the stock at the end of the simulation
    min_stock_value= np.min(St[-1])


    # Print Statements
    print(f"Expected stock price at time t={t*12} months: {expected_stock_price:.2f}")
    print(f"Maximum stock value at time t={t*12} months: {max_stock_value:.2f}")
    print(f"Minimum stock value at time t={t*12} months: {min_stock_value:.2f}")

    # Plot
    plt.plot(tt, St)
    plt.xlabel("Years $(t)$")
    plt.ylabel("Stock Price $(S_t)$")
    plt.title("Geometric Brownian Motion for " + stock_ticker + "\n $dS_t = \mu S_t dt + \sigma S_t dW_t$\n $S_0 = {0}, \mu = {1}, \sigma = {2}$".format(So, mu, sigma))
    plt.show()
