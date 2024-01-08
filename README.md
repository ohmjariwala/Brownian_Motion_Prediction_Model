# Brownian_Motion_Prediction_Model

This model returns the expected stock price of a stock using the Geometric Brownian Motion conditions and model. It would be best suited for **short-term projections** and could be further built out to give a more accurate long-term stock price projection.

# Geometric Brownian Motion (GBM):
- Implements the GBM model, a stochastic process used to simulate future stock movement.
- Calculates drift coefficient (μ) and volatility (σ) essential for GBM using log returns, Beta, and covariance between stock and market returns.
- Executes Monte Carlo simulations to generate multiple potential future stock price paths based on the GBM formula.

The expected stock price is calculated by finding the average of the price at the end of each Monte Carlo simulation.

# Simulation and Visualization:
- Plots simulated stock price trajectories over time, depicting the potential variability in future stock prices.
- Outputs statistics including expected stock price, maximum and minimum projected stock values at a specified time interval.

# Application and Relevance:
- Showcases the practical application of quantitative methods in forecasting financial asset movements, aiding in risk assessment and investment decision-making.
- Highlights the use of Python libraries (YFinance, NumPy, Pandas, Matplotlib) and statistical models to simulate and visualize stock price dynamics.
- Conducts statistical analysis, including linear regression, to quantify the relationship between the stock and market returns. Derives the Beta (β) as a measure of the stock's volatility relative to the market.
