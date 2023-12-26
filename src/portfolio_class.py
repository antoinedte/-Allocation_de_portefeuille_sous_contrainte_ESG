import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import re

class Portfolio:
    def __init__(self, tickers, start_date, end_date, interval, sustainalytics_scores):
        # List of tickers to download
        self.tickers = tickers
        # Download start date string (YYYY-MM-DD) or _datetime.
        self.start_date = start_date
        # Download end date string (YYYY-MM-DD) or _datetime.
        self.end_date = end_date
        # Interval between stock data points
        self.interval = interval
        # Sustainalytics scores of the stocks
        self.sustainalytics_scores = sustainalytics_scores
    
    ##############################
    # download functions         #
    ##############################
        
    def create_stock_dataframe(self):
        """Download stock data from Yahoo Finance and create a dataframe with the adjusted close price.
        Args:
            interval (str, optional): Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo. 
                                        Intraday data cannot extend last 60 days. Defaults to '1mo'.
        Returns:
            _type_: _description_
        """
        # Pour télécharger les données boursières avec une fréquence mensuelle "1mo"
        stock_data = yf.download(tickers=self.tickers, 
                                    start=self.start_date,
                                    end=self.end_date,
                                    interval=self.interval)
        # Pour extraire et stocker les prix de clôture ajustés "Adj Close"
        self.stock_dataframe = stock_data['Adj Close']
        return self.stock_dataframe
    
    def get_net_returns(self):
        _net_returns = self.create_stock_dataframe().pct_change()[1:]
        return _net_returns
    
    def get_gross_returns(self):
        _gross_returns = self.get_net_returns() + 1
        return _gross_returns

    def download_data(self):
        # create net returns
        self.net_returns = self.get_net_returns()
        # create gross returns
        self.gross_returns = self.get_gross_returns()
        # Keep tracks of optimizing functions called
        self.get_optimal_portfolio_called = False
        self.get_efficient_frontier_data_multiple_max_esg_scores_called = False

        return self

    ##############################
    # Optimization functions     #
    ##############################

    # Expected next gross return (naive approach with empirical mean)
    def get_mu_hat(self):
        _mu_hat = self.gross_returns.mean(axis=0).values
        return _mu_hat

    # Covariance matrix of next the gross returns (naive approach with empirical covariance)
    def get_omega_hat(self):    
        _mu_hat = self.get_mu_hat()
        Y_minus_mu_hat = self.gross_returns - _mu_hat

        _omega_hat = np.zeros((len(Y_minus_mu_hat.columns),len(Y_minus_mu_hat.columns)))
        
        for date in Y_minus_mu_hat.index:
            Y_date_minus_mu_hat = Y_minus_mu_hat.loc[date].values.reshape(len(Y_minus_mu_hat.columns), 1)
            matrix_product_for_date = np.dot(Y_date_minus_mu_hat, Y_date_minus_mu_hat.T)
            _omega_hat += matrix_product_for_date

        _omega_hat /= len(_omega_hat)
        
        return _omega_hat


    def gamma_markowitz_objective(self, weights, gamma, returns, cov_matrix):
        """_summary_

        Args:
            weights (_type_): Column vector of the weights
            gamma (_type_): Inverse of the risk aversion parameter
            returns (_type_):  Column vector of the expected returns
            cov_matrix (_type_): Matrix of covariance

        Returns:
            _type_: _description_
        """
        x_star = 1/2 * weights.T @ cov_matrix @ weights - gamma * weights.T @ returns
        return x_star

    def constraint_fully_invested(self, x_weights):
        return sum(x_weights) - 1
    
    def constraint_max_esg_score(self, x_weights, sustainalytics_scores, max_esg_score):
        return - (np.dot(sustainalytics_scores, x_weights) - max_esg_score)
    
    def get_optimal_portfolio(self, 
                              gammas, 
                              alpha,
                              risk_free_rate=0,
                              max_esg_score=np.inf,
                              fully_invested=True,
                              long_only=True):
        """_summary_

        Args:
            gamma (_type_): Inverse of the risk aversion parameter
            alpha (_type_): Proportion of risk-free asset in the portfolio
            strategy (_type_): 'fully_invested' or 'long_only'

        Returns:
            _type_: _description_
        """
        # Compute mu_hat and omega_hat
        self.mu_hat = self.get_mu_hat()
        self.omega_hat = self.get_omega_hat()

        # Initial weights
        _initial_weights = np.ones(len(self.tickers)) / len(self.tickers)

        # Define contraints
        _constraints = []

        if fully_invested:
            _constraints.append({'type': 'eq',
                                 'fun': self.constraint_fully_invested})

        if max_esg_score != np.inf:
            _constraints.append({'type': 'ineq',
                                 'fun': self.constraint_max_esg_score,
                                 'args': (self.sustainalytics_scores, max_esg_score)})

        # Define bounds strategy
        if long_only == True:
            self.bounds = [(0, 1) for ticker in self.tickers]
        else:
            self.bounds = [(-1, 1) for ticker in self.tickers]
        
        self.optimal_weights = []
        self.mu, self.mu_y = [], []
        self.sigma, self.sigma_y = [], []
        self.objective_values = []
        self.esg_scores = []

        # Optimize
        for gamma in gammas:
            _result = minimize(fun=self.gamma_markowitz_objective, 
                               x0=_initial_weights,
                               args=(gamma, self.mu_hat, self.omega_hat),
                               constraints=tuple(_constraints),
                               bounds=self.bounds)
            
            self.optimal_weights.append(list(_result.x))
            self.mu.append(np.dot(self.mu_hat, _result.x))
            self.mu_y.append((1 - alpha) * risk_free_rate + alpha * self.mu[-1])
            self.sigma.append(np.sqrt(np.dot(_result.x.T, np.dot(self.omega_hat, _result.x))))
            self.sigma_y.append(alpha * self.sigma[-1])
            self.objective_values.append(-_result.fun)
            self.esg_scores.append(np.dot(self.sustainalytics_scores, list(_result.x)))

        # Tangent portfolio
        self.theta = [(a - risk_free_rate) / b for a, b in zip(self.mu_y, self.sigma_y)]
        self.tangente_index = np.argmax(self.theta)
        self.theta_tagent = self.theta[self.tangente_index]
        self.gamma_tangent = gammas[self.tangente_index]
        self.mu_tangent = self.mu[self.tangente_index]
        self.sigma_tangent = self.sigma[self.tangente_index]
        self.score_esg_tangent = self.esg_scores[self.tangente_index]

        # Keep tracks of optimizing functions called
        self.get_optimal_portfolio_called = True

        return self
    
    def get_efficient_frontier_data_multiple_max_esg_scores(self,
                                                        gammas,
                                                        alpha,
                                                        risk_free_rate=0,
                                                        max_esg_scores=[np.inf],
                                                        fully_invested=True,
                                                        long_only=True):

        self.multiple_esg_simulations = {}

        for max_esg_score in max_esg_scores:
            self.get_optimal_portfolio(gammas,
                                    alpha,
                                    risk_free_rate,
                                    max_esg_score,
                                    fully_invested,
                                    long_only)
            self.multiple_esg_simulations[max_esg_score] = {'mu': self.mu,
                                        'sigma': self.sigma,
                                        'esg_scores': self.esg_scores,
                                        'mu_y': self.mu_y,
                                        'sigma_y': self.sigma_y,
                                        'theta': self.theta,
                                        'theta_tagent': self.theta_tagent,
                                        'gamma_tangent': self.gamma_tangent,
                                        'mu_tangent': self.mu_tangent,
                                        'sigma_tangent': self.sigma_tangent,
                                        'score_esg_tangent': self.score_esg_tangent}

        # Keep tracks of optimizing functions called
        self.get_efficient_frontier_data_multiple_max_esg_scores_called = True

        return self
        


    ##############################
    # plotting functions         #
    ##############################

    def plot_efficient_frontier(self,
                                gammas,
                                alpha,
                                risk_free_rate=0,
                                max_esg_score=np.inf,
                                fully_invested=True,
                                long_only=True):

        if ~self.get_optimal_portfolio_called:
            self.get_optimal_portfolio(gammas,
                                       alpha,
                                       risk_free_rate,
                                       max_esg_score,
                                       fully_invested,
                                       long_only)

        if max_esg_score!=np.inf:
            _c = self.esg_scores
            color = 'ESG score'
        else:
            _c = gammas
            color = 'gamma'

        # Plot the efficient frontier
        plt.figure(figsize=(10, 6))
        plt.scatter(self.sigma, 
                    self.mu, 
                    s=20, 
                    c=_c, 
                    cmap='viridis', 
                    label='Efficient Frontier')

        # Convert max difference to string
        _max_diff = str(np.max(_c) - np.min(_c))
        _pattern = r'(\d\.\d{1})\d*e(-\d+)'
        _replacement = r'\1e\2'
        _lisible_max_diff = float(re.sub(_pattern, _replacement, _max_diff))
        plt.colorbar(label=f'Values of {color}\nwith max difference of {_lisible_max_diff}',
                     format="{x:.2f}")
    
        # Plot the optimal portfolio
        plt.scatter(x=self.sigma_tangent,
                    y=self.mu_tangent,
                    marker='X',
                    s=50,
                    c='r',
                    label='Optimal Portfolio')
        
        # Plot the tangent portfolio
        abscisses = [0, np.max(self.sigma)]
        ordonnées = [risk_free_rate,
                    risk_free_rate + (self.mu_tangent - risk_free_rate) / self.sigma_tangent * np.max(self.sigma)]
        plt.plot(abscisses, ordonnées, alpha=0.7, c='grey')

        # Annotate the plot
        plt.title(f'Efficient Frontier with Varying Gamma reach optimum at {np.round(self.gamma_tangent,4)} and ESG score of {np.round(self.score_esg_tangent, 4)}.')
        plt.xlabel('Portfolio Standard Deviation')
        plt.ylabel('Portfolio Return')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_efficient_frontier_multiple_max_esg_scores(self,
                                                        gammas,
                                                        alpha,
                                                        risk_free_rate=0,
                                                        max_esg_scores=[np.inf],
                                                        fully_invested=True,
                                                        long_only=True,
                                                        with_optimal_portfolio=False,
                                                        with_linear_tangent=False):
        
        if ~self.get_efficient_frontier_data_multiple_max_esg_scores_called:
            self.get_efficient_frontier_data_multiple_max_esg_scores(gammas,
                                                                     alpha,
                                                                     risk_free_rate,
                                                                     max_esg_scores,
                                                                     fully_invested,
                                                                     long_only)

        # Plot the efficient frontier
        plt.figure(figsize=(10, 6))

        colors = plt.cm.tab10(range(len(max_esg_scores)))

        for max_esg_score in max_esg_scores:
            # Get the color index based on the value
            color_index = max_esg_scores.index(max_esg_score)

            plt.plot(self.multiple_esg_simulations[max_esg_score]['sigma'], 
                        self.multiple_esg_simulations[max_esg_score]['mu'], 
                        c=colors[color_index],
                        label=f'Efficient Frontier with max ESG score of {max_esg_score}')
        
        plt.xlim(left=0)

        if with_optimal_portfolio or with_linear_tangent:
            for max_esg_score in max_esg_scores:
                # Plot the optimal portfolio
                if with_optimal_portfolio:
                    plt.scatter(x=self.multiple_esg_simulations[max_esg_score]['sigma_tangent'],
                                y=self.multiple_esg_simulations[max_esg_score]['mu_tangent'],
                                marker='X',
                                s=50,
                                c='r',
                                label=f'Optimal Portfolio for max ESG score of {max_esg_score}')
                
                if with_linear_tangent:
                    # Plot the tangent portfolio
                    abscisses = [0, np.max(self.multiple_esg_simulations[max_esg_score]['sigma'])]
                    ordonnées = [risk_free_rate,
                                risk_free_rate + (self.multiple_esg_simulations[max_esg_score]['mu_tangent'] - risk_free_rate) / self.multiple_esg_simulations[max_esg_score]['sigma_tangent'] * np.max(self.multiple_esg_simulations[max_esg_score]['sigma'])]
                    plt.plot(abscisses, ordonnées, alpha=0.7, c='grey')
        
        # Annotate the plot
        plt.title(f'Multiple Efficient Frontiers with Varying Max ESG score')
        plt.xlabel('Portfolio Standard Deviation')
        plt.ylabel('Portfolio Return')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_sharp_ratio_vs_max_score(self,
                                gammas,
                                alpha,
                                risk_free_rate=0,
                                max_esg_scores=np.inf,
                                fully_invested=True,
                                long_only=True):

        _theta_tangent = []

        for max_esg_score in max_esg_scores:
            self.get_optimal_portfolio(gammas,
                                       alpha,
                                       risk_free_rate,
                                       max_esg_score,
                                       fully_invested,
                                       long_only)
            
            _theta_tangent.append(self.theta_tagent)
        
        # Plot the efficient frontier
        plt.figure(figsize=(10, 6))
        plt.plot(max_esg_scores, _theta_tangent, marker='o', linestyle='-', color='b')
        plt.title('Sharpe Ratio of Tangent Portfolio vs Max ESG Score Constraint')
        plt.xlabel('Max ESG Score Constraint')
        plt.ylabel('Sharpe Ratio')
        plt.grid(True)
        plt.show()