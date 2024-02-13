import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import re

class Portfolio:
    def __init__(self, tickers, start_date, end_date, interval, sustainalytics_scores,frequency_returns):
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
        # Frequency of our returns (should be 'M' or 'Y')
        self.frequency_returns=frequency_returns

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
        self.net_returns = self.create_stock_dataframe().pct_change()[1:]
        if self.frequency_returns=='Y':
            # Create annualized dataframe
            window_size = 12  # Specify the size of the rolling window (12 for annual)
            returns_annualized = (1 + self.net_returns).rolling(window=window_size).apply(lambda x: (x.prod() - 1)*12, raw=False)
            self.net_returns = returns_annualized.dropna()
            grouped_df_annualized = returns_annualized.mean()
            self.risk_annualized_free_rate = (grouped_df_annualized.mean()/12).min()-0.002
        return self

    def get_sector_for_tickers(self):
        _ticker_sector_dict = {}
        for ticker in self.tickers:
            try:
                ticker_info = yf.Ticker(ticker)
                sector = ticker_info.info["sector"]
                _ticker_sector_dict[ticker] = sector
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                _ticker_sector_dict[ticker] = None     
        return _ticker_sector_dict

    def download_data(self):
        # create net returns
        self.get_net_returns()
        # create a dictionary with the ticker and their sectors
        self.ticker_sector_dict = self.get_sector_for_tickers()
        # Keep tracks of optimizing functions called
        self.get_optimal_portfolio_called = False
        self.get_efficient_frontier_data_multiple_max_esg_scores_called = False
        return self

    ##############################
    # Optimization functions     #
    ##############################

    # Expected next gross return (naive approach with empirical mean)
    def get_mu_hat(self):
        _mu_hat = self.net_returns.mean(axis=0).values
        return _mu_hat

    # Covariance matrix of next the gross returns (naive approach with empirical covariance)
    def get_omega_hat(self):    
        _mu_hat = self.get_mu_hat()
        Y_minus_mu_hat = self.net_returns - _mu_hat

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
    
    def get_optimal_portfolio_markowitz(self, 
                              gammas, 
                              risk_free_rate=0,
                              max_esg_score=np.inf,
                              fully_invested=True,
                              long_only=True,
                              best_in_class_method=1):
        """_summary_

        Args:
            gamma (_type_): Inverse of the risk aversion parameter
            strategy (_type_): 'fully_invested' or 'long_only'

        Returns:
            _type_: _description_
        """
        # Compute mu_hat and omega_hat
        self.mu_hat = self.get_mu_hat()
        self.omega_hat = self.get_omega_hat()

        # Initial weights
        _initial_weights = np.ones(len(self.tickers)) / len(self.tickers)

        # Define bounds strategy
        if best_in_class_method<1:
            list_columns=self.stock_dataframe.columns
            esg_stocks=pd.DataFrame(self.sustainalytics_scores,index=list_columns,columns=['ESG Score'])
            percentage_to_keep = best_in_class_method
            esg_stocks_top_percent = esg_stocks.apply(lambda row: row[row <= row.quantile(percentage_to_keep)], axis=0)
            if long_only == True:
                self.bounds=[(0, 1) if ticker in esg_stocks_top_percent.index 
                            else (0,0) 
                            for ticker in self.tickers]
            else:
                self.bounds=[(-1,1) if ticker in esg_stocks_top_percent.index 
                            else (0,0) 
                            for ticker in self.tickers]
        else:
            if long_only == True:
                self.bounds = [(0, 1) for ticker in self.tickers]
            else:
                self.bounds = [(-1, 1) for ticker in self.tickers]
    
        # Define contraints
        _constraints = []
        if fully_invested:
            _constraints.append({'type': 'eq',
                                 'fun': self.constraint_fully_invested})
        if max_esg_score != np.inf:
            _constraints.append({'type': 'ineq',
                                 'fun': self.constraint_max_esg_score,
                                 'args': (self.sustainalytics_scores, max_esg_score)})

        # Lists to store results
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
            self.sigma.append(np.sqrt(np.dot(_result.x.T, np.dot(self.omega_hat, _result.x))))
            self.objective_values.append(-_result.fun)
            self.esg_scores.append(np.dot(self.sustainalytics_scores, list(_result.x)))

        # Tangent portfolio
        self.sharpe_ratio = [(a - risk_free_rate) / b for a, b in zip(self.mu, self.sigma)]
        self.sharpe_ratio_index = np.argmax(self.sharpe_ratio)
        self.sharpe_ratio_tagent = self.sharpe_ratio[self.sharpe_ratio_index]
        self.gamma_tangent = gammas[self.sharpe_ratio_index]
        self.mu_tangent = self.mu[self.sharpe_ratio_index]
        self.sigma_tangent = self.sigma[self.sharpe_ratio_index]
        self.weights_tangente_portfolio = self.optimal_weights[self.sharpe_ratio_index]
        self.score_esg_tangent = self.esg_scores[self.sharpe_ratio_index]
        # Keep tracks of optimizing functions called
        self.get_optimal_portfolio_called = True
        return self

    # def Cab(self,a,b):
    #     return a @ np.linalg.inv(self.omega_hat) @ b

    # def objective_perderson(self,weights, gamma, returns, cov_matrix):
    #     s_ = np.dot(self.sustainalytics_scores,weights)
    #     Cmumu = self.Cab(returns,returns)
    #     Csmu = self.Cab(self.sustainalytics_scores,returns)
    #     C1mu = self.Cab(np.ones(len(returns)),returns)
    #     Css = self.Cab(self.sustainalytics_scores,self.sustainalytics_scores)
    #     C1s = self.Cab(np.ones(len(returns)),self.sustainalytics_scores)
    #     C11 = self.Cab(np.ones(len(returns)),np.ones(len(returns)))
    #     sharpe_s = np.sqrt( Cmumu  - ((Csmu  - s_ *C1mu) **2) / ( Css - 2 * s_ * C1s + (s_** 2) * C11)  ) 
    #     obj = sharpe_s ** 2 + 2 * gamma * s_
    #     return -obj

    # def get_optimal_portfolio_Pedersen(self, 
    #                           gammas, 
    #                           alpha,
    #                           risk_free_rate=0,
    #                           max_esg_score=np.inf,
    #                           fully_invested=True,
    #                           long_only=True,best_in_class_method=False):

    #     SUSTAINALYTICS_SCORES_ = np.array(self.sustainalytics_scores)*0.001

    #     if best_in_class_method:
    #         list_columns=self.stock_dataframe.columns
    #         esg_stocks=pd.DataFrame(self.sustainalytics_scores,index=list_columns,columns=['ESG Score'])
    #         percentage_to_keep = 0.8
    #         esg_stocks_top_80_percent = esg_stocks.apply(lambda row: row[row <= row.quantile(percentage_to_keep)], axis=0)
    #         self.bounds=[(-1,1) if ticker in esg_stocks_top_80_percent.index else (-1,0) for ticker in self.tickers ]

    #     elif long_only == True:
    #             self.bounds = [(0, 1) for ticker in self.tickers]
    #     else:
    #         self.bounds = [(-1, 1) for ticker in self.tickers]

    #     self.mu_hat = self.get_mu_hat()
    #     self.omega_hat = self.get_omega_hat()

    #     # Initial weights
    #     self.initial_weights = np.ones(len(self.tickers)) / len(self.tickers)

    #     # Define contraints
    #     _constraints = []

    #     if fully_invested:
    #         _constraints.append({'type': 'eq',
    #                              'fun': self.constraint_fully_invested})

    #     if max_esg_score != np.inf:
    #         _constraints.append({'type': 'ineq',
    #                              'fun': self.constraint_max_esg_score,
    #                              'args': (self.sustainalytics_scores, max_esg_score)})

    #     # Calculate the mean returns and covariance matrix
    #     mean_returns = self.mu_hat 
    #     cov_matrix = self.omega_hat
    #     # Define constraints
    #     constraints = ({'type': 'eq', 'fun': self.constraint_fully_invested}) # fully invested portfolio
    #     bounds_ = [(-1, 1)]* len(self.tickers) 

    #     # Initialize weights
    #     # Set gamma values
    #     gamma_values = np.linspace(-0.5,0.5 , 100)  # Inverse of the risk aversion parameter

    #     risk_free_rate_pederson=0.03
    #     # Lists to store results
    #     self.sharpe_ratio = []
    #     self.esg_scores = []
    #     self.weights_gamma = []
    #     self.objective_values = []
    #     self.mu=[]
    #     self.sigma=[]

    #     # Solve the problem for different values of gamma
    #     for gamma_val in gamma_values:

    #         result = minimize(self.objective_perderson, self.initial_weights, args=(gamma_val, mean_returns, cov_matrix),
    #                         constraints=constraints, bounds=bounds_)

    #         # Save results
    #         self.weights_gamma.append(list(result.x))
    #         self.sharpe_ratio.append((np.dot(mean_returns, result.x)-risk_free_rate_pederson) / np.sqrt(np.dot(result.x.T, np.dot(cov_matrix, result.x))))
    #         self.esg_scores.append(np.dot(SUSTAINALYTICS_SCORES_,result.x))
    #         self.objective_values.append(-result.fun)
    #         self.mu.append(np.dot(self.mu_hat, result.x))
    #         self.sigma.append(np.sqrt(np.dot(result.x.T, np.dot(self.omega_hat, result.x))))
    #     self.sharpe_ratio_index = np.argmax(self.sharpe_ratio)
    #     self.sharpe_ratio_tagent = self.sharpe_ratio[self.sharpe_ratio_index]
    #     self.gamma_tangent = gammas[self.sharpe_ratio_index]
    #     self.mu_tangent = self.mu[self.sharpe_ratio_index]
    #     self.sigma_tangent = self.sigma[self.sharpe_ratio_index]
    #     self.score_esg_tangent = self.esg_scores[self.sharpe_ratio_index]
    #     return self

    def get_efficient_frontier_data_multiple_max_esg_scores(self,
                                                        gammas,
                                                        risk_free_rate=0,
                                                        max_esg_scores=[np.inf],
                                                        fully_invested=True,
                                                        long_only=True,
                                                        best_in_class_method=1,
                                                        opt_problem='Markowitz'):

        self.multiple_esg_simulations = {}

        for max_esg_score in max_esg_scores:
            if opt_problem=='Markowitz':
                self.get_optimal_portfolio_markowitz(gammas,
                                        risk_free_rate,
                                        max_esg_score,
                                        fully_invested,
                                        long_only,best_in_class_method)
                self.multiple_esg_simulations[max_esg_score] = {'mu': self.mu,
                                            'sigma': self.sigma,
                                            'esg_scores': self.esg_scores,
                                            'sharpe_ratio': self.sharpe_ratio,
                                            'sharpe_ratio_tagent': self.sharpe_ratio_tagent,
                                            'gamma_tangent': self.gamma_tangent,
                                            'mu_tangent': self.mu_tangent,
                                            'sigma_tangent': self.sigma_tangent,
                                            'score_esg_tangent': self.score_esg_tangent,
                                            'weights_tangente_portfolio': self.weights_tangente_portfolio}
            
            elif opt_problem=='Pedersen':
                self.get_optimal_portfolio_Pedersen(gammas,
                                        risk_free_rate,
                                        max_esg_score,
                                        fully_invested,
                                        long_only,best_in_class_method)
                self.multiple_esg_simulations[max_esg_score] = {'mu': self.mu,
                                            'sigma': self.sigma,
                                            'esg_scores': self.esg_scores,
                                            'sharpe_ratio': self.sharpe_ratio,
                                            'sharpe_ratio_tagent': self.sharpe_ratio_tagent,
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
                                risk_free_rate=0,
                                max_esg_score=np.inf,
                                fully_invested=True,
                                long_only=True,
                                best_in_class_method=1,
                                opt_problem='Markowitz'):

        if ~self.get_optimal_portfolio_called:
            if opt_problem=='Markowitz':
                self.get_optimal_portfolio_markowitz(gammas,
                                        risk_free_rate,
                                        max_esg_score,
                                        fully_invested,
                                        long_only,
                                        best_in_class_method)
            elif opt_problem=='Pedersen':
                self.get_optimal_portfolio_Pedersen(gammas,
                                        risk_free_rate,
                                        max_esg_score,
                                        fully_invested,
                                        long_only,best_in_class_method)

        if max_esg_score!=np.inf:
            _c = self.esg_scores
            color = 'ESG score'
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
            # ticks = np.linspace(np.min(_c), np.max(_c), 5)  # Adjust the number of ticks as needed
            # tick_labels = [f"{x:.1f}" for x in ticks]

            # cbar = plt.colorbar(label=f'Values with max difference of {_lisible_max_diff}', ticks=ticks)
            # cbar.set_ticks(ticks)
            # cbar.set_ticklabels(tick_labels)
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

        else:
            _c = gammas
            color = 'gamma'
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
            ticks = np.linspace(np.min(_c), np.max(_c), 5)  # Adjust the number of ticks as needed
            tick_labels = [f"{x:4f}" for x in ticks]

            cbar = plt.colorbar(label=f'Values with max difference of {_lisible_max_diff}', ticks=ticks)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(tick_labels)
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
                                                        risk_free_rate=0,
                                                        max_esg_scores=[np.inf],
                                                        fully_invested=True,
                                                        long_only=True,
                                                        best_in_class_method=1,
                                                        with_optimal_portfolio=False,
                                                        with_linear_tangent=False,
                                                        opt_problem='Markowitz'):
        
        if ~self.get_efficient_frontier_data_multiple_max_esg_scores_called:
            self.get_efficient_frontier_data_multiple_max_esg_scores(gammas,
                                                                     risk_free_rate,
                                                                     max_esg_scores,
                                                                     fully_invested,
                                                                     long_only,
                                                                     best_in_class_method,
                                                                     opt_problem)

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
                                risk_free_rate=0,
                                max_esg_scores=np.inf,
                                fully_invested=True,
                                long_only=True,
                                best_in_class_method=1,
                                opt_problem='Markowitz'):

        _sharpe_ratio_tangent = []

        for max_esg_score in max_esg_scores:
            if opt_problem=='Markowitz':
                self.get_optimal_portfolio_markowitz(gammas,
                                        risk_free_rate,
                                        max_esg_score,
                                        fully_invested,
                                        long_only,
                                        best_in_class_method)
                
            elif opt_problem=='Pedersen':
                self.get_optimal_portfolio_Pedersen(gammas,
                                        risk_free_rate,
                                        max_esg_score,
                                        fully_invested,
                                        long_only,
                                        best_in_class_method)
                
            _sharpe_ratio_tangent.append(self.sharpe_ratio_tagent)
        
        # Plot the efficient frontier
        plt.figure(figsize=(10, 6))
        plt.plot(max_esg_scores, _sharpe_ratio_tangent, marker='o', linestyle='-', color='b')
        plt.title('Sharpe Ratio of Tangent Portfolio vs Max ESG Score Constraint')
        plt.xlabel('Max ESG Score Constraint')
        plt.ylabel('Sharpe Ratio')
        plt.grid(True)
        plt.show()



    def plot_weights_evolution(self,
                                gammas,
                                risk_free_rate=0,
                                max_esg_scores=[np.inf],
                                fully_invested=True,
                                long_only=True,
                                best_in_class_method=1,
                                opt_problem='Markowitz'):
        
        if ~self.get_efficient_frontier_data_multiple_max_esg_scores_called:
            self.get_efficient_frontier_data_multiple_max_esg_scores(gammas,
                                                                     risk_free_rate,
                                                                     max_esg_scores,
                                                                     fully_invested,
                                                                     long_only,
                                                                     best_in_class_method,
                                                                     opt_problem)

        self.weights_evolution_with_max_score = [self.multiple_esg_simulations[max_score]['weights_tangente_portfolio'] for max_score in max_esg_scores]

        self.dict_weights_evolution = {ticker: np.array(self.weights_evolution_with_max_score)[:, i] for i, ticker in enumerate(self.ticker_sector_dict)}

        width = 0.7

        fig, ax = plt.subplots(figsize=(15, 5))
        bottom = np.zeros(len(max_esg_scores))

        for boolean, weight in self.dict_weights_evolution.items():
            p = ax.bar(np.arange(len(max_esg_scores)), weight, width, label=boolean, bottom=bottom)
            bottom += weight

        ax.set_title("Evolution of the weights of the stocks in the portfolio with the maximum ESG score constraint")
        ax.set_xticks(np.arange(len(max_esg_scores)))  
        ax.set_xticklabels(max_esg_scores)  
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.xlabel('Max ESG Score Constraint')
        plt.ylabel('Weights')

        plt.show()

    # def plot_weights_evolution(self,
    #                             gammas,
    #                             risk_free_rate=0,
    #                             max_esg_scores=[np.inf],
    #                             fully_invested=True,
    #                             long_only=True,
    #                             best_in_class_method=1,
    #                             opt_problem='Markowitz'):
        
    #     if not self.get_efficient_frontier_data_multiple_max_esg_scores_called:
    #         self.get_efficient_frontier_data_multiple_max_esg_scores(gammas,
    #                                                                 risk_free_rate,
    #                                                                 max_esg_scores,
    #                                                                 fully_invested,
    #                                                                 long_only,
    #                                                                 best_in_class_method,
    #                                                                 opt_problem)

    #     self.weights_evolution_with_max_score = [self.multiple_esg_simulations[max_score]['weights_tangente_portfolio'] for max_score in max_esg_scores]

    #     self.dict_weights_evolution = {ticker: np.array(self.weights_evolution_with_max_score)[:, i] for i, ticker in enumerate(self.ticker_sector_dict)}

    #     width = 0.7

    #     fig, ax = plt.subplots(figsize=(15, 5))
    #     bottom = np.zeros(len(max_esg_scores))

    #     for boolean, weight in self.dict_weights_evolution.items():
    #         p = ax.bar(np.arange(len(max_esg_scores)), weight, width, label=boolean, bottom=bottom)
    #         bottom += weight

    #         # Add value text in the middle of each bar
    #         for i, v in enumerate(weight):
    #             ax.text(i, v / 2, f'{v:.2f}', ha='center', va='center', color='white', fontweight='bold')

    #     ax.set_title("Evolution of the weights of the stocks in the portfolio with the maximum ESG score constraint")
    #     ax.set_xticks(np.arange(len(max_esg_scores)))  
    #     ax.set_xticklabels(max_esg_scores)  
    #     ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    #     plt.xlabel('Max ESG Score Constraint')
    #     plt.ylabel('Weights')

    #     plt.show()