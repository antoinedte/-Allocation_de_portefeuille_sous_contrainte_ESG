import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import re


class Portfolio:
    def __init__(self, tickers, start_date, end_date, interval, msci_data, frequency_returns):
        # List of tickers to download
        self.tickers = tickers
        # Download start date string (YYYY-MM-DD) or _datetime.
        self.start_date = start_date
        # Download end date string (YYYY-MM-DD) or _datetime.
        self.end_date = end_date
        # Interval between stock data points
        self.interval = interval
        # Sustainalytics scores of the stocks
        self.msci_data = msci_data
        # Frequency of our returns (should be 'M' or 'Y')
        self.frequency_returns = frequency_returns
        # Define colormap
        self.colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(self.tickers)))

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
    
    def constraint_max_esg_score(self, x_weights, msci_data, max_esg_score):
        return (np.dot(msci_data, x_weights) - max_esg_score)
    
    def constraint_at_least_x_percent_in_sector(self, x_weights, sector_min_weight_x_dict, ticker_sector_dict):
        if not isinstance(sector_min_weight_x_dict, dict):
            raise ValueError("sector_min_weight_x_dict should be a dictionary")
        else:
            list_of_min_weights_per_sector = []
            for sector, min_weight_x in sector_min_weight_x_dict.items():
                list_of_min_weights_per_sector.append(sum([x_weights[i] for i in range(len(x_weights)) if ticker_sector_dict[self.tickers[i]] == sector]) - min_weight_x)
        return list_of_min_weights_per_sector
    
    def compute_selection_exclusion_method(self,
                                           long_only = True,
                                           best_in_class_method = 1,
                                           best_in_class_strategy = 'global'):
        """_summary_

        Args:
            long_only (bool, optional): _description_. Defaults to True.
            best_in_class_method (float, optional): _description_. Defaults to 1.
            best_in_class_strategy (str, optional): _description_. Defaults to 'global'.

        Returns:
            _type_: _description_
        """
        # Define bounds strategy
        if best_in_class_method<1: # if we apply a best in class method 
            if best_in_class_strategy=='global': # if we apply this strategy on the global dataset
                list_columns=self.stock_dataframe.columns
                esg_stocks=pd.DataFrame(self.msci_data,index=list_columns,columns=['ESG Score'])
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

            elif best_in_class_strategy=='sector':# if we apply this strategy on each sector
                list_columns_=self.stock_dataframe.columns
                df_esg_stocks_=pd.DataFrame(self.msci_data,
                                         index=list_columns_,
                                         columns=['ESG_Score'])
                sectors_dict_ = self.ticker_sector_dict

                df_esg_stocks_['sector'] = [sectors_dict_[ticker] for ticker in df_esg_stocks_.index.tolist()]
                
                df_quantile_of_sector = df_esg_stocks_.groupby('sector').quantile(q=1-best_in_class_method, interpolation="lower")

                ticker_to_keep = {}

                for sect in set(sectors_dict_.values()):
                    quantile_of_sect = df_quantile_of_sector.loc[sect, "ESG_Score"]
                    ticker_to_keep[sect] = df_esg_stocks_.query(f'sector == "{sect}"').query(f"ESG_Score >= {quantile_of_sect}").index.tolist()

                esg_stocks_top_percent_per_sector_ = [ticker for sector_ticker in list(ticker_to_keep.values()) for ticker in sector_ticker]
                if long_only == True:
                    self.bounds=[(0, 1) if ticker in esg_stocks_top_percent_per_sector_ 
                                else (0,0) 
                                for ticker in self.tickers]
                else:
                    self.bounds=[(-1,1) if ticker in esg_stocks_top_percent_per_sector_
                                else (0,0) 
                                for ticker in self.tickers] 
        
        else: # if we don't apply a best in class method
            if long_only == True:
                self.bounds = [(0, 1) for ticker in self.tickers]
            else:
                self.bounds = [(-1, 1) for ticker in self.tickers]

        return self            

    def get_optimal_portfolio_markowitz(self, 
                              gammas, 
                              risk_free_rate=0,
                              max_esg_score=-np.inf,
                              fully_invested=True,
                              long_only=True,
                              best_in_class_method=1,
                              best_in_class_strategy='global',
                              sector_min_weight_x_dict=None):
        """_summary_

        Args:
            gammas (_type_): Inverse of the risk aversion parameter
            risk_free_rate (int, optional): _description_. Defaults to 0.
            max_esg_score (_type_, optional): _description_. Defaults to -np.inf.
            fully_invested (bool, optional): _description_. Defaults to True.
            long_only (bool, optional): True for long only, false else. Defaults to True.
            best_in_class_method (int, optional): _description_. Defaults to 1.
            best_in_class_strategy (str, optional): 'global' or 'sector'. Defaults to 'global'.

        Returns:
            _type_: _description_
        """
        #TODO rajouter dans returns l'actif sans risque de return constant égal à risk_free_rate et le mettre dnas l'optimisation

        # Compute mu_hat and omega_hat
        self.mu_hat = self.get_mu_hat()
        self.omega_hat = self.get_omega_hat()
        #TODO partir du principe théorique en rajoutant une ligne de cov nulle

        # Initial weights
        _initial_weights = np.ones(len(self.tickers)) / len(self.tickers)

        # Compute selection/exclusion method
        self.compute_selection_exclusion_method(long_only = long_only,
                                                best_in_class_method = best_in_class_method,
                                                best_in_class_strategy = best_in_class_strategy)
    
        # Define contraints
        _constraints = []
        if fully_invested:
            _constraints.append({'type': 'eq',
                                 'fun': self.constraint_fully_invested})
        if max_esg_score != np.inf:
            _constraints.append({'type': 'ineq',
                                 'fun': self.constraint_max_esg_score,
                                 'args': (self.msci_data, max_esg_score)})
        if sector_min_weight_x_dict != None:
            _constraints.append({'type': 'ineq',
                                    'fun': self.constraint_at_least_x_percent_in_sector,
                                    'args': (sector_min_weight_x_dict, self.ticker_sector_dict)})

        # Lists to store results
        self.optimal_weights = []
        self.mu, self.mu_y = [], []
        self.sigma, self.sigma_y = [], []
        self.objective_values = []
        self.esg_scores = []
        self.possible_solution = True

        # Optimize
        for gamma in gammas:
            _result = minimize(fun=self.gamma_markowitz_objective, 
                               x0=_initial_weights,
                               args=(gamma, self.mu_hat, self.omega_hat),
                               constraints=tuple(_constraints),
                               bounds=self.bounds)
            
            if _result.success:
                self.optimal_weights.append(list(_result.x))
                self.mu.append(np.dot(self.mu_hat, _result.x))
                self.sigma.append(np.sqrt(np.dot(_result.x.T, np.dot(self.omega_hat, _result.x))))
                self.objective_values.append(-_result.fun)
                self.esg_scores.append(np.dot(self.msci_data, list(_result.x)))

            else:
                self.possible_solution = False
                break
        
        if self.possible_solution:
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
        
        else :
            print(f"No solution found for esg score of {max_esg_score}")
            # Tangent portfolio
            self.sharpe_ratio = np.nan
            self.sharpe_ratio_index = np.nan
            self.sharpe_ratio_tagent = np.nan
            self.gamma_tangent = np.nan
            self.mu_tangent = np.nan
            self.sigma_tangent = np.nan
            self.weights_tangente_portfolio = [np.nan]*len(self.tickers)
            self.score_esg_tangent = np.nan
            # Keep tracks of optimizing functions called
            self.get_optimal_portfolio_called = True  

        return self.possible_solution



    def get_efficient_frontier_data_multiple_max_esg_scores(self,
                                                        gammas,
                                                        risk_free_rate=0,
                                                        max_esg_scores=[np.inf],
                                                        fully_invested=True,
                                                        long_only=True,
                                                        best_in_class_method=1,
                                                        best_in_class_strategy='global',
                                                        opt_problem='Markowitz',
                                                        sector_min_weight_x_dict=None):

        self.multiple_esg_simulations = {}

        for max_esg_score in max_esg_scores:
            if opt_problem=='Markowitz':
                if self.get_optimal_portfolio_markowitz(gammas,
                                                     risk_free_rate,
                                                     max_esg_score,
                                                     fully_invested,
                                                     long_only,
                                                     best_in_class_method,
                                                     best_in_class_strategy,
                                                     sector_min_weight_x_dict):
                
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
                else:
                    self.multiple_esg_simulations[max_esg_score] = {'mu': np.nan,
                                                'sigma': np.nan,
                                                'esg_scores': np.nan,
                                                'sharpe_ratio': np.nan,
                                                'sharpe_ratio_tagent': np.nan,
                                                'gamma_tangent': np.nan,
                                                'mu_tangent': np.nan,
                                                'sigma_tangent': np.nan,
                                                'score_esg_tangent': np.nan,
                                                'weights_tangente_portfolio': [np.nan]*len(self.tickers)}
            
            elif opt_problem=='Pedersen':
                self.get_optimal_portfolio_Pedersen(gammas,
                                        risk_free_rate,
                                        max_esg_score,
                                        fully_invested,
                                        long_only,
                                        best_in_class_method)
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
                                best_in_class_strategy='global',
                                opt_problem='Markowitz',
                                sector_min_weight_x_dict=None):

        # if ~self.get_optimal_portfolio_called:
        if opt_problem=='Markowitz':
            self.get_optimal_portfolio_markowitz(gammas=gammas,
                                    risk_free_rate=risk_free_rate,
                                    max_esg_score=max_esg_score,
                                    fully_invested=fully_invested,
                                    long_only=long_only,
                                    best_in_class_strategy=best_in_class_strategy,
                                    best_in_class_method=best_in_class_method,
                                    sector_min_weight_x_dict=sector_min_weight_x_dict)

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
            plt.xlim((0, 3.5))
            plt.ylabel('Portfolio Return')
            plt.ylim((0, 10))
            plt.legend(loc='upper left')
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

    def plot_tangente_portfolio_composition(self,
                                            gammas, 
                                            risk_free_rate=0,
                                            max_esg_score=np.inf,
                                            fully_invested=True,
                                            long_only=True,
                                            best_in_class_method=1,
                                            best_in_class_strategy='global',
                                            sector_min_weight_x_dict=None):

        self.get_optimal_portfolio_markowitz(gammas=gammas,
                                risk_free_rate=risk_free_rate,
                                max_esg_score=max_esg_score,
                                fully_invested=fully_invested,
                                long_only=long_only,
                                best_in_class_strategy=best_in_class_strategy,
                                best_in_class_method=best_in_class_method,
                                sector_min_weight_x_dict=sector_min_weight_x_dict)

        df_weigths_ = pd.DataFrame(data=self.ticker_sector_dict.values(), 
                               index=self.ticker_sector_dict.keys(), 
                               columns=['sector'])
        df_weigths_['weights'] = self.weights_tangente_portfolio
        df_weigths_['ticker_colors_indic'] = [i for i in range(len(df_weigths_))]
        dict_sector_colors = {sector: i for i, sector in enumerate(df_weigths_['sector'].unique())}

        # Aggregate weights by ticker and sector
        ticker_weights_ = df_weigths_.loc[df_weigths_["weights"] >= 1.e-10].sort_values(by='sector')['weights']
        sector_weights_ = df_weigths_.loc[df_weigths_["weights"] >= 1.e-10].groupby('sector')['weights'].sum().sort_index()

        # Create a more visually appealing pie chart
        fig, ax = plt.subplots(figsize=(12, 8))

        # Outer pie chart for sectors
        outer_colors = plt.cm.gray(np.linspace(0.4, 0.8, len(df_weigths_['sector'].unique())))
        ax.pie(sector_weights_, 
            labels=sector_weights_.index, 
            startangle=90, 
            colors=[outer_colors[dict_sector_colors[sector]] for sector in sector_weights_.index],
            autopct='%1.0f%%', 
            pctdistance=0.85,
            radius=1,
            wedgeprops=dict(width=0.3, edgecolor='w'))  # Add a hole to the pie chart

        # Inner pie chart for tickers
        inner_colors = self.colors[df_weigths_.loc[df_weigths_["weights"] >= 1.e-10].sort_values(by='sector')['ticker_colors_indic']]
        ax.pie(ticker_weights_, 
            labels=ticker_weights_.index, 
            startangle=90, 
            colors=inner_colors, 
            autopct='%1.0f%%', 
            pctdistance=0.7,
            radius=0.7,
            labeldistance=0.8,
            textprops={'fontsize':8},
            wedgeprops=dict(width=0.3, edgecolor='w'))  # Add a hole to the pie chart

        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set(aspect="equal")

        # Set title
        plt.title('Composition of the tangente portfolio by tickers and sectors')

        plt.show()              


    def plot_efficient_frontier_multiple_max_esg_scores(self,
                                                        gammas,
                                                        risk_free_rate=0,
                                                        max_esg_scores=[np.inf],
                                                        fully_invested=True,
                                                        long_only=True,
                                                        best_in_class_method=1,
                                                        best_in_class_strategy='global',
                                                        with_optimal_portfolio=False,
                                                        with_linear_tangent=False,
                                                        opt_problem='Markowitz',
                                                        sector_min_weight_x_dict=None):
        
        # if ~self.get_efficient_frontier_data_multiple_max_esg_scores_called:
        self.get_efficient_frontier_data_multiple_max_esg_scores(gammas=gammas,
                                                                    risk_free_rate=risk_free_rate,
                                                                    max_esg_scores=max_esg_scores,
                                                                    fully_invested=fully_invested,
                                                                    long_only=long_only,
                                                                    best_in_class_method=best_in_class_method,
                                                                    best_in_class_strategy=best_in_class_strategy,
                                                                    opt_problem=opt_problem,
                                                                    sector_min_weight_x_dict=sector_min_weight_x_dict)

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
                                best_in_class_strategy='global',
                                opt_problem='Markowitz',
                                sector_min_weight_x_dict=None):

        _sharpe_ratio_tangent = []

        for max_esg_score in max_esg_scores:
            if opt_problem=='Markowitz':
                self.get_optimal_portfolio_markowitz(gammas=gammas,
                                        risk_free_rate=risk_free_rate,
                                        max_esg_score=max_esg_score,
                                        fully_invested=fully_invested,
                                        long_only=long_only,
                                        best_in_class_method=best_in_class_method,
                                        best_in_class_strategy=best_in_class_strategy,
                                        sector_min_weight_x_dict=sector_min_weight_x_dict)
                
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
                                best_in_class_strategy='global',
                                opt_problem='Markowitz',
                                sector_min_weight_x_dict=None):
        
        # if not self.get_efficient_frontier_data_multiple_max_esg_scores_called:
        self.get_efficient_frontier_data_multiple_max_esg_scores(gammas=gammas,
                                                                risk_free_rate=risk_free_rate,
                                                                max_esg_scores=max_esg_scores,
                                                                fully_invested=fully_invested,
                                                                long_only=long_only,
                                                                best_in_class_method=best_in_class_method,
                                                                best_in_class_strategy=best_in_class_strategy,
                                                                opt_problem=opt_problem,
                                                                sector_min_weight_x_dict=sector_min_weight_x_dict)

        self.weights_evolution_with_max_score = [self.multiple_esg_simulations[max_score]['weights_tangente_portfolio'] for max_score in max_esg_scores]

        self.dict_weights_evolution = {ticker: np.array(self.weights_evolution_with_max_score)[:, i] for i, ticker in enumerate(self.ticker_sector_dict)}

        width = 0.7

        fig, ax = plt.subplots(figsize=(15, 5))
        bottom = np.zeros(len(max_esg_scores))

        for i, (boolean, weight) in enumerate(self.dict_weights_evolution.items()):
            p = ax.bar(np.arange(len(max_esg_scores)), weight, width, label=boolean, bottom=bottom, color=self.colors[i])
            bottom += weight

        ax.set_title("Evolution of the weights of the stocks in the portfolio with the maximum ESG score constraint")
        ax.set_xticks(np.arange(len(max_esg_scores)))  
        ax.set_xticklabels([f'{score:.2f}' for score in max_esg_scores])  # Round to 2 decimal places
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.xticks(rotation=45)
        plt.xlabel('Max ESG Score Constraint')
        plt.ylabel('Weights')

        plt.show()

    # def plot_combined_graphs(self,
    #                         gammas,
    #                         risk_free_rate=0,
    #                         max_esg_scores=[np.inf],
    #                         fully_invested=True,
    #                         long_only=True,
    #                         best_in_class_method=1,
    #                         best_in_class_strategy='global',
    #                         opt_problem='Markowitz',
    #                         sector_min_weight_x_dict=None):
        
    #     # Plotting Sharpe Ratio vs Max ESG Score
    #     _sharpe_ratio_tangent = []
    #     for max_esg_score in max_esg_scores:
    #         if opt_problem == 'Markowitz':
    #             self.get_optimal_portfolio_markowitz(gammas=gammas,
    #                                     risk_free_rate=risk_free_rate,
    #                                     max_esg_score=max_esg_score,
    #                                     fully_invested=fully_invested,
    #                                     long_only=long_only,
    #                                     best_in_class_method=best_in_class_method,
    #                                     best_in_class_strategy=best_in_class_strategy,
    #                                     sector_min_weight_x_dict=sector_min_weight_x_dict)
                
    #         elif opt_problem == 'Pedersen':
    #             self.get_optimal_portfolio_Pedersen(gammas,
    #                                     risk_free_rate,
    #                                     max_esg_score,
    #                                     fully_invested,
    #                                     long_only,
    #                                     best_in_class_method)
                
    #         _sharpe_ratio_tangent.append(self.sharpe_ratio_tagent)
        
    #     # Plotting Sharpe Ratio vs Max ESG Score
    #     plt.figure(figsize=(15, 6))
    #     plt.subplot(1, 2, 1)
    #     plt.plot(max_esg_scores, _sharpe_ratio_tangent, marker='o', linestyle='-', color='b')
    #     plt.title('Sharpe Ratio of Tangent Portfolio vs Max ESG Score Constraint')
    #     plt.xlabel('Max ESG Score Constraint')
    #     plt.ylabel('Sharpe Ratio')
    #     plt.grid(True)

    #     # Plotting Weights Evolution
    #     plt.subplot(1, 2, 2)
    #     self.get_efficient_frontier_data_multiple_max_esg_scores(gammas=gammas,
    #                                                             risk_free_rate=risk_free_rate,
    #                                                             max_esg_scores=max_esg_scores,
    #                                                             fully_invested=fully_invested,
    #                                                             long_only=long_only,
    #                                                             best_in_class_method=best_in_class_method,
    #                                                             best_in_class_strategy=best_in_class_strategy,
    #                                                             opt_problem=opt_problem,
    #                                                             sector_min_weight_x_dict=sector_min_weight_x_dict)

    #     self.weights_evolution_with_max_score = [self.multiple_esg_simulations[max_score]['weights_tangente_portfolio'] for max_score in max_esg_scores]

    #     self.dict_weights_evolution = {ticker: np.array(self.weights_evolution_with_max_score)[:, i] for i, ticker in enumerate(self.ticker_sector_dict)}

    #     width = 0.7

    #     bottom = np.zeros(len(max_esg_scores))
    #     for i, (boolean, weight) in enumerate(self.dict_weights_evolution.items()):
    #         p = plt.bar(np.arange(len(max_esg_scores)), weight, width, label=boolean, bottom=bottom)
    #         bottom += weight

    #     plt.title("Evolution of the weights of the stocks in the portfolio with the maximum ESG score constraint")
    #     plt.xticks(np.arange(len(max_esg_scores)), [f'{score:.2f}' for score in max_esg_scores], rotation=45)
    #     plt.xlabel('Max ESG Score Constraint')
    #     plt.ylabel('Weights')
    #     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
    #     plt.tight_layout()
    #     plt.show()


    def plot_sharpe_ratio_and_weights_varying_esg_limit(self,
                                                        gammas,
                                                        risk_free_rate=0,
                                                        max_esg_scores=[np.inf],
                                                        fully_invested=True,
                                                        long_only=True,
                                                        best_in_class_method=1,
                                                        best_in_class_strategy='global',
                                                        opt_problem='Markowitz',
                                                        sector_min_weight_x_dict=None):
        
        # Plotting Sharpe Ratio vs Max ESG Score
        _sharpe_ratio_tangent = []
        for max_esg_score in max_esg_scores:
            if opt_problem == 'Markowitz':
                self.get_optimal_portfolio_markowitz(gammas=gammas,
                                        risk_free_rate=risk_free_rate,
                                        max_esg_score=max_esg_score,
                                        fully_invested=fully_invested,
                                        long_only=long_only,
                                        best_in_class_method=best_in_class_method,
                                        best_in_class_strategy=best_in_class_strategy,
                                        sector_min_weight_x_dict=sector_min_weight_x_dict)
                
            elif opt_problem == 'Pedersen':
                self.get_optimal_portfolio_Pedersen(gammas,
                                        risk_free_rate,
                                        max_esg_score,
                                        fully_invested,
                                        long_only,
                                        best_in_class_method)
                
            _sharpe_ratio_tangent.append(self.sharpe_ratio_tagent)
        
        # Plotting Sharpe Ratio vs Max ESG Score
        plt.figure(figsize=(10, 12))

        plt.subplot(2, 1, 1)
        plt.plot(max_esg_scores, _sharpe_ratio_tangent, marker='o', linestyle='-', color='b')
        plt.title('Sharpe Ratio of Tangent Portfolio vs Max ESG Score Constraint')
        plt.xlabel('Max ESG Score Constraint')
        plt.ylabel('Sharpe Ratio')
        plt.grid(True)

        # Plotting Weights Evolution
        plt.subplot(2, 1, 2)
        self.get_efficient_frontier_data_multiple_max_esg_scores(gammas=gammas,
                                                                risk_free_rate=risk_free_rate,
                                                                max_esg_scores=max_esg_scores,
                                                                fully_invested=fully_invested,
                                                                long_only=long_only,
                                                                best_in_class_method=best_in_class_method,
                                                                best_in_class_strategy=best_in_class_strategy,
                                                                opt_problem=opt_problem,
                                                                sector_min_weight_x_dict=sector_min_weight_x_dict)

        self.weights_evolution_with_max_score = [self.multiple_esg_simulations[max_score]['weights_tangente_portfolio'] for max_score in max_esg_scores]

        self.dict_weights_evolution = {ticker: np.array(self.weights_evolution_with_max_score)[:, i] for i, ticker in enumerate(self.ticker_sector_dict)}

        width = 0.7

        bottom = np.zeros(len(max_esg_scores))
        for i, (boolean, weight) in enumerate(self.dict_weights_evolution.items()):
            p = plt.bar(np.arange(len(max_esg_scores)), weight, width, label=boolean, bottom=bottom,  color=self.colors[i])
            bottom += weight

        plt.title("Evolution of the weights of the stocks in the portfolio with the maximum ESG score constraint")
        plt.xticks(np.arange(len(max_esg_scores)), [f'{score:.2f}' for score in max_esg_scores], rotation=45)
        plt.xlabel('Max ESG Score Constraint')
        plt.ylabel('Weights')
        # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
        
        plt.tight_layout()
        plt.show()
