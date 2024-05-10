import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import copy
import os
import sys
# Assuming 'src' directory is located one level above the notebook directory
YOUR_FOLDER_DIRECTORY = os.getcwd()
project_directory = os.path.join(YOUR_FOLDER_DIRECTORY, '..')
sys.path.append(project_directory)

from src.backtesting_score import (

    retrieve_last_esg_date
)

# Min Method
def get_new_esg_score_min_method(esg_score, controverse_score):
    potential_new_esg_score = np.random.normal(esg_score, controverse_score)
    new_esg_score = np.max(np.min([esg_score, potential_new_esg_score]),0)
    return new_esg_score

# Probabilistic Method
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# def proba_controverse_effective(x, convexity="convex", lambda_=1):
#     if convexity=="concave":
#         proba = 1/(1 + np.exp(-lambda_*(x-1))) # -1 pour commencer à 1/2
#         proba -= 1/2
#     elif convexity=="convex":
#         proba = 1/(1 + np.exp(-lambda_*(x-4))) # -4 pour finir à 1/2
#     else:
#         raise ValueError("Enter a correct convexity label, be it convex or concave!")
#     return proba

def proba_controverse_effective(c, proba_pour_1, proba_pour_4):
    a_plus_b = np.log(proba_pour_1)
    quatre_a_plus_b = np.log(proba_pour_4)
    a = (quatre_a_plus_b - a_plus_b) / 3
    b = a_plus_b - a
    return np.exp(a * c + b)

def get_new_esg_score_proba_method(esg_score, controverse_score, proba_pour_1=0.1, proba_pour_4=0.5):

    controverse_might_occurs = np.random.binomial(1, proba_controverse_effective(controverse_score, proba_pour_1=proba_pour_1, proba_pour_4=proba_pour_4))
    if controverse_might_occurs == 1:
        effect_on_new_esg_score = -np.abs(np.random.normal(0, controverse_score))
        new_esg_score = esg_score + effect_on_new_esg_score
    else:
        new_esg_score = esg_score
    return new_esg_score

# boostrap method effect
def boostrap_method_effect(esg_score=30, method_function = get_new_esg_score_min_method, nb_boostrap=1000):
    # Define the range of controverse_score values
    controverse_scores = np.linspace(1, 4, 9)

    # Initialize a list to store the results
    results = {}

    # Bootstrap the code for each controverse_score value
    for controverse_score in controverse_scores:
        res = []
        for i in range(nb_boostrap):
            res.append(method_function(esg_score=esg_score, controverse_score=controverse_score))
        
        results[controverse_score] = res
    df=pd.DataFrame.from_dict(results)
    sns.boxplot(data=df)
    plt.xticks(rotation=90)
    plt.show()

def compute_mean_over_controverse(msci_score, ticker, dict_color_to_variance):
    return np.mean([dict_color_to_variance[color] for color in msci_score[ticker]['controversy_info'].values()])

def compute_new_esg_score_with_method(msci_score, end_date, dict_color_to_variance, method=get_new_esg_score_min_method):
    esg_score_test = [msci_score[ticker]['esg_score_dict'][retrieve_last_esg_date(msci_score, ticker, end_date)] for ticker in list(msci_score.keys())]
    controverse_score_test = [compute_mean_over_controverse(msci_score=msci_score, 
                                                            ticker=ticker, 
                                                            dict_color_to_variance=dict_color_to_variance) for ticker in list(msci_score.keys())]
    df_ = pd.DataFrame({'ticker':list(msci_score.keys()), 'esg_score': esg_score_test, 'controverse_score': controverse_score_test})
    df_['esg_score_new'] = [method(esg_score=df_.loc[index, 'esg_score'], controverse_score=df_.loc[index, 'controverse_score']) for index in range(len(df_))]
    return df_

def change_msci_score_with_controverse(msci_score, end_date, dict_color_to_variance,  method=get_new_esg_score_min_method):
    msci_score_ = copy.deepcopy(msci_score)

    df_ = compute_new_esg_score_with_method(msci_score=msci_score, 
                                            end_date=end_date, 
                                            dict_color_to_variance=dict_color_to_variance, 
                                            method=method)
    
    for index in range(len(df_)):
        date = retrieve_last_esg_date(msci_score, df_.loc[index, 'ticker'], end_date) 
        msci_score_[df_.loc[index, 'ticker']]['esg_score_dict'][date] = df_.loc[index, 'esg_score_new']
    
    return msci_score_
    
