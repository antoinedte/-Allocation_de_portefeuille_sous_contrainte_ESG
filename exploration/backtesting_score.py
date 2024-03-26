from datetime import datetime
import bisect
from config import TICKERS
from src.config import TICKERS



def get_first_date_with_all_msci_score_available(msci_score):
    list_of_date = [list(map(lambda x: datetime.strptime(x, '%Y-%m-%d'), list(msci_score[ticker]['esg_score_dict'].keys()))) for ticker in TICKERS]
    return max([min(date) for date in list_of_date])

def retrieve_last_esg_date(msci_score, ticker, fixed_date):
    fixed_date = datetime.strptime(fixed_date, '%Y-%m-%d')

    date_list_str = list(msci_score[ticker]['esg_score_dict'].keys())
    date_objs_datetime = [datetime.strptime(date_str, '%Y-%m-%d') for date_str in date_list_str]

    insertion_point = bisect.bisect_left(date_objs_datetime, fixed_date)
    if insertion_point > 0:
        last_date_before_fixed_date = date_objs_datetime[insertion_point - 1]
        return last_date_before_fixed_date.strftime('%Y-%m-%d')
    else:
        return None

def get_last_esg_scores(msci_score, fixed_date):
    msci_last_esg_score = [msci_score[ticker]['esg_score_dict'][retrieve_last_esg_date(msci_score, ticker, fixed_date)] for ticker in TICKERS]
    return msci_last_esg_score


