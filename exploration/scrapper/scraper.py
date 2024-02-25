import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

from tqdm.auto import tqdm

import json

def download_msci_esg_ratings_htmlfile(tickers):
    # Create a new folder for html files
    os.makedirs("./esg_html", exist_ok=True)

    # URL for MSCI ESG ratings
    msci_url = "https://www.msci.com/our-solutions/esg-investing/esg-ratings/esg-ratings-corporate-search-tool" 

    # Initialize Selenium webdriver
    chrome_options = Options()
    chrome_options.add_argument("--disable-extensions")
    # chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)

    # Initialize dict
    dict_info = {}

    # Scrape ESG ratings for all constituents
    for symbol in tickers:
        print("Crawling ESG Ticker: " + symbol)

        # Initialize lists to store scores and dates
        scores_list = []
        dates_list = []
        industry_list = []

        # Go to the website
        driver.get(msci_url)
        time.sleep(0.5)

        # Attempt to reject cookies (if a "reject all" button is available)
        try:
            cookie_button = driver.find_element(By.ID, "onetrust-reject-all-handler")
            cookie_button.click()
            print("Cookies rejected.")
            time.sleep(1)
        except:
            print("No 'reject all' button found for cookies or it's already rejected.")

        # Find search input field and enter Ticker
        element = driver.find_element(By.ID, "_esgratingsprofile_keywords")
        element.send_keys(symbol)
        element.send_keys(" ")
        time.sleep(2)

        # Select first search result
        element.send_keys(Keys.ARROW_DOWN)
        time.sleep(2)
        
        element.send_keys(Keys.RETURN)
        time.sleep(1)

        # -- ESG scores 
        # Click on the specified button
        button = driver.find_element(By.ID, "esg-transparency-toggle-link")
        button.click()
        time.sleep(0.5)

        # Find and extract scores
        g_elements_scores = driver.find_elements(By.CLASS_NAME, "highcharts-label") # highcharts-data-labels highcharts-series-0 highcharts-line-series
        scores_list.append([g_element.find_element(By.TAG_NAME, "text").get_attribute("textContent") for g_element in g_elements_scores])

        # Find and extract dates
        g_elements_dates = driver.find_elements(By.CLASS_NAME, "highcharts-axis-labels.highcharts-xaxis-labels")
        dates_list.append([text.get_attribute("textContent") for g_element in g_elements_dates for text in g_element.find_elements(By.TAG_NAME, "text")])

        # Find and extract industry information
        industry_element = driver.find_element(By.CLASS_NAME, "esg-rating-paragraph-distr")
        industry_text = industry_element.find_element(By.TAG_NAME, "b").text
        industry_list.append(industry_text)

        # get possible ratings in the industry
        possible_ratings = dates_list[0][4:11]
        # get proportion of peers in the industry
        dict_industry_peers_ticker = {possible_rating : float(proportion_in_industry[:-2])/100 for possible_rating,proportion_in_industry in zip(possible_ratings, scores_list[0][:len(possible_ratings)])}
        # get esg scores for the ticker
        dict_esg_scores_ticker = {date:score for date, score in zip(dates_list[0][-5:], scores_list[0][len(possible_ratings):])}
        # get industry ticker
        industry_ticker = industry_list[0]

        # save info in dict_info
        dict_info[symbol] = {"esg_score_dict": dict_esg_scores_ticker, 
                             "industry_scores_dict": dict_industry_peers_ticker, 
                             "industry": industry_ticker}
        # time.sleep(0.5)

        # -- Controversy scores
        # Click on the specified button for controversies
        controversies_button = driver.find_element(By.ID, "esg-controversies-toggle-link")
        controversies_button.click()
        time.sleep(0.5)

        # Find and extract controversy information
        controversy_elements = driver.find_elements(By.CSS_SELECTOR, "#controversies-table [class^='column-controversy']")
        controversy_info = {}
        for element in controversy_elements:
            class_name = element.get_attribute("class")
            text = element.text
            controversy_info[text] = class_name.split('-')[-1]

        dict_info[symbol]["controversy_info"] = controversy_info

        # # Find and extract controversy information
        # controversy_elements = driver.find_elements(By.CSS_SELECTOR, "#controversies-table .column-controversy, #controversies-table .subcolumn-controversy, #controversies-table .controversy")
        # controversy_info = {}
        # for element in controversy_elements:
        #     class_name = element.get_attribute("class")
        #     text = element.text
        #     controversy_info[class_name] = text

        # dict_info[symbol]["controversy_info"] =  {value: classe.split('-')[-1]  
        #                                           for classe, value in controversy_info.items() 
        #                                           if value in ['Environment', 'Social', 'Governance']}

    # Close webdriver gracefully
    driver.quit()

    return dict_info


def save_dict_to_json(dict, filename):
    with open(filename, 'w') as file:
            json.dump(dict, file)
    print(f"Dictionary saved to {filename}.")


def load_dict_from_json(filename):
    with open(filename, 'r') as file:
         loaded_data = json.load(file)
    return loaded_data


