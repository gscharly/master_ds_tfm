import requests
from bs4 import BeautifulSoup
import re
from scrapers.utils.date_utils import months_between_dates
from typing import List, Dict
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, ElementNotVisibleException, NoSuchElementException, ElementNotInteractableException
import time
import json
import random
from time import sleep


BBC_URL = "https://www.bbc.com"
# LEAGUES = ["premier-league", "german-bundesliga", "spanish-la-liga", "italian-serie-a", "french-ligue-one", "champions-league"]
LEAGUES = ["german-bundesliga"]
HDR = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2919.83 Safari/537.36'}
CHROMEDRIVER = "/home/carlos/selenium/chromedriver"
os.environ["webdriver.chrome.driver"] = CHROMEDRIVER
MIN_WAIT = 0.1
MAX_WAIT = 5


def get_news_links(month: str) -> List[str]:
    """
    Returns football news links for a given month. It acceses https://www.bbc.com/sport/football/premier-league/scores-fixtures/YYYY-mm
    and looks for https://www.bbc.com/sport/football/id
    :param month:
    :return:
    """
    url = "{}/sport/football/{}/scores-fixtures/{}".format(BBC_URL, league, month)
    print("Accessing", url)
    sleep(random.uniform(MIN_WAIT, MAX_WAIT))
    page = requests.get(url, headers=HDR)
    soup = BeautifulSoup(page.text, features="html.parser")
    a_links = soup.findAll("a", href=re.compile("(/sport/football/)([0-9].*)"))
    news_links = [BBC_URL + a["href"] for a in a_links]
    return list(set(news_links))


def process_news_texts(texts: List[str]) -> Dict:
    # Muy adhoc
    end_match_event = list(filter(lambda txt: txt.startswith("Match ends"), texts))
    if len(end_match_event) != 1:
        raise ValueError("There is 0, or more than 1 event with Match ends")
    ix_match_event = texts.index(end_match_event[0])
    # print(texts[ix_match_event:][::-1])
    text_dict = {
        "article": '\n'.join(texts[:ix_match_event]),
        "events": texts[ix_match_event:][::-1]
    }
    # Check si tienen info
    if len(text_dict['article']) == 0 or len(text_dict['events']) == 0:
        return dict()
    else:
        return text_dict


def get_text_from_link(url: str) -> Dict:
    print("Accesing", url)
    sleep(random.uniform(MIN_WAIT, MAX_WAIT))
    driver.get(url)
    # TODO poner en funcion generica
    # Cookies
    while True:
        try:
            print("Waiting for cookies...")
            cookies = WebDriverWait(driver, 2).until(EC.presence_of_element_located( (By.XPATH, '//*[@id="bbccookies-continue-button"]')))
            print("Cookies detected")
            cookies.click()
            print("Cookies accepted!")
        except ElementNotVisibleException:
            print("Cookies are not visible")
            break
        except TimeoutException:
            print("No cookies are detected")
            break
        except ElementNotInteractableException:
            print("Cookies not interactable")
            break
    # Survey
    while True:
        try:
            survey = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="no"]')))
            print("Survey detected")
            survey.click()
            print("Survey accepted!")
        except ElementNotVisibleException:
            print("Survet is not visible")
            break
        except TimeoutException:
            print("No survey is detected")
            break

    try:
        element = driver.find_element_by_link_text('Live Text')
        print("Click Live Text")
        element.click()
        while True:
            try:
                showmore = WebDriverWait(driver, 2).until(
                    EC.presence_of_element_located((By.XPATH,
                                                    '//*[@id="tab-2"]/div/div/div[1]/div[2]/button')))

                showmore.click()
            except TimeoutException:
                break
            except StaleElementReferenceException:
                break
        sleep(random.uniform(MIN_WAIT, MAX_WAIT))
        soup = BeautifulSoup(driver.page_source, "html.parser")
        p_tags = soup.find("div", {"id": "story-body"}).findAll("p")
        news_text = [p.get_text() for p in p_tags]
        article_event_dict = process_news_texts(news_text)
    except NoSuchElementException:
        print("There is no Live Text. Skipping")
        article_event_dict = dict()
    return article_event_dict


def save_update_json(path, data_dict):
    # print(path)
    if not os.path.exists(path):
        print('{} does not exists. Creating file'.format(path))
        with open(path, 'w') as json_file:
            json.dump(data_dict, json_file)
    else:
        # print('Updating {} with {}'.format(path, data_dict))
        # Read content
        with open(path, 'r') as outfile:
            data = json.load(outfile)
        # Rewrite whole json
        with open(path, 'w') as outfile:
            data.update(data_dict)
            json.dump(data, outfile)


# TODO poner bonito si funciona
if __name__ == "__main__":
    save_path = "bbc_news"
    start_date = "2019-08-01"
    end_date = "2020-06-30"
    months = months_between_dates(start_date, end_date)
    print(months)
    print("Searching between {} and {}".format(months[0], months[-1]))
    # Selenium
    driver = webdriver.Chrome(CHROMEDRIVER)
    ini = time.time()
    for league in LEAGUES:
        json_path = '../data/json/{}_2019_2020_prueba.json'.format(league.replace('-', '_'))
        for m in months:
            print("Searching", m)
            news_links = get_news_links(m)
            print('{} matches'.format(len(news_links)))
            if len(news_links) == 0:
                print("No data for", m)
                continue
            news_text_dict = dict()
            for link in news_links:
                news_text = get_text_from_link(link)
                if len(news_text) > 0:
                    news_text_dict[link] = news_text
            # news_text_dict = {link: get_text_from_link(link) for link in news_links if len(get_text_from_link(link))>0}
            # TODO: NUEVO 14/06 SIN COMPROBAR QUE FUNCIONA
            save_update_json(json_path, news_text_dict)
            # with open('../data/json/{}_{}_{}.json'.format(league, save_path, m), 'w') as outfile:
            #     json.dump(news_text_dict, outfile)
            print("Finished scraping {} in {} seconds".format(m, time.time() - ini))
        print("Finised in {} seconds".format(time.time() - ini))






