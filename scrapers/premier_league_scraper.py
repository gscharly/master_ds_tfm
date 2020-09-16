from scrapers.constants import CHROMEDRIVER
from bs4 import BeautifulSoup
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, ElementNotVisibleException, NoSuchElementException, WebDriverException
from typing import List, Tuple
import time
import json

SEASON_URL_DICT = {
    # '2016_2017': '54',
    # '2017_2018': '79',
    # '2018_2019': '210',
    '2019_2020': '274'
}

SCROLL_PAUSE_TIME = 0.5
REFRESH_PAUSE_TIME = 3
URL = "https://www.premierleague.com"


os.environ["webdriver.chrome.driver"] = CHROMEDRIVER


def get_matches_urls(url: str, driver: webdriver) -> List[str]:
    print('Accessing', url)
    driver.get(url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH,
                                                                    '//*[@id="mainContent"]/div[2]/div[1]/div[3]/section/div[1]/ul')))
    time.sleep(REFRESH_PAUSE_TIME)
    scroll_bottom(driver)
    # TODO Hay que meter un scroll up (bien hecho para 16-17 y 17-19)
    time.sleep(10)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    div_links = soup.findAll("div", {"class": "fixture postMatch"})
    news_links = ['http://' + div["data-href"][2:] for div in div_links]
    return news_links


def get_articles_and_events(url: str, driver: webdriver) -> Tuple[List[str], List[str]]:
    print('Accessing', url)
    driver.get(url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH,
                                                                    '//*[@id="mainContent"]/div/section/div[2]/div[2]/div[2]/section[1]/div/div[1]/div[3]/section/div/div/div[1]/div/div')))
    scroll_bottom(driver)
    time.sleep(REFRESH_PAUSE_TIME)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    # print(soup)
    p_article = soup.find("div", {"class": "standardArticle"}).findAll("p")
    # El ultimo es un link
    news_text = [p.get_text() for p in p_article[:-1]]
    # print(news_text)
    comments = soup.find("ul", {"class": "commentaryContainer"}).findAll("div", {"class": "innerContent"})
    comments_text = [div.get_text() for div in comments][::-1]
    # print(comments_text)
    return news_text, comments_text


def scroll_bottom(driver: webdriver):
    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        try:
            # Scroll down to bottom
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait to load page
            time.sleep(SCROLL_PAUSE_TIME)

            # Calculate new scroll height and compare with last scroll height
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        except WebDriverException:
            break


if __name__ == "__main__":
    driver = webdriver.Chrome(CHROMEDRIVER)
    for season, season_code in SEASON_URL_DICT.items():
        season_dict = dict()
        print(season)
        url_season = "{}/results?co=1&se={}&cl=-1".format(URL, season_code)
        matches_urls = get_matches_urls(url_season, driver)
        print(len(matches_urls))
        print(matches_urls[-1])
        match_articles_events = dict()
        for match_url in matches_urls:
            articles, events = get_articles_and_events(match_url, driver)
            match_articles_events[match_url] = {"article": '\n'.join(articles), "events": events}
        with open('../data/json/premier_league_{}.json'.format(season), 'w') as outfile:
            json.dump(match_articles_events, outfile)



