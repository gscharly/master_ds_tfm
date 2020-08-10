import os
from scrapers.constants import ESPN_LEAGUES_DICT, ESPN_SEASONS, HDR, CHROMEDRIVER
from typing import List
import requests
from bs4 import BeautifulSoup
import re
import random
from time import sleep
import json
import time
from selenium import webdriver
from scrapers.utils.date_utils import create_year_calendar

URL = "https://www.espn.com"
FIXTURES_URL = "/soccer/fixtures/_/date"
os.environ["webdriver.chrome.driver"] = CHROMEDRIVER
MIN_WAIT = 0.1
MAX_WAIT = 5


def filter_match_days(day_list: List[str], year: int) -> List[str]:
    # Seasons start in August
    min_start_date = '{}0801'.format(year)
    day_list_fil = list(filter(lambda x: x > min_start_date, day_list))
    return day_list_fil


def url_matchs_day(league: str, day_date: str):
    url = '{}/{}/league/{}.1'.format(URL + FIXTURES_URL, day_date, league)
    page = requests.get(url, headers=HDR)
    soup = BeautifulSoup(page.text, features="html.parser")
    try:
        a_list = soup.find('div', {"class": "schedule__card"}).findAll("a", href=re.compile("(/soccer/report\?gameId=)([0-9].*)"))
        # print(a_list)
        if len(a_list) == 0:
            return list()
        else:
            # print('Accessing', url)
            news_links = [URL + a["href"] for a in a_list]
            return news_links
    except:
        return list()


def process_text(text_list: List[str]):
    return ' '.join(text_list)


def get_match_info(url: str, driver: webdriver):
    # print("Accesing", url)

    driver.get(url)
    sleep(random.uniform(MIN_WAIT, MAX_WAIT))
    # page = requests.get(url, headers=HDR)
    # soup = BeautifulSoup(page.text, features="html.parser")
    soup = BeautifulSoup(driver.page_source, "html.parser")
    # print(soup)
    try:
        p_list = soup.find('div', {'class': 'article-body'}).findAll("p")
        # print(p_list)
        p_text = [p.get_text() for p in p_list]
        print(p_text)
        return p_text
    except AttributeError:
        return list()


def get_events_info(url: str):
    game_id = url.split('=')[-1]
    events_url = '{}/soccer/commentary?gameId={}'.format(URL, game_id)
    # print('Accessing', events_url)
    page = requests.get(events_url, headers=HDR)
    soup = BeautifulSoup(page.text, features="html.parser")
    try:
        table = soup.find('div', {'id': 'match-commentary-1-tab-1'}).findAll("td", {"class": "game-details"})
        # print(table)
        t_text = [t.get_text() for t in table]
        # print(t_text)
        proc_text = [text.replace('\n', '') for text in t_text]
        proc_text = [text.replace('\t', '') for text in proc_text]
        proc_text = [text.strip() for text in proc_text]
        end_match_event = list(filter(lambda txt: txt.startswith("Match ends"), proc_text))
        # print(end_match_event)
        ix_match_event = proc_text.index(end_match_event[0])
        reversed_text = proc_text[ix_match_event:][::-1]
        print(reversed_text)
        return reversed_text
    except AttributeError:
        return list()


if __name__ == "__main__":
    article_event_dict = dict()
    driver = webdriver.Chrome(CHROMEDRIVER)
    for league, league_code in ESPN_LEAGUES_DICT.items():
        print('League', league)
        for year_code, year in ESPN_SEASONS.items():
            ini = time.time()
            print('Year', year)
            dates = create_year_calendar(year)
            fil_dates = filter_match_days(dates, year)
            unique_urls = set()
            for match_day in fil_dates:
                urls = url_matchs_day(league_code, match_day)
                if len(urls) > 0:
                    unique_urls.update(urls)
            unique_urls_list = list(unique_urls)
            article_event_dict_season = dict()
            for match_url in unique_urls_list:
                article = get_match_info(match_url, driver)
                events = get_events_info(match_url)
                if len(article) > 0 and len(events) > 0:
                    article_text = process_text(article)
                    article_event_dict_season[match_url] = {
                        'article': article_text,
                        'events': events
                    }
            print('Partidos:', len(article_event_dict_season.keys()))
            with open('../data/json/{}_{}.json'.format(league, year_code), 'w') as outfile:
                json.dump(article_event_dict_season, outfile)
            print("Finished scraping league {} {} in {} seconds".format(league,year_code,
                                                              time.time() - ini))
            # print(unique_urls_list)
            # print(len(unique_urls_list))
    # get_match_info('https://www.espn.com/soccer/report?gameId=521893', driver)





