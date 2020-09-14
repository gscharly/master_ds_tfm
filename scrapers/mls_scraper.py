import requests
from bs4 import BeautifulSoup
from scrapers.constants import HDR, CHROMEDRIVER, MLS_SEASONS
import re
from selenium import webdriver
import os
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, ElementNotVisibleException, NoSuchElementException
import time
import random
from time import sleep
from scrapers.utils.date_utils import create_weekly_year_calendar
from typing import List
import json


os.environ["webdriver.chrome.driver"] = CHROMEDRIVER
URL = 'https://matchcenter.mlssoccer.com'
SCROLL_PAUSE_TIME = 0.5
MIN_WAIT = 0.1
MAX_WAIT = 2.5
REFRESH_PAUSE_TIME = 3


def click_initial_button(driver):
    while True:
        try:
            survey = WebDriverWait(driver, 3).until(EC.presence_of_element_located( (By.XPATH, '//*[@id="app"]/div/div[2]/div[2]/div[1]/div[2]/div[2]/div')))
            survey.click()
        except ElementNotVisibleException:
            # print("Button is not visible")
            break
        except TimeoutException:
            # print("No button is detected")
            break


def scroll_bottom(driver: webdriver):
    # NO FUNCIONA EN ESTA WEB :)
    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        try:
            # print(last_height)
            # Scroll down to bottom
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            # Wait to load page
            time.sleep(SCROLL_PAUSE_TIME)
            # Calculate new scroll height and compare with last scroll height
            new_height = driver.execute_script("return document.body.scrollHeight")
            # print(new_height)
            if new_height == last_height:
                break
            last_height = new_height
        except WebDriverException:
            break


def unique_match_url(list_dates: List[str], driver) -> List[str]:
    unique_urls = set()
    for day in list_dates:
        day_url = URL + '/schedule/' + day
        driver.get(day_url)
        click_initial_button(driver)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        # page = requests.get(day_url, headers=HDR)
        # soup = BeautifulSoup(page.text, features="html.parser")
        a_list = soup.findAll("a", href=re.compile("(/matchcenter/)(\d{4}-\d{2}-\d{2}-.*)(/feed)"))
        links = [a['href'] for a in a_list]
        unique_urls.update(links)
        # print(unique_urls)
        # print(len(unique_urls))
    return list(unique_urls)


def article_text(match_url):
    url_article = URL + match_url.replace('feed', 'recap')
    print(url_article)
    page = requests.get(url_article, headers=HDR)
    soup = BeautifulSoup(page.text, features="html.parser")
    try:
        p_list = soup.find('article', {'class': 'article'}).findAll('p')
        p_text = ' '.join([p.get_text() for p in p_list])
        # print(p_text)
        return p_text
    except:
        return list()


def event_text(match_url, driver):
    url_events = URL + match_url
    # print(url_events)
    driver.get(url_events)
    click_initial_button(driver)
    sleep(random.uniform(MIN_WAIT, MAX_WAIT))
    scroll_bottom(driver)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    try:
        div_events = soup.find('ul', {'class': 'feed'}).findAll('div', {'class': ['comment-description', 'goal-title',
                                                                                  'goal-text']})
        goal_text = None
        events_text = list()
        for div in div_events:
            # Goal text should be before goal-title
            if 'goal-text' in div.get('class'):
                goal_text = div.get_text()
            elif 'goal-title' in div.get('class') and goal_text:
                events_text.append(div.get_text() + '. ' + goal_text)
            else:
                events_text.append(div.get_text())

        # print(events_text)
        return events_text[::-1]
    except:
        return list()


def load_current_json(path):
    if not os.path.exists(path):
        print('{} does not exists. Returning empty dict'.format(path))
        return dict()
    else:
        with open(path) as json_file:
            data = json.load(json_file)
        return data


def save_update_json(path, data_dict, complete_url):
    # print(path)
    if not os.path.exists(path):
        print('{} does not exists. Creating file'.format(path))
        with open(path, 'w') as json_file:
            data = {complete_url: data_dict}
            json.dump(data, json_file)
    else:
        # print('Updating {} with {}'.format(path, data_dict))
        # Read content
        with open(path, 'r') as outfile:
            data = json.load(outfile)
        # Rewrite whole json
        with open(path, 'w') as outfile:
            data[complete_url] = data_dict
            json.dump(data, outfile)


def check_dates_urls(urls: List[str], year: int):
    """Keep only urls for the year"""
    ini_date = '{}-02-01'.format(year)
    end_date = '{}-12-31'.format(year)
    filter_urls = list(filter(lambda url: ini_date <= url.split('/')[-2][0:10] <= end_date, urls))
    print('Looking for matches between {} and {}'.format(ini_date, end_date))
    # filter_urls = list(filter(lambda url: url.split('/')[-2][0:10] == str(year), urls))
    return filter_urls


if __name__ == '__main__':
    for year_code, year in MLS_SEASONS.items():
        driver = webdriver.Chrome(CHROMEDRIVER)
        json_path = '../data/json/mls_{}_goals.json'.format(year_code)
        season_dict = dict()
        print(year)
        dates = create_weekly_year_calendar(year)
        # Temporadas suelen ir desde febrero-marzo hasta diciembre
        ini_date = '{}-02-01'.format(year)
        end_date = '{}-12-31'.format(year)
        fil_dates = list(filter(lambda x: ini_date <= x <= end_date, dates))
        # print(dates)
        match_urls = unique_match_url(fil_dates, driver)
        # print(match_urls)
        print("Original:", len(match_urls))
        fil_match_urls = check_dates_urls(match_urls, year)
        print("Filtered for year", len(fil_match_urls))
        # Load info and ignore parsed urls
        current_json = load_current_json(json_path)
        print('Current json has {} matches'.format(len(current_json.keys())))
        # Restart driver
        driver = webdriver.Chrome(CHROMEDRIVER)
        for match_url in fil_match_urls:
            complete_url = URL + match_url
            if complete_url in current_json.keys():
                print('{} already in json'.format(complete_url))
                continue
            text = article_text(match_url)
            events = event_text(match_url, driver)
            if len(text) > 0 and len(events) > 0:
                match_dict = {
                        'article': text,
                        'events': events
                }
                save_update_json(json_path, match_dict, complete_url)
            else:
                print('There is no article or events for', complete_url)
            # season_dict[URL + match_url] = match_dict
        print('Partidos:', len(season_dict.keys()))
