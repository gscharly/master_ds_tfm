from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException,StaleElementReferenceException
from bs4 import BeautifulSoup
import re
import os

"""
Han cambiado la página, y aún no se ven los marcadores :)
"""



ONE_F_URL = "https://onefootball.com/en/competition/premier-league-9/matches"
N_MATCHES = 38

chromedriver = "/home/carlos/selenium/chromedriver"
os.environ["webdriver.chrome.driver"] = chromedriver
driver = webdriver.Chrome(chromedriver)
driver.get(ONE_F_URL)

driver.implicitly_wait(10)

# for match in range(31, 36):
#     print(match)
#     xpath = "/html/body/of-app/main/of-competition-nav/div/div/of-competition-tab-matches/div/section/of-competition-matches/div[1]/of-group-navigation/div/nav/ul/li[{}]/a".format(match)
#     match_click = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, xpath)))
#     match_click.click()


# Navigation starts at the right
left_arrow_xpath = "/html/body/of-app/main/of-competition-nav/div/div/of-competition-tab-matches/div/section/of-competition-matches/div[1]/of-group-navigation/div/nav/span[1]"
for n_match in range(N_MATCHES+1, 1):
    match_day = "/html/body/of-app/main/of-competition-nav/div/div/of-competition-tab-matches/div/section/of-competition-matches/div[1]/of-group-navigation/div/nav/ul/li[{}]/a".format(n_match)


    left_click = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, left_arrow_xpath)))
    left_click.click()

# for n_match in range(1, N_MATCHES+1):
#     xpath = "/html/body/of-app/main/of-competition-nav/div/div/of-competition-tab-matches/div/section/of-competition-matches/div[1]/of-group-navigation/div/nav/ul/li[{}]/a".format(n_match)
#     match_click = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, xpath)))
#     match_click.click()


# element = driver.find_element_by_link_text('Live Text')
# element.click()
# while True:
#     try:
#         showmore = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="tab-2"]/div/div/div[1]/div[2]/button')))
#         showmore.click()
#     except TimeoutException:
#         break
#     except StaleElementReferenceException:
#         break

# soup = BeautifulSoup(driver.page_source, "html.parser")
# p_tags = soup.find("div", {"id": "story-body"}).findAll("p")
# news_text = [p.get_text() for p in p_tags]
# print(news_text)

