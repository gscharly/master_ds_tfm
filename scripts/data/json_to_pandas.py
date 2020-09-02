"""
Transforms json unified articles and events into a csv file
"""

from scripts.conf import DATA_PATH, LITERAL_TO_LEAGUE
import json
from typing import Dict
import pandas as pd


def read_json(path: str) -> Dict:
    with open('{}/json/final/all_files_processed.json'.format(path)) as json_file:
        all_news = json.load(json_file)
    return all_news


def json_to_pandas(news_dict: Dict) -> pd.DataFrame:
    tuple_list = list()
    for json_file, value_dict in news_dict.items():
        for url, article_info in value_dict.items():
            t = (json_file, url, article_info['article'], article_info['events'])
            tuple_list.append(t)
    pd_info = pd.DataFrame(tuple_list,
                           columns=['json_file', 'url', 'article', 'events'])
    return pd_info


def match_file_league(file_name, leagues):
    for league in leagues:
        if league in file_name:
            return LITERAL_TO_LEAGUE[league]
    return None


def process_df(pd_df: pd.DataFrame) -> pd.DataFrame:
    leagues = LITERAL_TO_LEAGUE.keys()
    # League name
    pd_df['league'] = pd_df['json_file'].map(lambda file_name: match_file_league(file_name, leagues))
    # Season name
    pd_df['season'] = pd_df['json_file'].map(lambda file_name: '_'.join(file_name.split('_')[-2:]).split('.')[0])
    # First two events are ignored
    # pd_df['events'] = pd_df['events'].map(lambda events_list: events_list[2:])
    # Filter empty articles or events
    pd_df_fil = pd_df[(pd_df['article'] != "") & (len(pd_df['events']) != 0)]
    return pd_df_fil


if __name__ == '__main__':
    news_dict = read_json(DATA_PATH)
    print(sorted(list(news_dict.keys())))
    pd_news = json_to_pandas(news_dict)
    print(pd_news.head())
    print('Articles before filtering:', len(pd_news))
    pd_proc = process_df(pd_news)
    print('Articles after filtering:', len(pd_proc))
    # Save csv
    pd_proc.to_csv('{}/csv/articles_events_processed.csv'.format(DATA_PATH), index=False)