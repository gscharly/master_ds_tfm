import json
from os import listdir
from os.path import isfile, join
from scripts.conf import DATA_PATH
from typing import List

PATH = '{}/json'.format(DATA_PATH)
FINAL_FILE_NAME = "all_files.json"

JSONS_WITH_BUGS = ['premier_league_2018_2019.json', 'premier_league_2016_2017.json', 'premier_league_2017_2018.json']
BANNED_EVENTS_TOKENS = ['Lineups', 'Half begins', 'Half ends', 'Match ends', 'Full-timeMatch ends']


def fix_event(event: str) -> str:
    event = event.replace('Goal!Goal!', 'Goal.')\
                 .replace('!', '! ')\
                 .replace('SubstitutionSubstitution', 'Substitution')
    return event


def process_events(file_name: str, events: List) -> List:
    # Get rid of first and last two events; they have no information
    save_events = [event for event in events if all(token not in event for token in BANNED_EVENTS_TOKENS)]
    # if len(save_events) != len(events) - 6 and len(save_events) != len(events) - 2 and len(save_events) != len(events) - 4:
    #     print(len(events))
    #     print(len(save_events))
    #     print()
    #     print(events)
    #     print()
    #     print(save_events)
    #     print()
    #     print('Something went wrong while removing events')
    # assert len(save_events) == len(events) - 6, "Something went wrong while removing events"
    if file_name in JSONS_WITH_BUGS:
        save_events = [fix_event(event) for event in save_events]
    return save_events


if __name__ == "__main__":
    onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f)) and f != FINAL_FILE_NAME]
    all_news = dict()
    # Read each json and store in dic
    n = 0
    act = 0
    for f in onlyfiles:
        with open('{}/{}'.format(PATH, f)) as json_file:
            data = json.load(json_file)
        # Delete certain events and fix some bugs
        new_data = dict()
        print(f)

        for match_url, match_dict in data.items():
            n += 1
            print(match_url)
            new_events = process_events(f, match_dict['events'])
            if len(new_events) != 0 and len(match_dict['article']) != 0:
                act += 1
                new_data[match_url] = {'article': match_dict['article'],
                                       'events': new_events}

        all_news[f] = new_data
    print('Articles before filtering:', n)
    print('Articles after filtering:', act)

    # Save to json
    with open('{}/final/{}'.format(PATH, FINAL_FILE_NAME), 'w') as outfile:
        json.dump(all_news, outfile)