"""
Generation of players and teams dataset from the articles and events files.
"""

from scripts.text.basic_text_processor import BasicTextProcessor
from scripts.conf import TEAMS, EN_LABELS
from typing import List, Tuple, Dict
from collections import Counter
from operator import add
from functools import reduce
import pandas as pd

MAIN_PATH = "/home/carlos/MasterDS/tfm"
JSON_PATH = '{}/json'.format(MAIN_PATH)

BANNED_CHARS = ['(', 'replaces', 'goal', 'Yellow']


class TeamPlayers:
    def __init__(self):
        self.text_proc = BasicTextProcessor()
        self.players_set = set()
        self.teams_set = set()
        # Team: list players
        self.teams_players_dict = dict()
        # Player: team
        self.players_teams_dict = dict()
        # Url: teams
        self.url_teams = dict()

    def teams_players_sets(self, en_events: List[List[Tuple[str, str]]], teams_file: str):
        """
        Generates sets corresponding to players and teams, and returns a processed list of events,
        adding new tags to entity names: TEAM or PLAYER
        :return:
        """
        league_season_teams = TEAMS[teams_file]
        # Init
        self.players_set = set()
        self.teams_set = set()
        en_events_processed = list()
        for event in en_events:
            new_event_list = self.team_players_event(event, league_season_teams)
            en_events_processed.append(new_event_list)
        return en_events_processed

    def team_players_event(self, event_list: List[Tuple[str, str]], league_season_teams: List[str]) -> List:
        new_event_list = list()
        for tuple_event in event_list:
            # Player conditions
            cond_player = tuple_event[1] in EN_LABELS['PLAYER']
            cond_teams = any([team in tuple_event[0] or tuple_event[0] in team for team in league_season_teams])
            cond_var = 'VAR' not in tuple_event[0]
            cond_goal = 'Goal' not in tuple_event[0]
            cond_banned_chars = any(ch in tuple_event[0] for ch in BANNED_CHARS)
            # Team conditions
            cond_team = tuple_event[1] in EN_LABELS['TEAM']
            cond_any_team = any(tuple_event[0] == team for team in league_season_teams)

            if cond_player and not cond_teams and cond_var and not cond_banned_chars and cond_goal:
                self.players_set.add(tuple_event[0])
                new_event_list.append((tuple_event[0], 'PLAYER'))
            elif cond_team and cond_any_team:
                # Check that correct team is selected
                # selected_teams = [(len(team), team) for team in league_season_teams if tuple_event[0] in team]
                # print(selected_teams)
                # team_to_add = sorted(selected_teams, reverse=True)[0]
                # print(team_to_add)
                self.teams_set.add(tuple_event[0])
                new_event_list.append((tuple_event[0], 'TEAM'))
            else:
                continue
        return new_event_list

    def teams_players_tags(self, en_events: List[List[Tuple[str, str]]]) -> List[List[Tuple[str, str]]]:
        """
        Assigns new tags to entity names: TEAM or PLAYER
        :return:
        """
        en_events_processed = list()
        for event in en_events:
            event_list = list()
            for tuple_event in event:
                if tuple_event[1] in EN_LABELS:
                    if tuple_event[0] in self.teams_set:
                        event_list.append((tuple_event[0], 'TEAM'))
                    elif tuple_event[0] in self.players_set:
                        event_list.append((tuple_event[0], 'PLAYER'))
                    else:
                        continue
                else:
                    continue
                en_events_processed.append(event_list)
        return en_events_processed

    def teams_players_dicts(self, en_events: List[List[Tuple[str, str]]]) -> [Dict, Dict]:
        """
        Build dictionaries for players and teams
        :param en_events:
        :return:
        """
        teams_players_dict = {team: list() for team in self.teams_set}
        players_teams_dict = {player: list() for player in self.players_set}
        for event in en_events:
            # Si hay equipo y jugador
            if len(event) > 1:
                # print(event)
                player_list = list()
                teams_list = list()
                # team = None
                for t_type in event:
                    # Actualiza lista jugadores
                    if t_type[1] == 'PLAYER':
                        if '(' in t_type[0]:
                            print(event)
                        player = t_type[0]
                        player_list.append(player)
                    # Actualiza lista de equipos
                    elif t_type[1] == 'TEAM':
                        team = t_type[0]
                        teams_list.append(team)
                    else:
                        continue
                for team in teams_list:
                    # Actualiza diccionario team: players
                    players_team = teams_players_dict[team]
                    players_team.extend(player_list)
                    teams_players_dict[team] = list(set(players_team))
                    # Actualiza diccionario player: teams
                    for player in player_list:
                        # print(player)
                        teams_player = players_teams_dict[player]
                        teams_player.extend(teams_list)
                        players_teams_dict[player] = teams_player
        return teams_players_dict, players_teams_dict

    @staticmethod
    def process_players_dict(players_teams_dict: Dict) -> Dict:
        """
        Sum up the number of appearances of each team for each player
        :return:
        """
        for player, teams in players_teams_dict.items():
            counter_teams = Counter(teams)
            count_teams = list(counter_teams.most_common())
            players_teams_dict[player] = count_teams
        return players_teams_dict

    def _update_teams(self, tp_dict: Dict):
        for team in self.teams_set:
            if self.teams_players_dict.get(team):
                # Saved players for the team
                team_players_list = set(self.teams_players_dict[team])
                # Current players for the team
                team_players_current_list = set(tp_dict[team])
                team_players_list.update(team_players_current_list)
                self.teams_players_dict[team] = list(team_players_list)
            else:
                self.teams_players_dict[team] = list(set(tp_dict[team]))

    def _update_players(self, pt_dict: Dict):
        for player in self.players_set:
            # Player has already appeared
            if self.players_teams_dict.get(player):
                # print('Player is already in dict...')
                # Saved tuples team-appearences for the player
                player_teams_list = self.players_teams_dict[player]
                # Current info
                player_teams_current_list = pt_dict[player]
                input_lists = [player_teams_list, player_teams_current_list]
                updated_player_teams_list = reduce(add, [Counter(dict(x)) for x in input_lists])
                # print(updated_player_teams_list)
                self.players_teams_dict[player] = updated_player_teams_list
            else:
                # print('Adding new player...')
                self.players_teams_dict[player] = pt_dict[player]

    def update_dicts(self, tp_dict, pt_dict):
        if len(self.players_teams_dict) == 0 and len(self.teams_players_dict) == 0:
            # print('Initializing dicts...')
            self.players_teams_dict = pt_dict
            self.teams_players_dict = tp_dict
        else:
            # print('Updating dicts...')
            self._update_teams(tp_dict)
            self._update_players(pt_dict)

    def final_update_dicts(self):
        new_players_dict = dict()
        new_teams_dict = dict()
        for player, teams in self.players_teams_dict.items():
            # print(player, teams)
            player_team = Counter(teams).most_common(1)[0][0]
            new_players_dict[player] = player_team
            if new_teams_dict.get(player_team):
                team_set = set(new_teams_dict.get(player_team))
                team_set.add(player)
                new_teams_dict[player_team] = team_set
            else:
                new_teams_dict[player_team] = [player]
        self.teams_players_dict = new_teams_dict
        self.players_teams_dict = new_players_dict

    def check_missing_players(self):
        """Check for players that appear in events where no team is specified. These players will appear one time
        in each time, and they will hopefully be correctly classified with more matches
        """
        for player, teams in self.players_teams_dict.items():
            if len(teams) == 0:
                teams_set_list = list(self.teams_set)
                # print(teams_set_list)
                # print(teams_set_list)
                new_teams = [(teams_set_list[0], 1), (teams_set_list[1], 1)]
                self.players_teams_dict[player] = new_teams
                # Also add to teams players dict
                for team in teams_set_list:
                    self.teams_players_dict[team].append(player)

    def check_for_errors(self):
        """Remove process errors: tuples in dictionary keys"""
        keys_to_remove = [key for key, v in self.teams_players_dict.items() if type(key) is tuple or type(v) is list]
        for k in keys_to_remove:
            del self.teams_players_dict[k]
        keys_to_remove = [key for key, v in self.players_teams_dict.items() if type(v) is tuple]
        for k in keys_to_remove:
            del self.players_teams_dict[k]

    def _player_dict_to_pandas(self, season_file: str) -> pd.DataFrame:
        player_list = list()
        team_list = list()
        for player, team in self.players_teams_dict.items():
            player_list.append(player)
            team_list.append(team)
        pd_df = pd.DataFrame({'player': player_list,
                              'team': team_list})
        pd_df['season_file'] = season_file
        return pd_df

    @staticmethod
    def _informed_events_articles(article_events_dict: Dict) -> bool:
        """Returns True only if articles and events are informed"""
        len_article = len(article_events_dict['article'])
        len_events = len(article_events_dict['events'])
        return len_article > 0 and len_events > 0

    def run_file(self, all_files: Dict, season_file: str):
        not_consired_matches = 0
        season_dict = all_files[season_file]
        # Init
        self.teams_players_dict = dict()
        self.players_teams_dict = dict()
        for url, article_events_dict in season_dict.items():
            # print(_)
            if not self._informed_events_articles(article_events_dict):
                print('No article or events for', url)
                continue
            # en_events_text = self.text_proc.entity_names_labels(article_events_dict['events'])
            en_events_text = self.text_proc.entity_names_events(article_events_dict['events'])
            en_events_proc = self.teams_players_sets(en_events_text, season_file.split('.')[0])
            if len(self.teams_set) != 2:
                print(url)
                print(self.teams_set)
                not_consired_matches += 1
                print(article_events_dict['events'])
                print(en_events_text)
                continue
            teams_players_dict, players_teams_dict = self.teams_players_dicts(en_events_proc)
            players_teams_dict_proc = self.process_players_dict(players_teams_dict)
            self.update_dicts(teams_players_dict, players_teams_dict_proc)
            self.check_missing_players()
            self.url_teams[url] = self.teams_set
        self.final_update_dicts()
        self.check_for_errors()
        print('{} not considered matches for {}'.format(not_consired_matches, season_file))

    def run(self, all_files: Dict):
        list_df = list()
        for season_file in all_files.keys():
            print(season_file)
            self.run_file(all_files, season_file)
            # print(self.players_teams_dict)
            # print(self.teams_players_dict)
            pd_df_season = self._player_dict_to_pandas(season_file)
            print(pd_df_season.head())
            list_df.append(pd_df_season)
        pd_all = reduce(lambda df1, df2: pd.concat([df1, df2]), list_df)
        pd_all.to_csv('{}/data/csv/players_teams.csv'.format(MAIN_PATH))
