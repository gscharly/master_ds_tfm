# Library imports
from scripts.text.basic_text_processor import BasicTextProcessor
from scripts.text.teams_players import TeamPlayers
from scripts.conf import DATA_PATH

# Other imports
import os
import json
from typing import Dict, List
import unidecode
from collections import Counter
from tqdm import tqdm
import pickle


class ArticleTextProcessor:
    SAVE_PATH = '{}/pickle'.format(DATA_PATH)

    def __init__(self, drop_teams: bool = False, lemma: bool = False, only_players: bool = False):
        self.text_proc = BasicTextProcessor()
        # Team players
        self.team_players = TeamPlayers()
        self.league_season_teams = None
        self.match_players = None
        # Text processing options
        self.lemma = lemma
        self.drop_teams = drop_teams
        self.only_players = only_players
        # Vocab dicts
        self.vocabulary_dict = dict()
        self.vocabulary_url_dict = dict()
        self.words = dict()

    @staticmethod
    def load_json() -> Dict:
        with open('{}/json/final/all_files.json'.format(DATA_PATH)) as json_file:
            # print(json_file)
            all_news = json.load(json_file)
        return all_news

    def _basic_clean_tokens(self, text: str) -> List[str]:
        doc = self.text_proc.token_list(text)
        return [token.text.lower() for token in doc if self.text_proc.token_filter(token)
                and self.text_proc.filter_noisy_characters(token)
                and not self.text_proc.has_numbers(token)]

    def _clean_tokens(self, doc) -> List[str]:
        if self.lemma:
            return [token.lemma_.lower() for token in doc if self.text_proc.token_filter(token)
                    and self.text_proc.filter_noisy_characters(token)
                    and not self.text_proc.has_numbers(token) and self.text_proc.token_filter_stopword(token)]
        else:
            return [token.text.lower() for token in doc if self.text_proc.token_filter(token)
                    and self.text_proc.filter_noisy_characters(token)
                    and not self.text_proc.has_numbers(token) and self.text_proc.token_filter_stopword(token)]

    def process_match_text(self, text: str, text_type: str = 'event') -> List:
        """
        Cleans tokens and retrieve entity names for a given text
        :param text:
        :param text_type: event or article
        :return:
        """
        if text_type == 'event':
            doc = self.text_proc.token_list(text)
        elif text_type == 'article':
            doc = text
        else:
            raise ValueError("text_type available values are event and article")

        # Token cleaning
        clean_tokens = self._clean_tokens(doc)
        # TODO Add result changes tokens
        # result_list = self.text_proc.get_numbers(doc)

        # Entity names
        en_text = self.text_proc.entity_names_labels(text)
        en_proc = self.team_players.team_players_event(en_text, self.league_season_teams)
        en_list = [event_proc[0] for event_proc in en_proc]
        # print(en_list)
        # Save players
        if not self.match_players:
            self.match_players = list()
        self.match_players.extend([en for en in en_list if en in self.team_players.players_set])
        # Filter repeated words
        # filtered_tokens = [t for t in clean_tokens if not any([t in en.lower() for en in en_list])]
        # Join
        # tokens_en = filtered_tokens + en_list
        # Strip accents
        tokens_en = [unidecode.unidecode(t) for t in clean_tokens]
        # TODO: this should be done with EN, but sometimes it fails to identify teams
        if self.drop_teams:
            teams = [team.lower() for team in self.league_season_teams]
            # Drop any token that is included in a team's name
            tokens_en = [tk for tk in tokens_en if not any(tk in team for team in teams)]

        if self.only_players:
            # Keep EN and delete teams
            tokens_en = [unidecode.unidecode(t).lower() for t in en_list]
            teams = [team.lower() for team in self.league_season_teams]
            # Drop any token that is included in a team's name
            tokens_en = [tk for tk in tokens_en if not any(tk in team for team in teams)]

        return tokens_en

    def process_match_article(self, article: str) -> List[str]:
        doc_sents = self.text_proc.get_sentences(article)
        processed_sentences = list()
        for sentence in doc_sents:
            tokens_en = self.process_match_text(sentence, text_type='article')
            if len(tokens_en) == 0:
                continue
            # print(tokens_en)
            processed_sentences.append(' '.join(tokens_en))
        return processed_sentences

    def _get_picke_paths(self, text_type: str = 'article', process: bool = False):
        if process and self.drop_teams and self.lemma:
            append = '_processed_lemma_teams_'
        elif process and self.drop_teams:
            append = '_processed_teams_'
        elif process and self.lemma:
            append = '_processed_lemma_'
        elif process:
            append = '_processed_'
        else:
            append = '_'
        return {'path': '{}/{}{}vocab.pickle'.format(self.SAVE_PATH, text_type, append),
                'path_files': '{}/{}{}vocab_files.pickle'.format(self.SAVE_PATH, text_type, append)}

    def _build_vocab_file(self, league_data: Dict, text_type: str = 'article', process: bool = False) -> Counter:
        """
        Computes the vocabulary for a single file and returns it as a Counter. A file contains multiple matches.
        :param league_data:
        :return:
        """
        vocab_file = Counter()
        for url, article_dict in league_data.items():
            text = article_dict[text_type]
            if text_type == 'events':
                text = ' '.join(text)
            if process:
                doc = self.text_proc.token_list(text)
                tokens = self._clean_tokens(doc)
            else:
                tokens = self._basic_clean_tokens(text)
            vocab = list(set(tokens))
            vocab_file.update(vocab)
        return vocab_file

    def build_vocab(self, text_type: str = 'article', process: bool = False, save: bool = False):
        """Returns a dict with the vocabulary for each article"""
        all_news_dict = self.load_json()
        self.vocabulary_dict[text_type] = dict()
        self.words[text_type] = Counter()
        vocab_files_dict = dict()
        for league_name, league_data in tqdm(all_news_dict.items()):
            vocab_file_counter = self._build_vocab_file(league_data, text_type, process)
            vocab_files_dict[league_name] = vocab_file_counter.most_common()
            self.words[text_type].update(vocab_file_counter)
        self.vocabulary_dict[text_type] = self.words[text_type].most_common()
        if save:
            path_dict = self._get_picke_paths(text_type, process)
            with open(path_dict['path'], 'wb') as fp:
                pickle.dump(self.vocabulary_dict[text_type], fp)
            with open(path_dict['path_files'], 'wb') as fp:
                pickle.dump(vocab_files_dict, fp)

    def get_vocabulary(self, text_type: str = 'article', process: bool = False) -> List:
        path_dict = self._get_picke_paths(text_type, process)
        if not os.path.exists(path_dict['path']):
            print('{} does not exists'.format(path_dict['path']))
            print('Building vocabulary for', text_type)
            self.build_vocab(text_type, process, save=True)

        with open(path_dict['path'], 'rb') as fp:
            vocab = pickle.load(fp)
        return vocab

    def get_vocabulary_file(self, text_type: str, league_name: str, process: bool):
        path_dict = self._get_picke_paths(text_type, process)
        with open(path_dict['path_files'], 'rb') as fp:
            vocab = pickle.load(fp)
        return vocab[league_name]
