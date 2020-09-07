from scripts.text.article_text_processor import ArticleTextProcessor
from scripts.text.basic_text_processor import BasicTextProcessor
from scripts.text.teams_players import TeamPlayers

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from typing import List, Tuple, Dict, Optional
import warnings
import unidecode


class KeyEvents:
    """
    Interface used to define common functions for performing summaries with key events using different approaches
    """

    def __init__(self, drop_teams: bool = False, lemma: bool = False, only_players: bool = False):
        """
        :param drop_teams: whether to include teams' names in tokens
        :param lemma: whether to lemmatize words during text processing
        """
        self.processor = ArticleTextProcessor()
        self.text_proc = BasicTextProcessor()
        # Necessary to keep track of chosen events
        self.events_mapping_list = list()
        # Will be updated with the different articles of each season
        self.league_season_teams = None
        # Players involved in a match
        self.team_players = TeamPlayers()
        self.match_players = None

        self.drop_teams = drop_teams
        self.lemma = lemma
        self.only_players = only_players

    def _check_league_season_teams(self, league_season_teams: Optional[str] = None):
        if not self.league_season_teams and league_season_teams:
            print('league_season_teams is empty. Initializing it to', league_season_teams)
            self.league_season_teams = league_season_teams
        if not self.league_season_teams:
            raise ValueError('league_season_teams is empty')

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
            # print(tokens_en)
            processed_sentences.append(' '.join(tokens_en))
        return processed_sentences

    def _match_summary(self, match_dict: Dict, count_vec_kwargs: Dict, **key_events_properties) -> Dict:
        self._check_league_season_teams()

        summary_events = self.process_match_events(match_dict['events'], **key_events_properties)
        processed_article_sentences = self.process_match_article(match_dict['article'])

        if len(summary_events) == 0 or len(processed_article_sentences) == 0:
            warnings.warn('Could not perform tfidf')
            return dict()

        # Train tfidf with article sentences
        vectorizer = CountVectorizer(**count_vec_kwargs)
        X = vectorizer.fit_transform(processed_article_sentences).toarray()
        tfidfconverter = TfidfTransformer()
        X = tfidfconverter.fit_transform(X).toarray()
        # Events
        X_events = vectorizer.transform(summary_events)
        X_events = tfidfconverter.transform(X_events).toarray()
        # Distances
        distances = cosine_similarity(X_events, X)
        sentences_ixs = distances.argmax(axis=1)
        # We need to save article's sentences to build the summary
        article_sents_list = [str(sent) for sent in self.text_proc.get_sentences(match_dict['article'])]
        summary_sents_list = [article_sents_list[ix] for ix in sorted(list(set(sentences_ixs)))]
        article_summary = ''.join(summary_sents_list)
        print('Number of sentences in original article:', len(article_sents_list))
        print('Number of sentences in summary:', len(summary_sents_list))

        return {
            'article_summary': article_summary,
            'sentences_ixs': sentences_ixs,
            'article_sents_list': article_sents_list
        }

    def process_match_events(self, events: List[str], **kwargs) -> List[str]:
        pass

    def match_summary(self, match_dict: Dict, count_vec_kwargs: Dict, save_relations: bool,
                      verbose: bool, **key_events_properties) -> Tuple[str, List, List]:
        pass

    def run(self, **kwargs):
        pass
