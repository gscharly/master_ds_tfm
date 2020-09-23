from scripts.text.article_text_processor import ArticleTextProcessor
from scripts.text.basic_text_processor import BasicTextProcessor

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from typing import List, Tuple, Dict, Optional
import warnings


class KeyEvents:
    """
    Interface used to define common functions for performing summaries with key events using different approaches
    """

    def __init__(self, drop_teams: bool = False, lemma: bool = False, only_players: bool = False):
        """
        :param drop_teams: whether to include teams' names in tokens
        :param lemma: whether to lemmatize words during text processing
        """
        self.processor = ArticleTextProcessor(drop_teams=drop_teams, lemma=lemma, only_players=only_players)
        self.text_proc = BasicTextProcessor()
        # Necessary to keep track of chosen events
        self.events_mapping_list = list()

    def _check_league_season_teams(self, league_season_teams: Optional[str] = None):
        if not self.processor.league_season_teams and league_season_teams:
            print('league_season_teams is empty. Initializing it to', league_season_teams)
            self.processor.league_season_teams = league_season_teams
        if not self.processor.league_season_teams:
            raise ValueError('league_season_teams is empty')

    def _match_summary(self, match_dict: Dict, count_vec_kwargs: Dict, **key_events_properties) -> Dict:
        self._check_league_season_teams()

        summary_events = self.process_match_events(match_dict['events'], **key_events_properties)
        processed_article_sentences = self.processor.process_match_article(match_dict['article'])

        if len(summary_events) == 0 or len(processed_article_sentences) == 0:
            warnings.warn('Could not perform tfidf')
            return dict()

        # Train tfidf with article sentences
        vectorizer = CountVectorizer(**count_vec_kwargs)
        try:
            x = vectorizer.fit_transform(processed_article_sentences).toarray()
            tfidfconverter = TfidfTransformer()
            x = tfidfconverter.fit_transform(x).toarray()
        except ValueError:
            warnings.warn('Could not perform tfidf')
            return dict()

        # Events
        x_events = vectorizer.transform(summary_events)
        x_events = tfidfconverter.transform(x_events).toarray()
        # Distances
        distances = cosine_similarity(x_events, x)
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
