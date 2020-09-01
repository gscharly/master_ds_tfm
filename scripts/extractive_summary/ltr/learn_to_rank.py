from typing import Dict, List, Optional

from scripts.extractive_summary.ltr.ltr_target_metrics import SummaryMetrics


class LearnToRank:

    def __init__(self, target_metric: str = 'rouge', drop_teams: bool = False):
        """

        :param target_metric: metric that will be used to build the target. One of AVAILABLE_METRICS
        :param drop_teams: whether to include teams in the tokens
        """

        self.target_metric = target_metric
        self.metrics = SummaryMetrics(metric=target_metric, drop_teams=drop_teams)

    def create_match_targets(self, match_dict: Dict, verbose: bool, league_season_team: Optional[str] = None,
                             **metrics_params):
        """
        Calculates the target for a match. Specific metric params can be passed.
        :param match_dict:
        :param verbose:
        :param league_season_team:
        :param metrics_params:
        :return:
        """

        if self.target_metric == 'rouge':
            assert all(k in self.metrics.ROUGE_PARAMS for k in metrics_params.keys()),\
                'Rouge params are {}'.format(self.metrics.ROUGE_PARAMS)
            event_article_list = self.metrics.rouge(match_dict, verbose, **metrics_params)
        elif self.target_metric == 'cosine_tfidf':
            self.metrics.key_events.league_season_teams = league_season_team
            event_article_list = self.metrics.cosine_distance(match_dict, verbose, **metrics_params)
        elif self.target_metric == 'wmd':
            self.metrics.key_events.league_season_teams = league_season_team
            event_article_list = self.metrics.wmd(match_dict, verbose, **metrics_params)
        else:
            raise ValueError('Metric {} is not available. Try one of {}'.format(self.target_metric,
                                                                                self.metrics.AVAILABLE_METRICS))
        return event_article_list

    def print_scores_info(self, match_dict: Dict, event_article_list: List[Dict], reverse=True):
        article_sentences = self.metrics.text_proc.get_sentences(match_dict['article'])
        article_sentences_text = [str(sent).replace('\n', '') for sent in article_sentences]
        scores = sorted([(el['score'], el['event_ix'], el['sentence_ix']) for el in event_article_list],
                        reverse=reverse)
        for info in scores:
            print('Score:', info[0])
            print('Event:', match_dict['events'][info[1]])
            print('Nearest article sentence:', article_sentences_text[info[2]])
            print()
