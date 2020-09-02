import json
from typing import Dict, List
from scripts.text.basic_text_processor import BasicTextProcessor
from scripts.conf import DATA_PATH


class ArticleTextProcessor:
    def __init__(self):
        self.tp = BasicTextProcessor()
        self.vocabulary_dict = dict()
        self.vocabulary_url_dict = dict()
        self.words = dict()

    @staticmethod
    def load_json() -> Dict:
        with open('{}/json/final/all_files_processed.json'.format(DATA_PATH)) as json_file:
            # print(json_file)
            all_news = json.load(json_file)
        return all_news

    def _clean_tokens(self, text: str) -> List[str]:
        """Filter punctuation, spaces, noisy characters (@, https..) and numbers"""
        doc = self.tp.token_list(text)
        clean_tokens = [token.text.lower() for token in doc if self.tp.token_filter(token) and self.tp.filter_noisy_characters(token) and
                        not self.tp.has_numbers(token)]
        # ents = self.tp.entity_names(text)
        return clean_tokens

    def entity_names(self, text: str) -> List[str]:
        ents = self.tp.entity_names(text)
        return ents

    def entity_names_events(self, list_events: List[str]):
        # Ignore start and end events
        ents_events = [self.tp.entity_names_labels(event) for event in list_events[1:-1]]
        return ents_events

    def vocabulary_type_dict(self, text_type='article') -> Dict:
        """Returns a dict with the vocabulary for each article"""
        all_news_dict = self.load_json()
        all_vocab_dict = dict()
        self.vocabulary_dict[text_type] = set()
        self.words[text_type] = dict()
        for league_name, league_data in all_news_dict.items():
            for url, article_dict in league_data.items():
                text = article_dict[text_type]
                if text_type == 'events':
                    text = ' '.join(text)
                tokens = self._clean_tokens(text)
                vocab = set(tokens)
                # Update global vocabulary
                self.vocabulary_dict[text_type].update(list(vocab))
                #self.words[text_type].extend(tokens)
                for token in tokens:
                    if self.words[text_type].get(token):
                        self.words[text_type][token] += 1
                    else:
                        self.words[text_type][token] = 1
                # Save article vocabulary
                if all_vocab_dict.get(league_name) is None:
                    all_vocab_dict[league_name] = dict()
                all_vocab_dict[league_name][url] = vocab
        self.vocabulary_url_dict[text_type] = all_vocab_dict
        return all_vocab_dict

    def vocabulary_length(self, text_type='article'):
        # TODO guardar vocab
        if self.vocabulary_dict.get(text_type) is None:
            print('Building vocabulary for', text_type)
            _ = self.vocabulary_type_dict(text_type)
        return len(self.vocabulary_dict[text_type])
