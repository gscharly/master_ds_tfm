from gensim.summarization.summarizer import summarize

from scripts.text.article_text_processor import ArticleTextProcessor

processor = ArticleTextProcessor()

all_files = processor.load_json()

prueba = all_files['premier_league_2019_2020.json']['https://www.bbc.com/sport/football/49791610']

events_prueba = ' '.join(prueba['events'])

print('Text')
print(events_prueba)
print('Summary')
print(summarize(events_prueba))

