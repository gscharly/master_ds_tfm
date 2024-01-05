# Automatic generation of sport news

## Abstract
Natural Language Processing techniques are currently one of the most active investigation fields in the area of Computer Science and Machine Learning.
These techniques can be applied to multiple applications, such as text summarization. The goal of text summarization is to provide a new and shorter text from a reference, keeping the main information.

This work uses these automatic text summarization techniques to create football news. Almost every online sport site provides both real time information and text summaries about
football matches. Therefore, the main goal is the design and implementation of a solution that builds a summary of a match using these real time events.

Regarding the data, as there is no open database that provides the required information, Web scraping techniques are used to extract text from different online newspapers and sites.
Each match is represented by a list of real-time events that describe match information and players, together with a summary that collects the most important events.
Both text sets present different characteristics in terms of vocabulary and structure, and must be processed in different ways.

Three different strategies that use the real-time events as summaries are proposed. The first one leverages the most important events (events that describe goals or red cards) and
uses them to perform a summary. The second strategy builds a conceptual graph, and uses the relation between the different events and concepts to build a summary.
Finally, the last solution treats the problem as a ranking problem, where each event has a score that represents the likelihood of that score being included in the real summary.

These events are evaluated in two different ways. The first one is one of the most used metrics to evaluate summaries (ROUGE), and it is based on counting the common words or
word sequences in both texts. Despite being widely used, this metric comes with a drawback: it can’t capture semantic relationships. To address this issue,
the use of a similarity based on word embeddings is proposed (SMS).

## Code structure
- analysis: notebooks with EDAs, model metrics, experiments...
- scrapers: package with web scrapers
- scripts: main package with most processes
  - data: scrapers information unification
  - experiments: abstract interface for experiments
  - extractive_summary: extractive summary scripts
  - metrics: evaluation metrics
  - models: model training and metrics definition
  - text: basic NLP processing

