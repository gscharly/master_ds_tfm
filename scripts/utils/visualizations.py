from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from typing import List, Optional


def plot_wordcloud(words, width=800, height=800):
    wordcloud = WordCloud(width=width, height=height,
                          background_color='white',
                          min_font_size=10, collocations=False,
                          stopwords=set()).generate(words)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()


def multiple_boxplot(df: pd.DataFrame, num_cols: List, ncols: int, x: Optional[str] = None, width: int = 10,
                     height: int = 10, order: Optional[List] = None):
    fig, axs = plt.subplots(round(len(num_cols)/ncols), ncols, figsize=(width, height))
    y = 0
    for col in num_cols:
        i, j = divmod(y, ncols)
        sns.boxplot(x=x, y=col, data=df, ax=axs[i, j], order=order)
        axs[i, j].xaxis.set_tick_params(rotation=90)
        y += 1
    plt.tight_layout()
    plt.show()


