"""
Public Health Sentiment Analysis
================================

This script collects tweets about COVID‑19 vaccination, cleans the text,
computes sentiment scores using the VADER lexicon and produces a few
simple visualisations.  It is designed as a reproducible workflow: data
collection, preprocessing, scoring and plotting are encapsulated in
functions so that you can rerun the analysis with different queries or
time ranges.  The focus is on clarity and explanatory comments rather
than over‑optimised code.

Key steps:

1. **Scrape tweets** containing user‑defined keywords using the
   `snscrape` command‑line tool.  This avoids the need for Twitter API
   credentials.  The script collects the latest *n* tweets matching
   the query and stores them in a DataFrame.
2. **Clean text** by removing URLs, mentions, hashtags and
   punctuation.  Convert to lowercase and strip whitespace.  This
   simplifies sentiment analysis.
3. **Score sentiment** using the VADER sentiment analyser from
   NLTK.  Each tweet receives a compound score between –1 and +1.
4. **Aggregate sentiment over time** to see how public mood evolves.
   A line chart of average daily sentiment is saved to the `images`
   directory.
5. **Visualise word frequencies** via a word cloud and bar chart.

Before running this script, ensure that `snscrape`, `pandas`, `nltk`,
`matplotlib`, `seaborn` and `wordcloud` are installed.  You may also
need to download the VADER lexicon by running

```python
import nltk
nltk.download('vader_lexicon')
```
"""

import os
import re
import subprocess
from datetime import datetime
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud


def scrape_tweets(query: str, max_tweets: int = 1000) -> pd.DataFrame:
    """Scrape tweets containing a search query using snscrape.

    Parameters
    ----------
    query : str
        Keyword or search phrase (e.g. "COVID vaccine").
    max_tweets : int, default=1000
        Maximum number of tweets to retrieve.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, text, username, source, hashtags.
    """
    # Use snscrape to collect tweets; output as JSON lines for easier parsing
    cmd = [
        "snscrape", "--jsonl", "--max-results", str(max_tweets),
        f"twitter-search:{query}"
    ]
    print(f"Running snscrape for query '{query}'…")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    tweets = []
    for line in result.stdout.splitlines():
        data = pd.read_json(line, typ="series")
        tweets.append({
            "date": pd.to_datetime(data["date"]).date(),
            "text": data["content"],
            "username": data["user"]["username"],
            "source": data.get("sourceLabel", ""),
            "hashtags": data.get("hashtags", [])
        })
    df = pd.DataFrame(tweets)
    return df


def clean_text(text: str) -> str:
    """Remove URLs, mentions, hashtags and non‑alphabetic characters."""
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)  # remove mentions
    text = re.sub(r"#[A-Za-z0-9_]+", "", text)  # remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # keep only letters and spaces
    return text.lower().strip()


def score_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Compute VADER sentiment scores for each tweet."""
    sia = SentimentIntensityAnalyzer()
    df["clean_text"] = df["text"].apply(clean_text)
    df["compound"] = df["clean_text"].apply(lambda x: sia.polarity_scores(x)["compound"])
    return df


def plot_sentiment_over_time(df: pd.DataFrame) -> None:
    """Plot average sentiment per day and save to images."""
    sentiment_by_date = df.groupby("date")["compound"].mean().reset_index()
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=sentiment_by_date, x="date", y="compound")
    plt.title("Average Sentiment Over Time")
    plt.xlabel("Date")
    plt.ylabel("Average VADER Compound Score")
    plt.xticks(rotation=45)
    images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    os.makedirs(images_dir, exist_ok=True)
    path = os.path.join(images_dir, 'sentiment_over_time.png')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved sentiment over time chart to {path}")


def plot_wordcloud(df: pd.DataFrame) -> None:
    """Generate a word cloud of the most frequent words."""
    all_words = " ".join(df["clean_text"])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    os.makedirs(images_dir, exist_ok=True)
    path = os.path.join(images_dir, 'wordcloud.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved word cloud to {path}")


if __name__ == "__main__":
    # Example usage: scrape tweets, compute sentiment and plot
    query = "COVID vaccine"
    tweets_df = scrape_tweets(query, max_tweets=1000)
    tweets_df = score_sentiment(tweets_df)
    plot_sentiment_over_time(tweets_df)
    plot_wordcloud(tweets_df)
