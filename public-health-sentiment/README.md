## Public Health Sentiment Analysis

### Overview

During the COVID‑19 pandemic, public perception of vaccines has influenced uptake and ultimately impacted community health outcomes.  This project analyses social media chatter to understand how people feel about vaccination.  Tweets mentioning keywords such as “COVID vaccine” or brand names are scraped in real time using the [`snscrape`](https://github.com/JustAnotherArchivist/snscrape) library.  After cleaning and preprocessing the text, sentiment is scored using the VADER lexicon to gauge whether messages are positive, neutral or negative.  Temporal and topical trends are then explored through visualisations and dashboards.

### Problem Statement

Public health campaigns often rely on social media to disseminate information and combat misinformation.  Understanding the prevailing sentiment towards vaccines helps health agencies tailor messaging, identify misinformation hotspots and allocate resources.  The objective is to measure sentiment over time and across platforms and to surface common themes in vaccine discussions.

### Data Source

Tweets are collected via the `snscrape` tool, which allows retrieval of public Twitter posts without requiring API credentials.  For this analysis, approximately 1 000 recent tweets containing phrases like “COVID vaccine” are scraped.  Each record includes metadata such as timestamp, text, hashtags, retweet count and whether the post is a retweet.  Example fields from a similar vaccination dataset include tweet ID, user name, user location, text and hashtags【43699772525297†L0-L18】.  Because tweets are user‑generated, the data may contain misspellings, slang or sarcasm that complicates sentiment analysis.

### Tools Used

- **Python** with `pandas` for data collection and manipulation.
- **snscrape** for scraping Twitter without API keys.
- **NLTK** and **VADER** sentiment lexicon for scoring sentiment.
- **Matplotlib**, **Seaborn** and **WordCloud** for visualisation.
- Optional: **Tableau Public** or **Plotly Dash** for interactive dashboards.

### Business Value

Health authorities and communicators can use sentiment insights to adjust outreach strategies.  Positive sentiment spikes may correspond to successful campaigns, whereas surges in negative sentiment could indicate viral misinformation requiring intervention.  Tracking hashtags also reveals which vaccine brands or policies are being discussed, helping teams tailor responses.

### Approach and Key Findings

1. **Data Collection** –  A Python script uses `snscrape` to retrieve the latest 1 000 tweets containing the term “COVID vaccine”.  Duplicate retweets are removed to avoid overweighting popular messages.
2. **Text Cleaning** –  URLs, mentions, hashtags and punctuation are stripped from the tweet text.  Tokens are converted to lowercase and stop words are removed.  Emojis and non‑ASCII characters are handled appropriately.
3. **Sentiment Scoring** –  The VADER lexicon assigns a compound sentiment score to each tweet.  Scores range from –1 (most negative) to +1 (most positive).  Tweets are categorised as positive, neutral or negative based on threshold values.
4. **Temporal Analysis** –  Sentiment scores are grouped by date to observe trends over time.  A line chart illustrates how overall sentiment evolved during the sampled period.
5. **Word Cloud and Hashtag Analysis** –  The most frequent words and hashtags are visualised to highlight key topics.  This helps identify which vaccine brands or public figures attract attention.
6. **Platform Comparison** –  If tweets from multiple sources (e.g. Twitter for iPhone vs. Android) are available, sentiment differences by platform are explored.

### Visualisations

- A **line chart** showing average sentiment scores per day
- A **word cloud** of the most frequent words in the tweets
- A **bar chart** of hashtag counts

These visuals reveal periods of optimism and concern and spotlight the language people use when discussing vaccines.  A simple dashboard could combine these elements for stakeholders to monitor in real time.
