"""Actual Public Health Agency of Canada RSS/Atom descriptive language audit.

Completely independent of the old unverified Twitter scraper. The VADER
scores describe the automated lexical polarity of *agency-written titles
and summaries*, NOT the opinions of the public or vaccine hesitancy.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html import unescape
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import feedparser
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SOURCE = "https://www.canada.ca/content/dam/phac-aspc/rss/new-eng.xml"
CATALOG = "https://www.canada.ca/en/public-health/corporate/stay-informed-stay-connected/public-health-updates.html"


def clean_markup(s):
    txt = re.sub(r"<[^>]*>", " ", str(s))
    txt = unescape(txt)
    return re.sub(r"\s+", " ", txt).strip()


def download_official_feed():
    req = Request(SOURCE, headers={
        "User-Agent": "jibrankazi-phac-source-audit/1.0 (https://github.com/jibrankazi)",
        "Accept": "application/atom+xml,application/rss+xml,application/xml,text/xml",
    })
    with urlopen(req, timeout=35) as res:
        raw = res.read()
    if len(raw) < 400:
        raise ValueError("PHAC actual RSS/Atom response missing or incomplete")
    parsed = feedparser.parse(raw)
    if len(parsed.entries) < 5:
        raise ValueError(f"PHAC feed returned only {len(parsed.entries)} published entries; cannot establish corpus")
    rows = []
    for entry in parsed.entries:
        title = clean_markup(entry.get("title", ""))
        summary = clean_markup(entry.get("summary", entry.get("description", "")))
        link = entry.get("link", "")
        if not title or not link:
            continue
        host = (urlparse(link).hostname or "").lower()
        if not (host == "canada.ca" or host.endswith(".canada.ca")):
            raise ValueError(f"PHAC source entry does not point to official Canada.ca domain: {link}")
        dt = entry.get("published_parsed") or entry.get("updated_parsed")
        if dt is None:
            continue
        published = datetime(*dt[:6], tzinfo=timezone.utc).date().isoformat()
        rows.append({"published_date": published, "title": title,
                     "summary": summary, "official_url": link})
    corpus = pd.DataFrame(rows).drop_duplicates(subset=["official_url"])
    if len(corpus) < 5 or corpus.title.str.len().median() < 10:
        raise ValueError("Insufficient genuine and dated PHAC feed entries")
    if not corpus.published_date.str.match(r"^20\d{2}-\d{2}-\d{2}$").all():
        raise ValueError("Invalid date parsing")
    return corpus, raw


def run(output="public-health-sentiment/results/official_phac_messages"):
    corpus, raw = download_official_feed()
    scorer = SentimentIntensityAnalyzer()
    corpus = corpus.copy()
    corpus["vader_compound_lexical_tone"] = corpus.apply(
        lambda row: scorer.polarity_scores(row["title"] + " " + row["summary"])["compound"],
        axis=1)
    corpus["lexical_category"] = corpus.vader_compound_lexical_tone.map(
        lambda score: "positive_lexical" if score >= .05
        else "negative_lexical" if score <= -.05 else "neutral_lexical"
    )
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    corpus = corpus.sort_values(["published_date", "official_url"])
    corpus.to_csv(out / "original_phac_feed_items_and_lexical_scores.csv", index=False)
    daily = corpus.groupby("published_date", as_index=False)["vader_compound_lexical_tone"].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(pd.to_datetime(daily.published_date), daily.vader_compound_lexical_tone, marker="o")
    ax.axhline(0, linestyle="--", linewidth=.8)
    ax.set(title="PHAC agency-authored updates: automated lexical tone by publication date",
           xlabel="Actual publication date", ylabel="Mean VADER compound score")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "official_phac_published_message_lexical_tone.png", dpi=140)
    plt.close(fig)
    result = {
        "publisher": "Public Health Agency of Canada, Government of Canada",
        "original_feed_url": SOURCE,
        "publisher_documented_feed_catalog": CATALOG,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "original_response_sha256": sha256(raw).hexdigest(),
        "real_unique_dated_publisher_entries": int(len(corpus)),
        "earliest_publication_date": corpus.published_date.min(),
        "latest_publication_date": corpus.published_date.max(),
        "lexical_tone_counts": {str(k): int(v) for k, v in corpus.lexical_category.value_counts().items()},
        "mean_lexical_tone_score": float(corpus.vader_compound_lexical_tone.mean()),
        "method": "VADER lexicon applied solely to original PHAC agency titles and feed summaries",
        "limits": ("No tweets or social-platform sampling. Agency-authored text is not public sentiment "
                   "or public vaccination attitudes. Lexical classifications are not gold labels, "
                   "human annotated, validated misinformation flags, or health outcomes."),
    }
    (out / "verified_results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",default="public-health-sentiment/results/official_phac_messages")
    run(p.parse_args().output)
