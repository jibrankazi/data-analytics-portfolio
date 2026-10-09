# Public Health Agency of Canada — verified publisher-message language analysis

This module now includes a **fully executed, original-source** audit of the [Public Health Agency of Canada's official RSS/Atom updates](https://www.canada.ca/en/public-health/corporate/stay-informed-stay-connected/public-health-updates.html).

**This is not a social-media public opinion survey.** Agency-authored announcements are not representative of vaccination attitudes, vaccine hesitancy, misinformation in social networks or population sentiment. The `VADER` lexicon measures basic *wording polarity*, not correctness or public response.

## Actual end-to-end execution (October 9, 2026)

[Successful actual-source GitHub Actions run](https://github.com/jibrankazi/data-analytics-portfolio/actions/runs/37963620463) fetched the original official published feed (without a social-network API token), parsed dates, titles, summaries and publisher links, removed duplicate links and **excluded a link pointing to a non-publication preview host**, assigned automated lexical scores, generated plots, and uploaded artifact files.

| Provenance/observed result | Actual measured output |
| --- | ---: |
| Unique and dated original official Canada.ca publication URLs | **43** |
| Excluded preview/non-official-domain link | **1** |
| Earliest official entry publication | 2024-10-09 |
| Latest official entry publication | 2026-09-23 |
| Entries with negative VADER lexical polarity | 17 |
| Entries with neutral VADER lexical polarity | 15 |
| Entries with positive VADER lexical polarity | 11 |
| Average automated lexical score | −0.04965 |

These counts characterize **just the feed entries returned on the retrieval date**, not the full history of PHAC publications and not measured public emotion. VADER scores may incorrectly handle scientific or public-health terminology and have **no independently annotated reference labels** here.

## Reproduce

```bash
python -m pip install pandas matplotlib feedparser vaderSentiment
python public-health-sentiment/run_official_phac_rss.py
```

Official original publisher: `https://www.canada.ca/content/dam/phac-aspc/rss/new-eng.xml`.

The output folder `public-health-sentiment/results/official_phac_messages/` contains:
- `original_phac_feed_items_and_lexical_scores.csv`: original published timestamps, headlines, feed summaries and source URLs, plus calculated lexical scores.
- `official_phac_published_message_lexical_tone.png`: generated actual publication-date trend chart.
- `verified_results.json`: SHA-256 fingerprint of actual raw official feed bytes, true sample size, original publication period, extraction time and explicit scope limitations.

The workflow uploads them as the `observed-phac-publisher-communications-language-audit` artifact.

## Unverified original Twitter component

The original `notebooks/public_health_sentiment.py` depends on `snscrape` scraping public Twitter/X posts with no API credentials. **It has not been successfully executed or independently verified on genuine social-media content in this work**, and historical claims about collecting a thousand vaccination tweets or interpreting public opinion are **not supported**.

No missing tweets, social media user profiles, opinions or data labels have been fabricated. A proper public-opinion study would need a legally accessible actual social-media corpus, defensible sampling and human/validated sentiment labels. This government publication lexical-tone audit is an **honestly narrower, independently working analysis**.
