## Reddit Opioid-Related word2vec Word Embedding Models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4323343.svg)](https://doi.org/10.5281/zenodo.4323343)

This repository contains 3 word2vec word embedding models that were used to better understand vocabulary around opioid use. 

### Streamlit App

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/rtiinternational/reddit-opiates-w2v/main/app.py)

You can interact with these models search terminology via a streamlit app linked above.

### Data

Data was collected from two subreddits: `r/opiates` and `r/opiatesrecovery`. The model was trained using three types of content from the two subreddits: post titles, post selftexts, and all comments. The table below describes the number of documents per source.

| Content Type   | r/Opiates | r/OpiatesRecovery |
| -------------- | --------- | ----------------- |
| Post Titles    | 146046    | 14503             |
| Post Selftexts | 103970    | 12238             |
| Comments       | 3162152   | 277387            |

### Preprocessing

All content was transformed to lowercase and tokenized with spaCy or scispaCy.

For the bigram model, [gensim's Phrase tool](https://radimrehurek.com/gensim_3.8.3/models/phrases.html) was used for collocation detection, with a min_count of 50 and a threshold of 10 using the PMI scorer. Bigrams are separated in the vocabulary by a double underscore: `__`

### Model Info

Three models were trained with [gensim's Word2Vec](https://radimrehurek.com/gensim_3.8.3/models/word2vec.html) using the same parameter set:

```min_count=5, size=100, window=5, iter=5, sg=1, seed=1234```

| Model Name                 | Vocab Size           |
| -------------------------- | -------------------- |
| Lowercase Unigram          | 85777                |
| Lowercase scispaCy Unigram | 85489                |
| Lowercase scispaCy Bigrams | 90935 (5518 bigrams) |
