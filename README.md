# IR-Project

This repo presents a search engine project on the entire Wikipedia corpus for information retrival course at BGU.

## Content

- IndexerGCP - Folder contains 9 .ipynb files for our indexes builders, pre-made calculations for better running time and one python file that builds the Inverted Index structure:

  - Indexes builders:

    - title_index - creates the titles extraction for each doc from the entire corpus.
    - stem_title_index - creates the titles extraction with stemming for each doc from the entire corpus.
    - body_index - creates the text extraction for each doc from the entire corpus.
    - stem_body_index - creates the text extraction with stemming for each doc from the entire corpus.
    - anchor_index - creates the anchor text extraction for each doc from the entire corpus.

  - Pre-made calculations:

    - page_rank - makes dictionary of page rank for each document from the entire corpus.
    - page_view - makes dictionary of page view for each document from the entire corpus.

  - Inverted Index:

    - inverted_index_gcp - builds the Inverted Index structure for our indexes.

## Retrieval Methods

In this project we support five different ranking methods called from search_frontend file:

- Search -
- Search_body - returns up to a 100 search results for the query using tf-idf and cosine aimilarity of the body of articles only.
- Search_title - binary ranking, returns all search results that contain a query word in the title of articles.
- search_anchor - binary ranking, returns all search results that contain a query word in the anchor text of articles.
- get_pagerank - Returns PageRank values for a list of provided wiki article IDs.
- get_pageview - Returns the number of page views that each of the provide wiki articles.

## Evaluation

We evaluated our engine using MAP@40. Our engine average score is ... and the average retrival time is ...
