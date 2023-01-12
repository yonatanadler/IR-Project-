# IR-Project

This repository presents a search engine project on the entire Wikipedia corpus for information retrival course at BGU.

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
