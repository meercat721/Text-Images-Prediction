#!/usr/bin/env bash

pip install -r requirements.txt

python -m spacy download de_core_news_lg
python -m spacy download de_core_news_sm
