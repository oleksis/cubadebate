#!/usr/bin/env python
# coding: utf-8

# Cell
import base64
from collections import Counter, OrderedDict
import json
import math
import os
import re
import time
from typing import List, Optional, Dict, Tuple, Iterable, Any

import requests
import dask
from dask import bag as db
from dask.delayed import delayed, Delayed
import numpy as np
from PIL import Image
import spacy
from spacy.tokens import Token
from spacy.lang.es.stop_words import STOP_WORDS
from wordcloud import WordCloud

# Annotations
CommentList = List[Optional[Dict[str, Any]]]
Document = str
DocumentList = List[Document]
DocumentNormalizedList = List[DocumentList]
TermFrequency = Dict[str, float]
IDF = Dict[str, float]
TFIDF = Dict[str, float]
TFIDF_List = List[TFIDF]

CUBADEBATE_API = "http://www.cubadebate.cu/wp-json/wp/v2/"
COMMENTS_ENDPOINT = CUBADEBATE_API + "comments/"

session = requests.Session()
ConnectionErrorRequests = requests.exceptions.ConnectionError


@delayed
def get_comments_json(page: int = 1) -> CommentList:
    """Get JSON of comments from endpoint API Wordpress v2

    RETURNS (CommentList): Delayed Response List of comments in JSON format
    to compute for Dask.Bag
    """
    params = {"page": str(page)}
    results = []

    try:
        with session:
            time.sleep(1)
            resp = session.get(COMMENTS_ENDPOINT,
                               params=params)
            if resp.status_code == 200:
                results = resp.json()
                return results
    except ConnectionErrorRequests:
        pass  # Try again! Occurred Connection Error
    return results


def get_comments(pages: int) -> List[str]:
    """Get List of comments per page"""
    comments_delayed = (get_comments_json(page)
                        for page in range(1, pages + 1))

    comments_bag = db.from_delayed(comments_delayed).map(lambda d: d['content']['rendered'])
    # In Windows the Dask.Bag is multiprocessing by default, change to threads
    comments: List[str] = comments_bag.compute(scheduler='threads')
    return comments


_start = time.time()
comments_list = get_comments(pages=1)
_end = time.time()
print(f'{len(comments_list)} Comments downloaded in {(_end - _start):.2f}s')
# comments_list


# Cell
NUM_PAGES = 100

_start = time.time()
corpus = get_comments(pages=NUM_PAGES)
_end = time.time()
print(f'{len(corpus)} Comments downloaded in {(_end - _start):.2f}s')
# corpus


# Cell
# Load model
nlp = spacy.load('es_core_news_sm')


def remplace_accents(text: str) -> str:
    """Remplace spanish accents vocab lower case: Unicode code point literal to str"""
    text = re.sub(r'á', 'a', text, flags=re.I)
    text = re.sub(r'é', 'e', text, flags=re.I)
    text = re.sub(r'í', 'i', text, flags=re.I)
    text = re.sub(r'ó', 'o', text, flags=re.I)
    text = re.sub(r'ú', 'u', text, flags=re.I)
    return text


def get_text(markup: str) -> str:
    """Remove html tags, URLs and spaces using regexp"""
    text = re.sub(r'<.*?>', '', markup)
    url_pattern = r'(http|ftp)s?://(?:[a-zA-Z]|[0-9]|[$-_@.&#+]|[!*\(\),]|\
                   (?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_pattern, '', text)
    text = re.sub(r'\s+', " ", text)
    return text.strip()


@delayed
def clean(doc: str) -> List[str]:
    """Remove grave accents, stopwords, the punctuations and normalize the corpus."""

    def is_token_allowed(token: Token) -> bool:
        """No Stop words, No Punctuations or len token >= 3"""
        if (not token or not token.string.strip() or
                token.is_stop or token.is_punct or len(token) < 3):
            return False
        return True

    def preprocess_token(token: Token) -> str:
        """Remove grave accents and return lemmatized token lower case"""
        result = remplace_accents(token.lemma_.strip().lower())
        return result

    text = get_text(doc)
    text = text.lower()
    tokens = [preprocess_token(token) for token in nlp(text)
              if is_token_allowed(token)]
    normalized = " ".join(word for word in tokens)
    return normalized.split()


def get_documents_delayed(documents: DocumentList) -> List[Delayed]:
    """Return List[Delayed] clean document"""
    results = []  # type: List[dask.delayed.Delayed]

    for document in documents:
        results.append(clean(document))

    return results


def get_documents_normalized(documents: DocumentList) -> DocumentNormalizedList:
    """Return DocumentNormalizedList clean document"""
    return list(*dask.compute(get_documents_delayed(documents)))


# Process corpus using Dask.delayed
_start = time.time()
documents_normalized = get_documents_normalized(corpus)
_end = time.time()
print(f'{len(documents_normalized)} Documents normalized in {(_end - _start):.2f}s')


# Cell:
# TF-IDF
def comment_tf(document: DocumentList) -> TermFrequency:
    """Term Frequency for a document"""
    bow = Counter(document)
    tf_dict = dict()

    for k, v in bow.most_common():
        tf_dict[k] = v / len(document)

    return tf_dict


def comments_tf(documents: DocumentNormalizedList) -> List[TermFrequency]:
    """Term Frequency for many document"""
    return [comment_tf(comment)
            for comment in documents]


def count_dict(documents: DocumentNormalizedList) -> Dict[str, int]:
    """Counter the word in all documents at least ones"""
    counts = dict()
    for document in documents:
        uniq_words = set(document)
        for word in uniq_words:
            _value = counts.get(word, 0)
            counts[word] = _value + 1
    return counts


def idf_dict(documents: DocumentNormalizedList) -> IDF:
    """Inverse Document Frequency in all Documents"""
    idf_d = dict()

    counts = count_dict(documents)

    for word in counts:
        idf_d[word] = math.log(len(documents) / counts[word])

    return idf_d


def comments_tfidf(documents: DocumentNormalizedList) -> TFIDF_List:
    tfidf_comments = []

    idf_comments = idf_dict(documents)

    def compute_tfidf_comment(document: DocumentList) -> TFIDF:
        tfidf_comment = dict()

        tf_comment = comment_tf(document)

        for word in set(document):
            tfidf_comment[word] = tf_comment[word] * idf_comments[word]

        return tfidf_comment

    for comment in documents:
        tfidf_comments.append(compute_tfidf_comment(comment))

    return tfidf_comments


tfidf_list = comments_tfidf(documents_normalized)
# tfidf_list
print('TF-IDF comments list')


# Cell
unordered_tfidf: Dict[str, float] = dict()

for tfidf in tfidf_list:
    for w, value in tfidf.items():
        if w in unordered_tfidf:
            unordered_tfidf[w] += value
        else:
            unordered_tfidf[w] = value


def sort_tfidf(tfidf_unordered) -> Iterable[Tuple[Any, Any]]:
    return sorted(tfidf_unordered.items(), key=lambda x: x[1], reverse=True)


# unordered_tfidf
ordered_comments_tfidf = OrderedDict(sort_tfidf(unordered_tfidf))
# Save to JSON file
with open('comments_tfidf.json', 'w') as file_json:
    json.dump(ordered_comments_tfidf, file_json)
print('TF-IDF ordered saved to comments_tfidf.json')


# Cell
# Capitolio Habana
IMG_CAPITOLIO = os.getenv("IMG_CAPITOLIO") or "capitolio.jpg"

# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()

IMG_CAPITOLIO: str = os.path.join(_dir, IMG_CAPITOLIO)
IMG_CAPITOLIO_LINK: str = 'https://upload.wikimedia.org/wikipedia/commons/8/8f/Capitolio_full.jpg'

# download mask images
# !wget http://media.cubadebate.cu/wp-content/gallery/la-habana-nocturna/app_la-habana_05.jpg -O la_hababa.jpg
if not os.path.isfile(IMG_CAPITOLIO):
    response = requests.get(IMG_CAPITOLIO_LINK)

    if response.status_code == 200:
        with open(IMG_CAPITOLIO, 'wb') as _capitolio:
            _capitolio.write(response.content)
            print('Image downloaded.')
    else:
        print('Image No downloaded!')
else:
    image_downloaded: str = IMG_CAPITOLIO.split("\\")[-1].split("/")[-1]
    print(f"Image: {image_downloaded}")

# Cell
# WordsCloud Cubadebate
IMG_WORDCLOUD = 'wordcloud_cubadebate.png'

# read the mask image
_mask = np.array(Image.open(IMG_CAPITOLIO))

# Generate Word Cloud
wordcloud = WordCloud(
    max_words=1000,
    #     max_font_size=50,
    height=1440,
    width=2160,
    background_color='white',
    mask=_mask,
    contour_width=1,
    contour_color='steelblue',
    stopwords=STOP_WORDS).generate_from_frequencies(ordered_comments_tfidf)
# Save to file
wordcloud.to_file(IMG_WORDCLOUD)
print('WordCloud Cubadebate image saved.')

# Cell
with open(IMG_WORDCLOUD, 'rb') as file_img:
    data = base64.b64encode(file_img.read()).decode('utf-8')

img_str = '''
<img width="100%" height="100%" 
src="data:image/png;base64,{}" />
'''.format(data)

with open('wordcloud_cubadebate.html', 'w') as _html:
    _html.write(img_str)

print('Exported image to html.')
