#!/usr/bin/env python
# coding: utf-8

# Cell
import base64
from collections import Counter, OrderedDict
from datetime import datetime
import json
from json import JSONDecodeError
import math
import os
import re
from string import Template
import time
from typing import List, Optional, Dict, Tuple, Iterable, Any

import requests
import dask
from dask import bag as db
from dask import dataframe as dd
from dask.delayed import delayed
import numpy as np
from pandas import DataFrame
from PIL import Image
import spacy
from spacy.tokens import Token
from spacy.lang.es.stop_words import STOP_WORDS
from wordcloud import WordCloud

# Annotations
ElementList = List[Optional[Dict[str, Any]]]
Document = str
DocumentList = List[Document]
DocumentNormalizedList = List[DocumentList]
TermFrequency = Dict[str, float]
IDF = Dict[str, float]
TFIDF = Dict[str, float]
TFIDF_List = List[TFIDF]

CUBADEBATE_API = "http://www.cubadebate.cu/wp-json/wp/v2/"
COMMENTS_ENDPOINT = CUBADEBATE_API + "comments/"
SEARCH_ENDPOINT = CUBADEBATE_API + "search/"

session = requests.Session()
ConnectionErrorRequests = requests.exceptions.ConnectionError

FECHA_FORMAT = "%Y-%m-%dT%H:%M:%S"


def create_datetime(string: str) -> Optional[datetime]:
    """Create datetime from string '2014-07-03T23:27:51'"""
    date_time = None

    try:
        if string is not None:
            date_time = datetime.strptime(string, FECHA_FORMAT)
    except ValueError:
        print("Format no valid!")

    return date_time


def save_elements_json(name: str, line: str = None, mode="w"):
    """Save elements to file, by lines JSON elments."""
    with open(name, mode) as _file:
        if line:
            _file.write(line + "\n")


def read_elements_json(name: str) -> Optional[list]:
    """Read by lines JSON data in file name.
    Return list of elements or None"""
    data = []
    try:
        with open(name, "r") as _file:
            for line in _file.readlines():
                if line:
                    data += json.loads(line)
    except JSONDecodeError:
        print(f"Error reading lines of _file: {name}")
    except FileNotFoundError as f_error:
        print(f_error)

    return data


def get_comments_file(name: str) -> DataFrame:
    """Get List of comments from file name"""
    comments_json = read_elements_json(name)
    comments_bag = db.from_sequence(comments_json).map(
        lambda d: {
            "id": d["id"],
            "post": d["post"],
            "author_name": d["author_name"],
            "date": create_datetime(d["date"]),
            "content": d["content"]["rendered"],
            "link": d["link"],
        }
    )
    # In Windows the Dask.Bag is multiprocessing by default, change to threads
    with dask.config.set(scheduler="threads"):
        comments: List[str] = comments_bag.compute()
    df_comments = DataFrame(comments)
    return df_comments


@delayed
def get_elements_json(url: str, **kwargs) -> ElementList:
    """Get JSON of list elements from endpoint API Wordpress v2

    RETURNS (ElementList): Delayed Response List of elements in JSON format
    to compute for Dask.Bag
    """
    params = {"page": "1"}
    params.update(kwargs)
    results = []

    try:
        with session:
            time.sleep(1)
            resp = session.get(url, params=params)
            if resp.status_code == 200:
                results = resp.json()
                if params.get("file"):
                    save_elements_json(params.get("file"), resp.text, "a")
                return results
    except ConnectionErrorRequests:
        pass  # Try again! Occurred Connection Error
    return results


def get_comments(pages: int) -> DataFrame:
    """Get List of comments per page"""
    comments_delayed = (
        get_elements_json(url=COMMENTS_ENDPOINT, page=str(page), file="comments.dat")
        for page in range(1, pages + 1)
    )

    comments_bag = db.from_delayed(comments_delayed).map(
        lambda d: {
            "id": d["id"],
            "post": d["post"],
            "author_name": d["author_name"],
            "date": create_datetime(d["date"]),
            "content": d["content"]["rendered"],
            "link": d["link"],
        }
    )
    # In Windows the Dask.Bag is multiprocessing by default, change to threads
    with dask.config.set(scheduler="threads"):
        comments: List[dict] = comments_bag.compute()
    df_comments = DataFrame(comments)
    return df_comments


def get_searches(words: List[str]) -> List[Dict[str, Any]]:
    """Get List of searches one page"""
    search_delayed = (
        get_elements_json(url=SEARCH_ENDPOINT, search=word) for word in words
    )

    searches_bag = db.from_delayed(search_delayed)
    with dask.config.set(scheduler="threads"):
        searches = searches_bag.compute()
    return searches


# Load model
nlp = spacy.load("es_core_news_sm")
bow_lemma_token: Dict[str, Token] = dict()


def remplace_accents(text: str) -> str:
    """Remplace spanish accents vocab lower case: Unicode code point literal to str"""
    text = re.sub(r"á", "a", text, flags=re.I)
    text = re.sub(r"é", "e", text, flags=re.I)
    text = re.sub(r"í", "i", text, flags=re.I)
    text = re.sub(r"ó", "o", text, flags=re.I)
    text = re.sub(r"ú", "u", text, flags=re.I)
    return text


def get_text(markup: str) -> str:
    """Remove html tags, URLs and spaces using regexp"""
    text = re.sub(r"<.*?>", "", markup)
    url_pattern = r"(http|ftp)s?://(?:[a-zA-Z]|[0-9]|[$-_@.&#+]|[!*\(\),]|\
                   (?:%[0-9a-fA-F][0-9a-fA-F]))+"
    text = re.sub(url_pattern, "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_token(token: Token) -> str:
    """Remove grave accents and return lemmatized token lower case"""
    # TODO: Avoid token like "inmiscuir el"
    result = remplace_accents(token.lemma_.strip().split(" ")[0].lower())
    return result


def is_token_allowed(token: Token) -> bool:
    """No Stop words, No Punctuations or len token >= 3"""
    if (
        not token
        or not token.text.strip()
        or token.is_stop
        or token.is_punct
        or len(token) < 3
    ):
        return False
    return True


def clean(doc: str) -> List[str]:
    """Remove grave accents, stopwords, the punctuations and normalize the corpus."""
    text = get_text(doc)
    text = text.lower()
    tokens = []
    global bow_lemma_token

    for token in nlp(text):
        if is_token_allowed(token):
            word_lemma = preprocess_token(token)
            bow_lemma_token[word_lemma] = token
            tokens.append(word_lemma)

    return tokens


def create_ngram(document: DocumentList, ngram=1) -> DocumentList:
    """Create N-Gram from document"""
    return [
        " ".join(document[i : i + ngram]) for i in range(len(document) - (ngram - 1))
    ]


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
    return [comment_tf(comment) for comment in documents]


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
    """TF-IDF of all Documents Normalized"""
    tfidf_comments = []

    idf_comments = idf_dict(documents)

    def compute_tfidf_comment(document: DocumentList) -> TFIDF:
        """Compute TF-IDF for a Document"""
        tfidf_comment = dict()

        tf_comment = comment_tf(document)

        for word in set(document):
            tfidf_comment[word] = tf_comment[word] * idf_comments[word]

        return tfidf_comment

    for comment in documents:
        tfidf_comments.append(compute_tfidf_comment(comment))

    return tfidf_comments


def sort_tfidf(tfidf_unordered) -> Iterable[Tuple[Any, Any]]:
    return sorted(tfidf_unordered.items(), key=lambda x: x[1], reverse=True)


def export_image2html(image_name: str) -> None:
    """Export a image to html file"""
    with open(image_name, "rb") as file_img:
        data = base64.b64encode(file_img.read()).decode("utf-8")

    img_str = """
    <img width="100%" height="100%"
    src="data:image/png;base64,{}" />
    """.format(
        data
    )

    with open("wordcloud_cubadebate.html", "w") as _html:
        _html.write(img_str)


if __name__ == "__main__":
    """Entry point"""
    # Cell
    _start = time.time()
    df_comments = get_comments(pages=1)
    _end = time.time()
    print(f"{len(df_comments)} Comments downloaded in {(_end - _start):.2f}s")
    # df_comments.info()
    # df_comments.head()

    # Cell
    # Truncate file comments.dat
    save_elements_json("comments.dat")
    NUM_PAGES = 100

    _start = time.time()
    df_comments = get_comments(pages=NUM_PAGES)
    _end = time.time()
    print(f"{len(df_comments)} Comments downloaded in {(_end - _start):.2f}s")
    # print(df_comments.info())
    df_comments

    # Cell
    assert not df_comments.empty, "No comments to process"
    # Uncomment the next for get the comments from file
    # df_comments = get_comments_file("comments.dat")
    # print(f"{len(df_comments)} Comments from file comments.dat")
    # df_comments

    # Cell
    # Process corpus. Normalize. Using threads
    _start = time.time()
    ddf_comments = dd.from_pandas(df_comments, chunksize=10)
    df_comments["text"] = (
        ddf_comments["content"].apply(clean, meta=("content", "object")).compute()
    )
    documents_normalized = list(df_comments["text"].values)
    _end = time.time()
    print(f"{len(documents_normalized)} Documents normalized in {(_end - _start):.2f}s")
    # print(df_comments.info())
    df_comments

    # Cell
    # TF-IDF Comments list
    tfidf_list = comments_tfidf(documents_normalized)

    unordered_tfidf: Dict[str, float] = dict()

    for tfidf in tfidf_list:
        for w, value in tfidf.items():
            unordered_tfidf[w] = unordered_tfidf.get(w, 0) + value

    # unordered_tfidf
    ordered_comments_tfidf = OrderedDict(sort_tfidf(unordered_tfidf))

    # WordCloud with word_token bigrams (ngram=2)
    token_comments_tfidf = dict()
    unordered_tfidf_ngram: Dict[str, float] = dict()
    documents_normalized_ngram = list()

    for doc_norm in documents_normalized:
        documents_normalized_ngram.append(create_ngram(doc_norm, ngram=2))

    tfidf_list_ngram = comments_tfidf(documents_normalized_ngram)

    for tfidf in tfidf_list_ngram:
        for w, value in tfidf.items():
            unordered_tfidf_ngram[w] = unordered_tfidf_ngram.get(w, 0) + value

    ordered_comments_tfidf_ngram = OrderedDict(sort_tfidf(unordered_tfidf_ngram))

    for lemma_, value in ordered_comments_tfidf_ngram.items():
        _token = " ".join([str(bow_lemma_token[lm]) for lm in lemma_.split(" ")])
        token_comments_tfidf[_token] = value

    # Save to JSON file
    with open("comments_tfidf.json", "w") as file_json:
        json.dump(ordered_comments_tfidf_ngram, file_json)
    print("TF-IDF ordered saved to comments_tfidf.json")

    # Cell
    # Capitolio Habana
    IMG_CAPITOLIO = os.getenv("IMG_CAPITOLIO") or "capitolio.jpg"

    # get data directory (using getcwd() is needed to support running example
    # in generated IPython notebook)
    _dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()

    IMG_CAPITOLIO: str = os.path.join(_dir, IMG_CAPITOLIO)
    IMG_CAPITOLIO_LINK: str = (
        "https://upload.wikimedia.org/wikipedia/commons/8/8f/Capitolio_full.jpg"
    )

    # download mask images
    if not os.path.isfile(IMG_CAPITOLIO):
        response = requests.get(IMG_CAPITOLIO_LINK)

        if response.status_code == 200:
            with open(IMG_CAPITOLIO, "wb") as _capitolio:
                _capitolio.write(response.content)
                print("Image downloaded.")
        else:
            print("Image No downloaded!")
    else:
        image_downloaded: str = IMG_CAPITOLIO.split("\\")[-1].split("/")[-1]
        print(f"Image: {image_downloaded}")

    # Cell
    # WordsCloud Cubadebate
    IMG_WORDCLOUD = "wordcloud_cubadebate.png"

    # read the mask image
    _mask = np.array(Image.open(IMG_CAPITOLIO))

    # Generate Word Cloud
    wordcloud = WordCloud(
        max_words=500,
        #     max_font_size=50,
        height=1440,
        width=2160,
        background_color="white",
        mask=_mask,
        contour_width=1,
        contour_color="steelblue",
        stopwords=STOP_WORDS,
    ).generate_from_frequencies(token_comments_tfidf)
    # Save to file
    wordcloud.to_file(IMG_WORDCLOUD)
    print("WordCloud Cubadebate image saved.\n")

    # Cell
    # Get Words (Lemma) with most TF-IDF (unigram: ngram=1)
    df = (
        DataFrame.from_dict(
            data=ordered_comments_tfidf, dtype=int, orient="index", columns=["TF-IDF"]
        )
        .reset_index()
        .rename(index=str, columns={"index": "Word"})
    )
    rows, columns = df.shape
    print(f"Panda DataFrame shape:({rows}, {columns})")

    df_gb = df.groupby(["TF-IDF"], sort=False).first().reset_index()
    rows, columns = df_gb.shape
    print(f"GroupBy DataFrame shape:({rows}, {columns})\n")
    print(df_gb)

    words_token = list(df_gb["Word"].values)
    searches_dict = dict()
    searches_ids = set()

    for w_token in words_token:
        search_list = get_searches([w_token])
        # Get first searched result
        if len(search_list) > 0:
            search_id = search_list[0]["id"]
            if search_id not in searches_ids:
                searches_ids.add(search_id)
                searches_dict[w_token] = search_list[0]

    # DataFrame First Search Posts
    if len(searches_dict) > 0:
        searches_df = DataFrame.from_dict(data=searches_dict, orient="index",).drop(
            columns=["type", "subtype", "_links"]
        )

        df_dates = searches_df["url"].str.extract(r"(?P<date>\d{4}/\d{2}/\d{2})")

        searches_by_dates = searches_df.join(df_dates).sort_values(
            by="date", ascending=False
        )

        print("\nTop Word Token and Post\n")
        print(searches_by_dates)
        searches_by_dates.to_json(os.path.join(_dir, "top_word_post.json"))
        print("\nSaved top_word_post.json\n")

        if os.path.isfile("index.tpl"):
            with open("index.html", "w", encoding="utf-8") as f_index, open(
                "index.tpl", "r"
            ) as file:
                tpl = file.read()
                index_template = Template(tpl)

                text_link = "<ul>\n"
                for _index, _title, _url in searches_by_dates.reset_index()[
                    ["index", "title", "url"]
                ].values:
                    text_link += (
                        f"\t<li><a href='{_url}' rel='external' "
                        f"data-token='{_index}'>{_title}</a></li>\n"
                    )
                text_link += "</ul>"
                # Create index.html from index.tpl
                f_index.write(index_template.substitute(CUBADEBATE_LINKS=text_link))
            print("Rewrite index.html.\n")
