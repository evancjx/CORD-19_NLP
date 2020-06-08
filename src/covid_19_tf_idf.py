
from math import log, sqrt
import numpy as np
import pandas as pd

def doc_dot_product(vector, doc_vector):
    dot_product = 0.0
    if len(vector) < len(doc_vector):
        for key in vector:
            if key in doc_vector:
                dot_product+=vector[key]*doc_vector[key]
    else:
        for key in doc_vector:
            if key in vector:
                dot_product+=vector[key]*doc_vector[key]
    return dot_product

def sk_tfidf_search(
    query_string, data, sk_tfidf, n_articles=10
):
    query_term_tfidf = sk_tfidf.tfidf_text(query_string)
    
    query_parsed_list_corpus_id = sorted(
        [
            (idx, doc_dot_product(query_term_tfidf, sk_tfidf.corpus_doc_tfidf[idx]))
            for idx in range(len(sk_tfidf.corpus_doc_tfidf))
        ], key=lambda item: item[1], reverse=False
    )

    return pd.concat(
        [
            data.iloc[query_parsed_list_corpus_id[-idx][0]]
            for idx in range(n_articles)
        ], ignore_index=True, axis=1
    ).T