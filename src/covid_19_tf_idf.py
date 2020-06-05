
# from .tf_idf import term_freq, doc_tf_idf
from .tf_idf import TFIDF

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

def search_relevant_articles_tf_idf(
    query, n_articles, data_df, 
    corpus_doc_tf_idf, term_doc_freq, 
    query_preprocess_func
):
    query_processed = query_preprocess_func(query)

    query_tf = TFIDF.term_freq(query_processed)

    query_td_idf = TFIDF.doc_tf_idf(
        doc_tf = query_tf, 
        term_doc_freq = term_doc_freq, 
        len_corpus = len(corpus_doc_tf_idf)
    )

    query_parsed_list = [
        (idx, doc_dot_product(query_td_idf, corpus_doc_tf_idf[idx]))
        for idx in range(len(corpus_doc_tf_idf))
    ]
    query_parsed_list = sorted(query_parsed_list, key=lambda x: x[1], reverse=True)

    result_df = pd.concat(
        [data_df.iloc[query_parsed_list[-idx][0]] for idx in range(n_articles)],
        ignore_index=True, axis=1
    ).T
    
    return result_df

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