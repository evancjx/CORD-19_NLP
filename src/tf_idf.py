
from collections import Counter
from math import log, sqrt

def term_freq(doc):
    tf = Counter()
    tf.update(doc.split())
    return tf

def document_frequency(corpus):
    vocabulary = set()
    term_doc_freq = Counter() # Count the number of time the word (term) happens in different documents

    corpus_doc_tf = []
    for doc in corpus:
        corpus_doc_tf.append(term_freq(doc))
        doc_terms = set(doc.split())
        vocabulary = vocabulary | doc_terms # Collect all possible vocabulary without duplicates
        term_doc_freq.update(doc_terms) # Update term count
    
    return corpus_doc_tf, term_doc_freq, vocabulary

def inverse_document_frequency(corpus):
    corpus_doc_tf, term_doc_freq, vocabulary = document_frequency(corpus)

    '''
        Rare items are more informative than frequent items
        Low positive weights for frequent terms
        High positive weights for rare terms
    '''
    term_inverse_doc_freq = {
        word: log(len(corpus)/doc_freq) # Inverse Document Frequency scoring
        for word, doc_freq in term_doc_freq.items()
    }
    
    return corpus_doc_tf, term_inverse_doc_freq, vocabulary

def doc_tf_idf(doc_tf, term_doc_freq, len_corpus):
    mag_weight = 0.0
    for term, term_freq in doc_tf.items():
        term_freq = doc_tf[term]

        if term in term_doc_freq:
            idf = log(len_corpus / term_doc_freq[term])
        else:
            idf = log(len_corpus)
        
        tfidf = log(1 + term_freq) * idf

        doc_tf[term] = tfidf
        mag_weight += tfidf**2

    mag_weight = sqrt(mag_weight)
    
    if mag_weight != 0:
        for term in doc_tf:
            doc_tf[term] /= mag_weight
    return doc_tf

def corpus_tf_idf(corpus):
    corpus_doc_tf, term_doc_freq, _ = document_frequency(corpus)
    
    return [
        doc_tf_idf(doc_tf, term_doc_freq, len(corpus_doc_tf))
        for doc_tf in corpus_doc_tf
    ], term_doc_freq

from .text_preprocessing import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
class sklearn_TFIDF(object):
    feature_names = None

    def __init__(
        self, 
        count_vectorizer=CountVectorizer(
            max_df=0.8, stop_words=STOP_WORDS, max_features=10000, ngram_range=(1,1)
        ), 
        tfidf_transformer=TfidfTransformer(
            smooth_idf=True,use_idf=True
        )
    ):
        self.count_vectorizer = count_vectorizer
        self.tfidf_transformer = tfidf_transformer
    
    def tfidf_corpus(self, corpus):
        self.corpus = corpus

        self.tfidf_transformer.fit(
            self.count_vectorizer.fit_transform(
                corpus
            )
        )
    
        self.feature_names = self.count_vectorizer.get_feature_names()

    def tfidf_text(self, text):
        if self.feature_names is None:
            print('Conduct tfidf on corpus')
            return None
        if not isinstance(text, list):
            text = [text]

        return self.tfidf_transformer.transform(
            self.count_vectorizer.transform(
                text
            )
        )

    # sort coordinate matrix based by score then word index
    def sort_coo(self, coo_matrix, descending=True):
        return {
            k: v
            for k, v in sorted(
                zip(coo_matrix.col, coo_matrix.data), 
                key=lambda x: (x[1], x[0]), 
                reverse=descending
            )
        }

    def get_text_keywords(self, text, n_keywords=10):
        return np.array(self.feature_names)[
            list(self.sort_coo(self.tfidf_text(text).tocoo()).keys())[:n_keywords]
        ]
