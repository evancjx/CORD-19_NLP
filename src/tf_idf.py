
from .helper import sort_dict
from collections import Counter
from math import log, sqrt
from tqdm import tqdm
class TFIDF(object):

    def __init__(self):
        pass
    
    def term_freq(self, doc):
        tf = Counter()
        tf.update(doc.split())
        return tf

    def document_frequency(self, corpus):
        if not isinstance(corpus, list):
            raise TypeError('corpus has to be a list of documents')
        self.corpus = corpus

        vocabulary = set()
        term_doc_freq = Counter() # Count the number of time the word (term) happens in different documents

        corpus_doc_tf = []
        for doc in tqdm(
            corpus, desc='Conducting DF (Document Frequency) on corpus'
        ):
            corpus_doc_tf.append(self.term_freq(doc))
            doc_terms = set(doc.split())
            vocabulary = vocabulary | doc_terms # Collect all possible vocabulary without duplicates
            term_doc_freq.update(doc_terms) # Update term count
        
        self.corpus_doc_tf = corpus_doc_tf
        self.term_doc_freq = term_doc_freq
        self.vocabulary = vocabulary

    def inverse_document_frequency(self, corpus):
        self.document_frequency(corpus)

        '''
            Rare items are more informative than frequent items
            Low positive weights for frequent terms
            High positive weights for rare terms
        '''
        self.term_inverse_doc_freq = {
            word: log(len(corpus)/doc_freq) # Inverse Document Frequency scoring
            for word, doc_freq in self.term_doc_freq.items()
        }

        return self.term_inverse_doc_freq

    def tfidf_text(self, doc_tf):
        len_corpus = len(self.corpus_doc_tf)
        mag_weight = 0.0
        for term, term_freq in doc_tf.items():
            term_freq = doc_tf[term]

            if term in self.term_doc_freq:
                idf = log(len_corpus / self.term_doc_freq[term])
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

    def tfidf_corpus(self, corpus):
        self.document_frequency(corpus)
        
        self.corpus_doc_tfidf = [
            self.tfidf_text(doc_tf)
            for doc_tf in tqdm(
                self.corpus_doc_tf, desc='Conducting TF-IDF on each document'
            )
        ]

    def get_text_keywords(self, text, n_keywords=10):
        return list(sort_dict(self.tfidf_text(self.term_freq(text)), 'value', True))[:n_keywords]

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

        self.corpus_doc_tfidf = [
            self.tfidf_text(doc)
            for doc in tqdm(corpus, desc='Conduct TFIDF for individual documents')
        ]

    def tfidf_text(self, text):
        if self.feature_names is None:
            print('Conduct tfidf on corpus')
            return None
        if not isinstance(text, list):
            text = [text]

        return self.sort_coo(
            self.tfidf_transformer.transform(
                self.count_vectorizer.transform(
                    text
                )
            ).tocoo()
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
            list(self.tfidf_text(text).keys())[:n_keywords]
        ].tolist()
