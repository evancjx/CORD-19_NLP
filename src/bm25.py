
from .tfidf import TFIDF
from .helper import sort_dict
from collections import Counter
from math import log
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np

class BM25(TFIDF):
    '''
        Following the formula of https://blog.mimacom.com/bm25-got/
        https://opensourceconnections.com/blog/2015/10/16/bm25-the-next-generation-of-lucene-relevation/
    '''
    def __init__(self, corpus, text_preprocessor=None, k1=1.2, b=0.75):
        self.n_doc = len(corpus)
        self.text_preprocessor = text_preprocessor
        self.k1=k1; self.b=b

        if text_preprocessor:
            corpus = self._text_prep(corpus)

        self.corpus = corpus
        self._initialize()
        self._cal_idf()

    def _initialize(self):
        self.len_doc = []
        self.corpus_doc_tf = []
        self.term_df = Counter()

        total_length = 0
        for document in tqdm(
            self.corpus, desc='Conducting TF and DF on corpus'
        ):
            document = self._check_doc(document)
            self.len_doc.append(len(document))
            total_length += len(document)
            tf = self._term_freq(document)
            self.corpus_doc_tf.append(tf)
            self.term_df.update(tf.keys())

        self.avgDl = total_length/self.n_doc

    def _cal_idf(self):
        '''
            Rare items are more informative than frequent items
            Low weights for frequent terms
            High weights for rare terms

            https://lucene.apache.org/core/8_0_0/core/org/apache/lucene/search/similarities/BM25Similarity.html
            idf(long docFreq, long docCount)
            Implemented as log(1 + (docCount - docFreq + 0.5)/(docFreq + 0.5))
        '''
        self.term_idf = {
            word: log(
                # 1 + # prevent negative values (?) 
                (
                    (self.n_doc - doc_freq + 0.5) /
                    (doc_freq + 0.5)
                )
            )
            for word, doc_freq in tqdm(
                self.term_df.items(), desc='[BM25] IDF for each term'
            )
        }
    
    def get_scores(self, query, top_n=10):
        query = self._check_doc(query)

        score = np.zeros(self.n_doc)
        for query_token in query:
            query_token_freq = np.array([(doc.get(query_token) or 0) for doc in self.corpus_doc_tf])
            score += (self.term_idf.get(query_token) or 0) * (
                (
                    query_token_freq * self.k1 + 1
                ) /
                (
                    query_token_freq + self.k1 * (
                        1 - self.b + self.b * (
                            np.array(self.len_doc) / 
                            # self.n_doc /
                            self.avgDl
                        )
                    )
                )
            )
        
        return score

class BM25L(BM25):
    def __init__(self, corpus, text_preprocessor=None, k1=1.2, b=0.75, delta=0.5):
        self.delta = 0.5
        super().__init__(corpus, text_preprocessor, k1, b)

    def _cal_idf(self):
        '''
            Rare items are more informative than frequent items
            Low weights for frequent terms
            High weights for rare terms

            http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf
        '''
        self.term_idf = {
            word: log(
                (self.n_doc + 1) /
                (doc_freq + 0.5)
            )
            for word, doc_freq in tqdm(
                self.term_df.items(), desc='[BM25] IDF for each term'
            )
        }

    def get_scores(self, query, top_n=10):
        query = self._check_doc(query)

        score = np.zeros(self.n_doc)
        for query_token in query:
            query_token_freq = np.array([(doc.get(query_token) or 0) for doc in self.corpus_doc_tf])
            c_td = query_token_freq / (1 - self.b + self.b * (np.array(self.len_doc) / self.avgDl))
            score += (self.term_idf.get(query_token) or 0) * (
                (
                    (self.k1 + 1) * (c_td + self.delta)
                ) /
                (
                    self.k1 + (c_td + self.delta)
                )
            )

        return score

class BM25plus(BM25):
    def __init__(self, corpus, text_preprocessor=None, k1=1.2, b=0.75, delta=0.5):
        self.delta = 0.5
        super().__init__(corpus, text_preprocessor, k1, b)

    def _cal_idf(self):
        '''
            Rare items are more informative than frequent items
            Low weights for frequent terms
            High weights for rare terms

            http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf
        '''
        self.term_idf = {
            word: log(
                (self.n_doc + 1) / doc_freq
            )
            for word, doc_freq in tqdm(
                self.term_df.items(), desc='[BM25] IDF for each term'
            )
        }
    
    def get_scores(self, query, top_n=10):
        query = self._check_doc(query)

        score = np.zeros(self.n_doc)
        for query_token in query:
            query_token_freq = np.array([(doc.get(query_token) or 0) for doc in self.corpus_doc_tf])

            score += (self.term_idf.get(query_token) or 0) * (
                (
                    (
                        (self.k1 + 1) * query_token_freq
                    ) /
                    (
                        self.k1 * (1 - self.b + self.b * (np.array(self.len_doc)/self.avgDl)) + query_token_freq
                    ) + 
                    self.delta
                )
            )
        
        return score
