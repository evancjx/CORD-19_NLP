
from collections import Counter
from math import log
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np

class BM25:
    def __init__(self, corpus, text_preprocessor=None, k1=1.2, b=0.75):
        self.n_doc = len(corpus)
        self.text_preprocessor = text_preprocessor
        self.k1=k1; self.b=b

        if text_preprocessor:
            corpus = self._text_prep(corpus)

        self.corpus = corpus
        self._initialize()
        self._cal_idf()

    def _text_prep(self, corpus):
        with Pool(cpu_count()) as pool:
            return list(
                tqdm(
                    pool.imap(
                        self.text_preprocessor, corpus
                    ), total=self.n_doc
                ), 
            )
    
    def _check_doc(self, document):
        if isinstance(document, list):
            return document
        elif isinstance(document, str):
            return document.split()
        else:
            raise TypeError('Document has to be either a String or a List. Given {} instead.'.format(type(document)))

    def _term_freq(self, document_tokens):
        tf = Counter()
        tf.update(document_tokens)
        return tf

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
                self.term_df.items(), desc='Calculate IDF'
            )
        }
    
    def get_scores(self, query):
        query = self._check_doc(query)

        score = np.zeros(self.n_doc)
        for query_token in query:
            query_token_freq = np.array([(doc.get(query_token) or 0) for doc in self.corpus_doc_tf])
            score += (self.term_idf.get(query_token) or 0) * (
                (query_token_freq * self.k1 + 1) /
                (
                    query_token_freq + self.k1 * (
                        1 - self.b + self.b * (
                            # np.array(self.len_doc) / 
                            self.n_doc /
                            self.avgDl
                        )
                    )
                )
            )
        
        return dict(zip(range(self.n_doc), score))
