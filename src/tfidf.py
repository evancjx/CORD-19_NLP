
from .helper import doc_dot_product, sort_dict
from collections import Counter
from math import log, sqrt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

class TFIDF:
    def __init__(self, corpus, text_preprocessor=None):
        self.n_doc = len(corpus)
        self.text_preprocessor = text_preprocessor

        if text_preprocessor:
            corpus = self._text_prep(corpus)

        self.corpus = corpus
        self._initialize()
        self._cal_idf()
        self._cal_tfidf()

    def _text_prep(self, corpus):
        with Pool(cpu_count()) as pool:
            return list(
                tqdm(
                    pool.imap(
                        self.text_preprocessor, corpus
                    ), total=self.n_doc
                )
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
        if not isinstance(self.corpus, list):
            raise TypeError('corpus has to be a list of documents')
        
        self.corpus_doc_tf = list()
        self.term_df = Counter() # Count the number of time the word (term) happens in different documents

        for document in tqdm(
            self.corpus, 
            desc='TF and DF for each term on each document'
        ):
            document = self._check_doc(document)
            tf = self._term_freq(document)
            self.corpus_doc_tf.append(tf)
            self.term_df.update(tf.keys())

    def _cal_idf(self):
        '''
            Rare items are more informative than frequent items
            Low weights for frequent terms
            High weights for rare terms
        '''
        self.term_idf = {
            word: log(self.n_doc / doc_freq) # original IDF
            for word, doc_freq in tqdm(
                self.term_df.items(), desc='IDF for each term'
            )
        }

    def _cal_tfidf(self):
        self.corpus_doc_tfidf = [
            self.tfidf_doc(doc_tf)
            for doc_tf in tqdm(
                self.corpus_doc_tf,
                desc='TF-IDF on each document'
            )
        ]

    def tfidf_doc(self, doc_tf):
        mag_weight = 0.0
        for term, term_freq in doc_tf.items():
            tf = log(1 + term_freq) # log normalization
            # idf = log(self.n_doc / (self.term_df.get(term) or 1)) # original IDF
            idf = self.term_idf.get(term) or log(self.n_doc)
            tfidf = tf*idf

            doc_tf[term] = tfidf
            mag_weight += tfidf**2

        mag_weight = sqrt(mag_weight)

        if mag_weight != 0:
            doc_tf = {
                term: tfidf/mag_weight
                for term, tfidf in doc_tf.items()
            }
        
        return doc_tf

    def doc_keywords(self, document, n_keywords=10):
        document = self._check_doc(document)
        return list(sort_dict(self.tfidf_doc(self._term_freq(document)), 'value', True))[:n_keywords]

    def search_similar(self, document, top_n=10):
        document = self._check_doc(document)
        query_tf_idf = self.tfidf_doc(self._term_freq(document))

        return sort_dict(
            dict(
                [
                    (idx, doc_dot_product(query_tf_idf, self.corpus_doc_tfidf[idx]))
                    for idx in range(self.n_doc)
                ]
            ), 'value', True, top_n
        )
