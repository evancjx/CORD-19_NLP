
from .helper import doc_dot_product, sort_dict
from .text_preprocessing import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tqdm import tqdm
import numpy as np

class sklearn_TFIDF:
    feature_names = None

    def __init__(
        self, 
        corpus,
        count_vectorizer=CountVectorizer(
            max_df=0.8, stop_words=STOP_WORDS, max_features=10000, ngram_range=(1,1)
        ), 
        tfidf_transformer=TfidfTransformer(
            smooth_idf=True,use_idf=True
        )
    ):
        self.n_doc = len(corpus)
        
        self.count_vectorizer = count_vectorizer
        self.tfidf_transformer = tfidf_transformer

        self.corpus = corpus
        self._cal_tfidf()
    
    # sort coordinate matrix based by score then word index
    def _sort_coo(self, coo_matrix, descending=True):
        return {
            k: v
            for k, v in sorted(
                zip(coo_matrix.col, coo_matrix.data), 
                key=lambda x: (x[1], x[0]), 
                reverse=descending
            )
        }
    
    def _cal_tfidf(self):
        self.tfidf_transformer.fit(
            self.count_vectorizer.fit_transform(
                self.corpus
            )
        )
    
        self.feature_names = self.count_vectorizer.get_feature_names()

        self.corpus_doc_tfidf = [
            self.tfidf_text(doc)
            for doc in tqdm(self.corpus, desc='Conduct TFIDF for individual documents')
        ]

    def tfidf_text(self, text):
        if self.feature_names is None:
            print('Conduct tfidf on corpus')
            return None
        if not isinstance(text, list):
            text = [text]

        return self._sort_coo(
            self.tfidf_transformer.transform(
                self.count_vectorizer.transform(
                    text
                )
            ).tocoo()
        )

    def get_text_keywords(self, text, n_keywords=10):
        return np.array(self.feature_names)[
            list(self.tfidf_text(text).keys())[:n_keywords]
        ].tolist()

    def search_similar(self, document, top_n=10):
        query_term_tfidf = self.tfidf_text(document)
        
        return [
            doc_dot_product(query_term_tfidf, self.corpus_doc_tfidf[idx])
            for idx in range(self.n_doc)
        ]
