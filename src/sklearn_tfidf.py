
from .text_preprocessing import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
class sklearn_TFIDF:
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
