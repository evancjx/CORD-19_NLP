
from .bm25 import BM25
from .helper import form_dict, sort_dict
from .tfidf import TFIDF
from tqdm import tqdm
import numpy as np
import pandas as pd

class rel_retrieve(TFIDF):
    def __init__(self, data_df, text_preprocessor, corpus=None, k1=1.2, b=0.75):
        self.k1 = k1; self.b = b

        if not corpus:
            corpus = list(data_df['title']+' '+data_df['abstract']+' '+data_df['text'])
        
        super().__init__(corpus, text_preprocessor)

        data_df = data_df.reindex(columns=list(data_df.columns)+['keywords'])
        tqdm.pandas()
        print('Obtain 20 keywords from each documents using TFIDF')
        data_df['keywords'] = pd.Series(corpus).progress_apply(
            lambda doc: self.doc_keywords(doc, 20)
        )

        self.keywords_paperId = form_dict(data_df, 'keywords', 'paper_id')
        self.df_dict = data_df.set_index('paper_id').T.to_dict('list')
        self.col_idx_title = {
            title: idx
            for idx, title in enumerate(data_df.drop(columns='paper_id').columns)
        }

    def _query_bm25(self, potential_doc_ids_list, weights, query_tokens):
        potential_doc_dict = {
            col: [
                self.df_dict.get(doc_id)[self.col_idx_title[col]].split()
                for doc_id in potential_doc_ids_list
            ]
            for col in weights.keys()
        }

        potential_doc_dict = {
            col+'_scores': BM25(potential_doc_dict[col]).get_scores(query_tokens)**weight
            for col, weight in weights.items()
        }
        
        return dict(zip(
            potential_doc_ids_list, 
            np.add(
                *[potential_doc_dict[col+'_scores'] for col in weights]
            )
        ))
    
    def search_similar(self, query_tokens, weights={'title': 1, 'abstract': 2}):
        '''
            Get list of potential documents ids (paper_id)
            from matching keywords 
            possibly generated from possible TFIDF
        '''
        query_tokens = self._check_doc(query_tokens)
        potential_documents_id = [
            doc_id 
            for word in query_tokens
            for doc_id in self.keywords_paperId.get(word, [])
        ]

        return self._query_bm25(
            potential_documents_id,
            weights, query_tokens
        )
