
from src.helper import form_dict, sort_dict

'''
    Install BM25 with:
        !pip install rank_bm25
'''
from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd

class BM25(object):
    def __init__(self, data_df):
        self.keywords_paperId = form_dict(data_df, 'keywords', 'paper_id')
        self.df_dict = data_df.set_index('paper_id').T.to_dict('list')
        self.col_idx_title = {
            title: idx
            for idx, title in enumerate(data_df.drop(columns='paper_id').columns)
        }

    def query_bm25(self, potential_doc_ids_list, weights, query_tokens):
        potential_doc_dict = {
            col: [
                self.df_dict.get(doc_id)[self.col_idx_title[col]].split()
                for doc_id in potential_doc_ids_list
            ]
            for col in weights.keys()
        }

        potential_doc_dict = {
            col+'_scores': BM25Okapi(potential_doc_dict[col]).get_scores(query_tokens)**weight
            for col, weight in weights.items()
        }
        
        score_dict = sort_dict(dict(zip(
            potential_doc_ids_list, 
            np.add(
                *[potential_doc_dict[col+'_scores'] for col in weights]
            )
        )), 'value', True)
        
        return {
            paper_id: score_dict.get(paper_id)
            for paper_id in list(score_dict)[:10]
        }

    def search_similar(self, query_tokens, weights={'title': 1, 'abstract': 2}):
        potential_documents_id = [
            doc_id 
            for word in query_tokens
            for doc_id in self.keywords_paperId.get(word, [])
        ]

        result = self.query_bm25(
            potential_documents_id,
            weights, query_tokens
        )

        return pd.DataFrame(
            result.values(), 
            [
                self.df_dict.get(key)[self.col_idx_title['title']] 
                for key in result.keys()
            ], 
            columns=['Score']
        )
