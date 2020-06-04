
from src.helper import sort_dict
from src.helper import form_dict

'''
    Install BM25 with:
        !pip install rank_bm25
'''
from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd

def bm25(potential_doc_ids_list, df_dict, col_idx_title, weights, query):
    potential_doc_dict = {
        col: [
            df_dict.get(doc_id)[col_idx_title[col]].split()
            for doc_id in potential_doc_ids_list
        ]
        for col in weights.keys()
    }
    for col, weight in weights.items():
        potential_doc_dict.update(
            {
                col+'_scores': BM25Okapi(potential_doc_dict[col]).get_scores(query)**weight
            }
        )
    
    score_dict = sort_dict(dict(zip(
        potential_doc_ids_list, 
        np.add(
            *[potential_doc_dict[col+'_scores']for col in weights]
        )
    )), 'value', True)
    
    return {
        paper_id: score_dict.get(paper_id)
        for paper_id in list(score_dict)[:10]
    }

def query_bm25(queries_tokens, data_df):
    keywords_paperId = form_dict(data_df, 'keywords', 'paper_id')
    df_dict = data_df.set_index('paper_id').T.to_dict('list')
    col_idx_title = {
        title: idx
        for idx, title in enumerate(data_df.drop(columns='paper_id').columns)
    }

    potential_documents_id = [
        doc_id 
        for word in queries_tokens
        for doc_id in keywords_paperId.get(word, [])
    ]

    result = bm25(
        potential_documents_id, df_dict, col_idx_title,
        {'title': 1, 'abstract': 2}, queries_tokens        
    )
    
    return pd.DataFrame(
        result.values(), 
        [
            df_dict.get(key)[col_idx_title['title']] 
            for key in result.keys()
        ], 
        columns=['Score']
    )