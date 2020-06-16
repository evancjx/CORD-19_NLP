
from src.helper import chunks
from transformers import BertForQuestionAnswering as Bert4QA, BertTokenizer
import numpy as np
import torch

class BERT:
    def __init__(self, pretrained='bert-large-uncased-whole-word-masking-finetuned-squad'):
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.QA_MODEL = Bert4QA.from_pretrained(pretrained)
        self.QA_MODEL.to(self.torch_device)
        self.QA_MODEL.eval()
        self.QA_TOKENIZER = BertTokenizer.from_pretrained(pretrained)

    def _split_document(self, question, doc):
        self.seq_ids = self.QA_TOKENIZER.encode(question, doc)
        doc_tokens = doc.split()
        num_split = int(np.ceil(len(self.seq_ids)*1.2/256))
        if num_split > 1:
            length_words = len(doc_tokens)
            group_num = length_words//num_split
            overlap = int(group_num*1.2//2)
            
            return [
                self.QA_TOKENIZER.encode(question, dp) 
                for dp in [
                    ' '.join(doc_tokens[start:end])
                    for start, end in chunks(length_words, group_num, overlap)
                ]
            ]
        else:
            return [self.seq_ids]

    def reconstructText(self, tokens, start=0, stop=-1):
        tokens = tokens[start:stop]
        if '[SEP]' in tokens:
            tokens = tokens[tokens.index('[SEP]')+1:]
        txt = ' '.join(tokens)
        txt = txt.replace(' ##', '')
        txt = txt.replace('##', '')
        txt = txt.strip()
        txt = " ".join(txt.split())
        txt = txt.replace(' .', '.')
        txt = txt.replace('( ', '(')
        txt = txt.replace(' )', ')')
        txt = txt.replace(' - ', '-')
        txt_list = txt.split(' , ')
        txt = ''
        nTxtL = len(txt_list)
        if nTxtL == 1:
            return txt_list[0]
        newList =[]
        for i,t in enumerate(txt_list):
            if i < nTxtL -1:
                if t[-1].isdigit() and txt_list[i+1][0].isdigit():
                    newList += [t,',']
                else:
                    newList += [t, ', ']
            else:
                newList += [t]
        return ''.join(newList)

    def _get_scores(self, doc_part_seq_ids):
        answers, confidences = [], []
        for part_seq_ids in doc_part_seq_ids:
            part_seq_tokens = self.QA_TOKENIZER.convert_ids_to_tokens(part_seq_ids)

            num_seg_a = part_seq_ids.index(self.QA_TOKENIZER.sep_token_id)+1
            num_seg_b = len(part_seq_ids)-num_seg_a

            segment_ids = [0]*num_seg_a+[1]*num_seg_b
            assert len(segment_ids) == len(part_seq_ids)
            
            limit = 512
            if len(part_seq_ids) > limit:
                input_ids, type_ids = part_seq_ids[:limit], segment_ids[:limit]
            else:
                input_ids, type_ids = part_seq_ids, segment_ids
            
            start_scores, end_scores = self.QA_MODEL(
                input_ids=torch.tensor([input_ids]).to(self.torch_device), 
                token_type_ids=torch.tensor([type_ids]).to(self.torch_device)
            )

            start_scores, end_scores = start_scores[:,1:], end_scores[:,1:]

            answer_start, answer_end = torch.argmax(start_scores), torch.argmax(end_scores)
            answer = self.reconstructText(part_seq_tokens, answer_start, answer_end+2)

            if not answer: continue
            if answer.startswith('. ') or answer.startswith(', '): answer = answer[2:]

            answers.append(answer)
            confidences.append(start_scores[0,answer_start].item()+end_scores[0,answer_end].item())

        return answers, confidences

    def predict(self, question, doc):
        answers, confidences = self._get_scores(self._split_document(question, doc))

        if not answers: return {'answer': ''}

        best_idx = confidences.index(max(confidences))
        confidence, answer= confidences[best_idx], answers[best_idx]
        
        seq_tokens = self.QA_TOKENIZER.convert_ids_to_tokens(self.seq_ids)
        return {
            'answer': answer,
            'confidence': -1000000 if answer.startswith('[CLS]') or answer.endswith('[SEP]') else confidence,
            'abstract_bert': self.reconstructText(seq_tokens[seq_tokens.index('[SEP]')+1:])
        }
