
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer as nltk_regex_tokenizer
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
nltk.download('stopwords')
import spacy
import re

regex_abbreviation = r'\((?!\s)[^a-z^()\s]+(?<!\s)\)|\b[A-Z]{2,}'
regex_special_char_digit = r'(\d|\W)'
regex_white_spaces = r'\s\s+'

STOP_WORDS = STOP_WORDS | set(stopwords.words('english'))

def retrieve_abbreviation(
    text, syntax=regex_abbreviation
):
    return re.findall(pattern=syntax, string=text)

def regex_replace(
    text, syntax, replace=''
):
    return re.sub(pattern=syntax, repl=replace, string=text)

def text_preprocess(
    text, tokenizer=None, stopwords=[]
): 
    text_abbreviations = set(
        regex_replace(abb, r'\(|\)', '')
        for abb in set(retrieve_abbreviation(text))
    )
    # print('{} abbreviation in text'.format(', '.join(text_abbreviations)), end='\n\n')

    text = regex_replace(text, regex_abbreviation)

    text = ' '.join(
        [
            word.lower()
            for word in text.split()
            if not word in text_abbreviations
        ]
    )

    text = regex_replace(text, regex_special_char_digit, ' ')

    text = regex_replace(text, regex_white_spaces, ' ')

    if tokenizer:
        text = " ".join(
            [
                token.lower()
                for token in tokenizer(text)
                if not token.lower() in stopwords
            ]
        )
    elif stopwords:
        text = " ".join(
            [
                token.lower()
                for token in text.strip().split()
                if not token.lower() in stopwords
            ]
        )
    
    return text

class spacy_NLP(object):
    def __init__(self, model='en_core_web_sm'):
        self.nlp = spacy.load(model)
        self.nlp.max_length = 1500000

    def __valid_token(self, tk):
        return tk.is_alpha and not tk.is_stop

    def __get_lemma(self, tk):
        if tk.pos_ == 'PRON' or tk.lemma_ == '-PRON-': return tk.text.lower()
        return tk.lemma_.lower()

    def tokenize_API(self):
        return lambda record: [
            self.__get_lemma(tk) 
            for tk in self.nlp(record) 
            if self.__valid_token(tk)
        ]

class nltk_NLP(object):
    def __init__(self, stemming=None, lemmatisation=None):
        if stemming and lemmatisation:
            stemmer = stemming()
            self.stemming = stemmer.stem

            lemmatizer = lemmatisation()
            self.lemmatisation = lemmatizer.lemmatize
        else:
            pass
    def tokenize_API(self, tokenizer=nltk_regex_tokenizer(pattern=r'\s+', gaps=True).tokenize):
        return lambda text: tokenizer(text)
    def custom_API(self):
        if self.stemming and self.lemmatisation:
            return lambda text: [
                self.lemmatisation(token) for token in [
                    self.stemming(word)
                    for word in text.split()
                ]
            ]