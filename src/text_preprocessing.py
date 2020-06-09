
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer as nltk_regex_tokenizer
from spacy.lang.en.stop_words import STOP_WORDS
import html
import nltk
nltk.download('stopwords')
import spacy
import re

regex_abbreviation = r'\((?!\s)[^a-z^()\s]+(?<!\s)\)|\b[A-Z]{2,}'
regex_special_char_digit = r'(\d|\W)'
regex_white_spaces = r'\s\s+'
regex_single_char = r'\b[a-zA-Z]\b'

STOP_WORDS = STOP_WORDS | set(stopwords.words('english'))

class spacy_NLP(object):
    def __init__(self, model='en_core_web_sm', max_length=2000000):
        self.nlp = spacy.load(model)
        self.nlp.max_length = max_length

    def __valid_token(self, tk):
        return tk.is_alpha and not tk.is_stop

    def __get_lemma(self, tk):
        if tk.pos_ == 'PRON' or tk.lemma_ == '-PRON-': return tk.text.lower()
        return tk.lemma_.lower()

    def tokenize(self, record):
        return [
            self.__get_lemma(tk) 
            for tk in self.nlp(record) 
            if self.__valid_token(tk)
        ]

class nltk_NLP(object):
    tokenizer = nltk_regex_tokenizer(pattern=r'\s+', gaps=True).tokenize

    def __init__(self, stemming=None, lemmatisation=None):
        if stemming and lemmatisation:
            self.stemming = stemming().stem
            self.lemmatisation = lemmatisation().lemmatize

            self.tokenizer = lambda text: [
                self.stemming(token) for token in [
                    self.lemmatisation(word)
                    for word in text.split()
                ]
            ]
    
    def tokenize(self, text):
        return self.tokenizer(text)

def retrieve_abbreviation(
    text, syntax=regex_abbreviation
):
    return re.findall(pattern=syntax, string=text)

def regex_replace(
    text, syntax, replace=''
):
    return re.sub(pattern=syntax, repl=replace, string=text)

def remove_html_tags(text):
    return ' '.join(item.strip() for item in BeautifulSoup(text, features='lxml').find_all(text=True))

def remove_html_elements(text):
    return html.unescape(remove_html_tags(remove_url(text)))

def remove_url(text):
    url_regex = r'\b(?:(?:http|https):\/\/)?([-a-zA-Z0-9.]{2,256}\.[a-z]{2,4})\b(?:\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?\b|\b(www)([-a-zA-Z0-9.]{2,256}[a-z]{2,4})\b'
    return re.sub(url_regex, '', text)

def text_preprocess(
    text, tokenizer=spacy_NLP('en_core_web_sm').tokenize_API, stopwords=STOP_WORDS
): 
    text = remove_html_elements(text)

    text_abbreviations = set(
        regex_replace(abb, r'\(|\)', '')
        for abb in set(retrieve_abbreviation(text))
    )
    # print('{} abbreviation in text'.format(', '.join(text_abbreviations)), end='\n\n')

    text = ' '.join(
        [
            word.lower() 
            for word in regex_replace(text, regex_abbreviation).split()
            if not word in text_abbreviations
        ]
    )

    text = regex_replace(text, regex_special_char_digit, ' ')

    if tokenizer:
        text = " ".join(
            [
                token
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
    
    return regex_replace(regex_replace(text, regex_single_char, ''), regex_white_spaces, ' ')
