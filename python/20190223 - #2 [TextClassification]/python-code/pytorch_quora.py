import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data
from torchtext import datasets
import string
from torchtext import vocab

import random
import regex as re
import unicodedata
import time

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

# To run experiments deterministically
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

########################################################

# For kaggle
import os

print(os.listdir("../input"))
TRAIN_PATH = '../input/train.csv'
TEST_PATH = '../input/test.csv'
GLOVE_PATH = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
GLOVE = 'glove.840B.300d.txt'

########################################################

# Pretrained embedding to use
EMBEDDING_PATH = GLOVE_PATH
# Should we limit the vocab size?
# (A) 120000, 95000
MAX_SIZE = 120000
# (A) Should we limit number of words in a sentence?
MAX_LEN = 70

# Split ratio for test/valid
SPLIT_RATIO = 0.9

# (A)
BATCH_SIZE = 512

# Model parameters
# (A) Could be lesser I think.
HIDDEN_DIM = 32
# (A)
N_LAYERS = 2
BIDIRECTIONAL = True
# (C)
DROPOUT = 0.5


###########################################################

# TODO implement text_cleaning

def spell_correct(text):
    mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',
                    'counselling': 'counseling',
                    'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
                    'organisation': 'organization',
                    'wwii': 'world war 2',
                    'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary',
                    'Whta': 'What',
                    'narcisist': 'narcissist',
                    'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much',
                    'howmany': 'how many', 'whydo': 'why do',
                    'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does',
                    'mastrubation': 'masturbation',
                    'mastrubate': 'masturbate',
                    "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum',
                    'narcissit': 'narcissist',
                    'bigdata': 'big data',
                    '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend',
                    'airhostess': 'air hostess', "whst": 'what',
                    'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
                    'demonitization': 'demonetization',
                    'demonetisation': 'demonetization'}

    def replace(match):
        return mispell_dict[match.group(0)]

    mispellings_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispellings_re.sub(replace, text)


def normalize_unicode(text):
    return unicodedata.normalize('NFKD', text)


def remove_newlines(text):
    return ' '.join(text.split())


def decontract(text):
    text = re.sub(r"(W|w)on(\'|\’)t", "will not", text)
    text = re.sub(r"(C|c)an(\'|\’)t", "can not", text)
    text = re.sub(r"(Y|y)(\'|\’)all", "you all", text)
    text = re.sub(r"(Y|y)a(\'|\’)ll", "you all", text)

    # general
    text = re.sub(r"(I|i)(\'|\’)m", "i am", text)
    text = re.sub(r"(A|a)in(\'|\’)t", "aint", text)
    text = re.sub(r"n(\'|\’)t", " not", text)
    text = re.sub(r"(\'|\’)re", " are", text)
    text = re.sub(r"(\'|\’)s", " is", text)
    text = re.sub(r"(\'|\’)d", " would", text)
    text = re.sub(r"(\'|\’)ll", " will", text)
    text = re.sub(r"(\'|\’)t", " not", text)
    return re.sub(r"(\'|\’)ve", " have", text)


def space_puncts(text):
    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$',
              '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',
              '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`', '<',
              '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â',
              '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢',
              '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
              '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’',
              '▀', '¨', '▄', '♫', '☆', 'é', '¯',
              '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
              '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³',
              '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']
    for punct in puncts:
        if punct in text:
            text = text.replace(punct, f' {punct} ')
    return text


def remove_puncts(text):
    re_tok = re.compile(f'([{string.punctuation}])')
    return re_tok.sub(' ', text)


def clean_numbers(text):
    if bool(re.search(r'\d', text)):
        text = re.sub('[0-9]{5,}', '#####', text)
        text = re.sub('[0-9]{4}', '####', text)
        text = re.sub('[0-9]{3}', '###', text)
        text = re.sub('[0-9]{2}', '##', text)
    return text


def clean_text(text):
    text = clean_numbers(text)
    text = decontract(text)
    text = remove_puncts(text)
    text = space_puncts(text)
    text = remove_newlines(text)
    text = normalize_unicode(text)
    text = spell_correct(text)
    return text


import torchtext

spacy_tokenizer = torchtext.data.utils.get_tokenizer('spacy')  # just an example


def my_tokenizer(example):
    preprocessed_example = clean_text(example)
    tokenized_example = spacy_tokenizer(preprocessed_example.rstrip('\n'))
    return tokenized_example


###########################################################

# Defining the Fields for our dataset
# Skipping the id column
ID = data.Field()
# Because of a weird setup by torchtext we want to clean in the tokenizer before tokenizing.
# Otherwise preprocessing is applied after tokenization
TEXT = data.Field(tokenize=my_tokenizer)  # TODO fix this shit.
TARGET = data.LabelField(dtype=torch.float)

train_fields = [('id', None), ('text', TEXT), ('target', TARGET)]
test_fields = [('id', ID), ('text', TEXT)]
t0 = time.time()
# Creating our train and test data
train_data = data.TabularDataset(
    path=TRAIN_PATH,
    format='csv',
    skip_header=True,
    fields=train_fields
)

test_data = data.TabularDataset(
    path=TEST_PATH,
    format='csv',
    skip_header=True,
    fields=test_fields
)
print("Time taken: %s" % (t0-time.time()))

# Create validation dataset (default 70:30 split)
train_data, valid_data = train_data.split(split_ratio=SPLIT_RATIO, random_state=random.seed(SEED))

#############################################################################

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of test examples: {len(test_data)}')

#############################################################################

# One training example
vars(train_data.examples[0])

#############################################################################
