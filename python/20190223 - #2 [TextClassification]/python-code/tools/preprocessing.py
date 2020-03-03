import spacy
import unicodedata
import regex as re


class PreProcessor(object):
    def __init__(self, text):
        self.text = text
        self.puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$',
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
        # TODO this varies depending on what task!
        self.mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',
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
        self.mispellings_re = re.compile('(%s)' % '|'.join(self.mispell_dict.keys()))

    def get_text(self):
        return self.text

    # TODO fix misspellings
    def replace_typical_misspell(self):
        def replace(match):
            return self.mispell_dict[match.group(0)]

        self.text = self.mispellings_re.sub(replace, self.text)

        return self

    def spacy_tokenize_words(self):
        raise NotImplementedError

    def normalize_unicode(self):
        self.text = unicodedata.normalize('NFKD', self.text)
        return self

    def remove_newline(self):
        """
        remove \n and  \t
        """
        self.text = ' '.join(self.text.split())
        return self

    def decontracted(self):
        # specific
        text = re.sub(r"(W|w)on(\'|\’)t", "will not", self.text)
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
        self.text = re.sub(r"(\'|\’)ve", " have", text)

        return self

    def space_punctuation(self):
        for punct in self.puncts:
            if punct in self.text:
                self.text = self.text.replace(punct, f' {punct} ')

                # We could also remove all non p\{L}...

        return self

    def remove_punctuation(self):
        import string
        re_tok = re.compile(f'([{string.punctuation}])')
        self.text = re_tok.sub(' ', self.text)

        return self

    def clean_numbers(self):
        text = self.text
        if bool(re.search(r'\d', text)):
            text = re.sub('[0-9]{5,}', '#####', text)
            text = re.sub('[0-9]{4}', '####', text)
            text = re.sub('[0-9]{3}', '###', text)
            text = re.sub('[0-9]{2}', '##', text)
        self.text = text
        return self

    def clean_and_get_text(self):
        self.clean_numbers() \
            .space_punctuation() \
            .decontracted() \
            .normalize_unicode() \
            .remove_newline() \
            .replace_typical_misspell()

        return self.text


# TODO add this at the spacy tokenize. Might help? :)
# nlp = spacy.load('en_core_web_sm')
# doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')
# for token in doc:
#    print(token.text)

# special_case = [{ORTH: u'gim', LEMMA: u'give', POS: u'VERB'}, {ORTH: u'me'}]
# nlp.tokenizer.add_special_case(u'gimme', special_case)

# nlp = spacy.load('en_core_web_sm')
# doc = nlp(u"This is a sentence. This is another sentence.")
# for sent in doc.sents:
#     print(sent.text)
