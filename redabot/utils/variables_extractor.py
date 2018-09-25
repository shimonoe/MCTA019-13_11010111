# -*- coding:utf-8 -*-

import re
import unicodedata
import pt_core_news_sm as spacy

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from threading import Lock

from scipy.stats import variation
from hyphen import Hyphenator

class Variables(object):

    __slots__ = ['data', 'features', 'nlp', 'doc', '__executor', 'h_br', '__lock']

    def __init__(self, text=None, lock=None):

        self.data = {
                "text": None,
                "sentences": None
                }

        self.features = {
                "four_letter": None,
                "five_letter": None,
                "six_letter": None,
                "seven_letter": None,
                "unique_ratio": None,
                "hapax_ratio": None,
                "unique_ratio_sent": None,
                "hapax_ratio_sent": None,
                "avg_word_len_per_sentence": None,
                "avg_word_len_overall": None,
                "cov_word_length": None,
                "avg_sentence_length": None,
                "cov_sentence_length": None,
                "unique_accentuated_words": None,
                "unique_lemmas_overall": None,
                "entities_per_sentence": None,
                "verb_ratio": None,
                "noun_ratio": None,
                "adjective_ratio": None,
                "adposition_ratio": None,
                "pronoun_ratio": None,
                "determiner_ratio": None,
                "stop_words_ratio": None,
                "person_per_sentence": None,
                "org_per_sentence": None,
                "loc_per_sentence": None,
                "misc_per_sentence": None,
                "digraph_words_ratio": None,
                "passive_voice_ratio": None,
                "flesch_reading_ease_br": None,
                "flesch_kincaid_grade_level": None,
                "pct_of_smog_hard_words": None,
                "bormuth_cloze_mean": None,
                }

        if lock is None:
            self.__lock = Lock()
        else:
            self.__lock = lock

        self.__executor = ThreadPoolExecutor(max_workers=cpu_count()*4)

        self.__normalize_text(text)

        self.__lock.acquire()
        self.nlp = spacy.load()
        self.doc = self.nlp(self.data["text"])
        self.__lock.release()

        self.__text_to_sentences()

        self.h_br = Hyphenator('pt_BR')

        with self.__executor as e:
            self.__main_features_per_sentence(e)
            self.__main_features_overall(e)
            self.__main_features_formulas(e)

    def __getattr__(self, attr=None):
        if attr == "features":
            return self.features
        elif attr:
            return self.features[attr]

    ##-----------------------------------------------------------------------
    # Class methods to calculate text variables
    ##-----------------------------------------------------------------------

    def __normalize_text(self, text=None):
        self.data['text'] = unicodedata.normalize("NFKC", text)

    def __text_to_sentences(self):
        self.data['sentences'] = [sent.string.strip() for sent in self.doc.sents]

    def __sentences_to_words(self, sentence=None):
        return sentence.split()

    def __words_lengths(self, words):
        return [len(w) for w in words]

    def words_with_length(self, words, length, feature_key):
        nLetters = sum(1 for w in words if len(w) >= length)
        self.features[feature_key] = nLetters/len(words)

    def unique_ratio(self, words):
        self.features['unique_ratio'] = len(set(words))/len(words)

    def hapax_ratio(self, words):
        counts = [(w, words.count(w)) for w in set(words)]
        counts_one = [w for w in counts if w[1] == 1]
        self.features['hapax_ratio'] = len(counts_one)/len(counts)

    def unique_ratio_sent(self):
        ratios = []
        for each in self.data['sentences']:
            words = self.__sentences_to_words(each)
            ratios.append(len(set(words))/len(words))
        self.features['unique_ratio_sent'] = sum(ratios)/len(ratios)

    def hapax_ratio_sent(self):
        ratios = []
        for each in self.data['sentences']:
            words = self.__sentences_to_words(each)
            counts = [(w, words.count(w)) for w in set(words)]
            counts_one = [w for w in counts if w[1] == 1]
            ratios.append(len(counts_one)/len(counts))
        self.features['hapax_ratio_sent'] = sum(ratios)/len(ratios)

    def avg_word_length_per_sentence(self):
        lens = []
        for each in self.data['sentences']:
            words = self.__sentences_to_words(each)
            lens = [len(w) for w in words]
            lens.append(sum(a for a in lens)/len(words))
        self.features['avg_word_len_per_sentence'] = sum(lens)/len(lens)

    def avg_word_length_overall(self, words, lens):
        self.features['avg_word_len_overall'] = sum(lens)/len(words)

    def cov_word_length(self, words, lens):
        self.features['cov_word_length'] = variation(lens, axis=0)

    def avg_sentence_length(self):
        lens = [len(s.split()) for s in self.data['sentences']]
        self.features['avg_sentence_length'] = sum(lens)/len(lens)

    def cov_sentence_length(self):
        lens = [len(s.split()) for s in self.data['sentences']]
        self.features['cov_sentence_length'] = variation(lens, axis=0)

    def unique_accentuated_words(self, words):
        regex = r"(\S+|)[àáéíóúãõâêô][a-zA-Z]*"
        matches = [m.group() for m in re.finditer(regex, self.data['text'], re.UNICODE | re.IGNORECASE | re.VERBOSE)]
        self.features['unique_accentuated_words'] = len(set(matches))/len(words)

    def unique_lemmas_overall(self, words):
        lemmas = [token.lemma_ for token in self.doc]
        self.features['unique_lemmas_overall'] = len(set(lemmas))/len(words)

    def entities_per_sentence(self):
        ents_count = sum(1 for ent in self.doc.ents)
        self.features['entities_per_sentence'] = ents_count/len(self.data["sentences"])

    def persons_per_sentence(self):
        persons_count = sum(1 for ent in self.doc.ents if ent.label_ == "PER")
        self.features['person_per_sentence'] = persons_count/len(self.data["sentences"])

    def orgs_per_sentence(self):
        orgs_count = sum(1 for ent in self.doc.ents if ent.label_ == "ORG")
        self.features['org_per_sentence'] = orgs_count/len(self.data["sentences"])

    def locs_per_sentence(self):
        locs_count = sum(1 for ent in self.doc.ents if ent.label_ == "LOC")
        self.features['loc_per_sentence'] = locs_count/len(self.data["sentences"])

    def miscs_per_sentence(self):
        miscs_count = sum(1 for ent in self.doc.ents if ent.label_ == "MISC")
        self.features['misc_per_sentence'] = miscs_count/len(self.data["sentences"])

    def verbs_ratio(self, words):
        verbs_count = sum(1 for token in self.doc if token.pos_ == "VERB")
        self.features['verb_ratio'] = verbs_count/len(words)

    def nouns_ratio(self, words):
        nouns_count = sum(1 for token in self.doc if token.pos_ == "NOUN")
        self.features['noun_ratio'] = nouns_count/len(words)

    def adjectives_ratio(self, words):
        adjs_count = sum(1 for token in self.doc if token.pos_ == "ADJ")
        self.features['adjective_ratio'] = adjs_count/len(words)

    def adposition_ratio(self, words):
        adps_count = sum(1 for token in self.doc if token.pos_ == "ADP")
        self.features['adposition_ratio'] = adps_count/len(words)

    def pronoun_ratio(self, words):
        prons_count = sum(1 for token in self.doc if token.pos_ == "PRON")
        self.features['pronoun_ratio'] = prons_count/len(words)

    def determiner_ratio(self, words):
        dets_count = sum(1 for token in self.doc if token.pos_ == "DET")
        self.features['determiner_ratio'] = dets_count/len(words)

    def stop_words_ratio(self, words):
        sw_count = sum(sw.is_stop for sw in self.doc)
        self.features['stop_words_ratio'] = sw_count/len(words)

    def digraph_words_ratio(self, words):
        regex = r"(\w+)(ch|lh|nh|qu|gu|rr|ss|sc|sç|xc|xs)(\w+)"
        matches = [m.group() for m in re.finditer(regex, self.data['text'], re.UNICODE | re.IGNORECASE | re.VERBOSE)]
        self.features['digraph_words_ratio'] = len(matches)/len(words)

    def passive_voice_ratio(self, words):
        # Spacy token's attribute "dep" is a syntactic dependency relation.
        #   - Is passive when the lefties of ROOT in sentence are "aux:pass" or "nsubj"
        count = sum([sum(1 for d in t.lefts if "aux:pass" in d.dep_ or "nsubj" == d.dep_) for t in self.doc])
        self.features['passive_voice_ratio'] = count/len(words)

    def flesch_reading_ease_and_grade_br(self, words, words_syllables_count):
        # Flesch read easy redability formula for Portuguese:
        # http://www.nilc.icmc.usp.br/nilc/download/Reltec28.pdf
        ASL = len(words)/len(self.data['sentences'])
        ASW = sum(words_syllables_count)/len(words)
        self.features['flesch_reading_ease_br'] = 184.835 - (1.015*ASL) - (84.6*ASW)
        #self.features['flesch_kincaid_grade_level'] = -15.59 + (0.39*ASL) + (11.8*ASW) #Original
        self.features['flesch_kincaid_grade_level'] = -1*(-15.59 + (-7.120071171168128*ASL) + (4.7018780114723135*ASW))/10\
                                                      -(-0.12723741405046063*ASL + 0.629826609095385*ASW)

    def pct_of_smog_hard_words(self, words_syllables_count):
        self.features['pct_of_smog_hard_words'] = sum(1 for w in words_syllables_count if w > 2)/len(words_syllables_count)

    def bormuth_cloze_mean(self):
        # source: http://www.ibrarian.net/navon/paper/Rea_dabili_ty_Fo_rmulas.pdf?paperid=1136157
        while self.features['avg_word_len_overall'] is None: pass
        ACW = self.features['avg_word_len_overall']
        while self.features['stop_words_ratio'] is None: pass
        SWPS = self.features['stop_words_ratio']
        while self.features['avg_sentence_length'] is None: pass
        AWPS = self.features['avg_sentence_length']
        self.features['bormuth_cloze_mean'] = 0.886593 - (0.03640 * ACW)\
                                                       + (0.161911 * SWPS)\
                                                       - (0.21401 * AWPS)\
                                                       - (0.000577 * AWPS ** 2)\
                                                       - (0.000005 * AWPS ** 3)

    ##-----------------------------------------------------------------------
    # Main class methods to extract variables based on case:
    #   - Per sentence features;
    #   - Overall features;
    ##-----------------------------------------------------------------------

    def __main_features_per_sentence(self, e):
        e.submit(self.unique_ratio_sent())
        e.submit(self.hapax_ratio_sent())
        e.submit(self.avg_word_length_per_sentence())
        e.submit(self.avg_sentence_length())
        e.submit(self.cov_sentence_length())
        e.submit(self.entities_per_sentence())
        e.submit(self.persons_per_sentence())
        e.submit(self.orgs_per_sentence())
        e.submit(self.locs_per_sentence())
        e.submit(self.miscs_per_sentence())

    def __main_features_overall(self, e):
        words = self.__sentences_to_words(self.data['text'])
        words_lens = self.__words_lengths(words)
        words_syllables_count = [len(self.h_br.syllables(w)) for w in words]

        e.submit(self.words_with_length(words, 4, 'four_letter'))
        e.submit(self.words_with_length(words, 5, 'five_letter'))
        e.submit(self.words_with_length(words, 6, 'six_letter'))
        e.submit(self.words_with_length(words, 7, 'seven_letter'))
        e.submit(self.unique_ratio(words))
        e.submit(self.hapax_ratio(words))
        e.submit(self.avg_word_length_overall(words, words_lens))
        e.submit(self.cov_word_length(words, words_lens))
        e.submit(self.unique_accentuated_words(words))
        e.submit(self.unique_lemmas_overall(words))
        e.submit(self.verbs_ratio(words))
        e.submit(self.nouns_ratio(words))
        e.submit(self.adjectives_ratio(words))
        e.submit(self.adposition_ratio(words))
        e.submit(self.pronoun_ratio(words))
        e.submit(self.determiner_ratio(words))
        e.submit(self.stop_words_ratio(words))
        e.submit(self.digraph_words_ratio(words))
        e.submit(self.passive_voice_ratio(words))

    def __main_features_formulas(self, e):
        words = self.__sentences_to_words(self.data['text'])
        words_lens = self.__words_lengths(words)
        words_syllables_count = [len(self.h_br.syllables(w)) for w in words]

        e.submit(self.flesch_reading_ease_and_grade_br(words, words_syllables_count))
        e.submit(self.pct_of_smog_hard_words(words_syllables_count))
        e.submit(self.bormuth_cloze_mean())


if __name__ == "__main__":

    from pprint import pprint

    text = open("./examples/example6.txt", 'r').read()

    var = Variables(text)
    pprint(var.features)
    print("\nNúmero de variáveis: ", len(var.features))
