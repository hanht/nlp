# -*- coding: utf-8 -*-
import logging
import os

import jieba
from jieba import posseg

jieba.setLogLevel(log_level="ERROR")

def segment(sentence, cutType='word', pos=False):
    if pos:
        if cutType == 'word':
            wordPosSeq = posseg.lcut(sentence)
            wordSeq, posSeq = [], []
            for w, p in wordPosSeq:
                wordSeq.append(w)
                posSeq.append(p)
            return wordSeq, posSeq
        elif cutType == 'char':
            wordSeq = list(sentence)
            posSeq = []
            for w in wordSeq:
                wP = posseg.lcut(w)
                posSeq.append(wP[0].flag)
            return wordSeq,posSeq
    else:
        if cutType == 'word':
            return jieba.lcut(sentence)
        elif cutType == 'char':
            return list(sentence)


class Tokenizer(object):
    def __init__(self, dict_path='', custom_word_freq_dict=None, custom_confusion_dict=None):
        self.model = jieba
        self.model.default_logger.setLevel(logging.ERROR)
        # 初始化大词典
        if os.path.exists(dict_path):
            self.model.set_dictionary(dict_path)
        # 加载用户自定义词典
        if custom_word_freq_dict:
            for w, f in custom_word_freq_dict.items():
                self.model.add_word(w, freq=f)

        # 加载混淆集词典
        if custom_confusion_dict:
            for k, word in custom_confusion_dict.items():
                # 添加到分词器的自定义词典中
                self.model.add_word(k)
                self.model.add_word(word)

    def tokenize(self, sentence):
        """
        切词并返回切词位置
        :param sentence: query
        :return: (w, start, start + width) model='default'
        """
        return list(self.model.tokenize(sentence))