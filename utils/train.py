# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys

import tensorflow as tf

sys.path.append('../..')

def tokenize(lang, maxlen):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    seq = lang_tokenizer.texts_to_sequences(lang)
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=maxlen, padding='post')
    lang_word2id = lang_tokenizer.word_index
    return seq, lang_word2id


if __name__ == "__main__":
   pass