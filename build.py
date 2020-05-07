# -*- coding: utf-8 -*-
import sys
from codecs import open
import six
# Define constants associated with the usual special tokens.
PAD_TOKEN = '<pad>'
GO_TOKEN = '<go>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

sys.path.append('../..')
from utils.train import tokenize

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")



def preprocess_sentence(w):
    w = convert_to_unicode(w.lower().strip())
    w = w.rstrip().strip()
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = GO_TOKEN + ' ' + w + ' ' + EOS_TOKEN
    return w

def create_dataset(path, num_examples):
    """
    # 1. Remove the accents
    # 2. Clean the sentences
    # 3. Return word pairs in the format: [ENGLISH, SPANISH]
    :param path:
    :param num_examples:
    :return:
    """
    lines = open(path, 'r', encoding='utf-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]

    #return zip(*word_pairs)
    return word_pairs

def save_word_dict(dict_data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for k, v in dict_data.items():
            f.write("%s\t%d\n" % (k, v))

def build(train_path='',maxlen=400,save_src_vocab_path='',save_trg_vocab_path=''):
    source_texts, target_texts = create_dataset(train_path, 2)
    source_seq, source_word2id = tokenize(source_texts, maxlen)
    target_seq, target_word2id = tokenize(target_texts, maxlen)
    save_word_dict(source_word2id, save_src_vocab_path)
    save_word_dict(target_word2id, save_trg_vocab_path)


if __name__ == '__main__':
    build('./datasets/train_set.seg_x.txt',32,'./datasets/vocab_source.txt','./datasets/vocab_target.txt')