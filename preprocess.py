#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import jieba
from codecs import open
from jieba import posseg
import sys

reload(sys)
sys.setdefaultencoding('utf8')

REMOVE_WORDS = ['|', '[', ']', '语音', '图片']

def parseData(path):
    df = pd.read_csv(path,encoding='utf-8')
    #print df
    #print type(df)
    #print df.dtypes
    dataX = df.Question.str.cat(df.Dialogue)
    dataY = []

    if 'Report' in df.columns:
        dataY = df.Report
    return dataX,dataY

def readStopWords(path):
    lines = set()
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.encode("utf-8")
            line = line.strip()
            lines.add(line)
    return lines

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

def removeWord(wordsList):
    wordsList = [word for word in wordsList if word not in REMOVE_WORDS]
    return wordsList

def removeWord2(segList,stopWords):
    segWords = []
    for j in segList:
        if j in stopWords:
            continue
        segWords.append(j)
    return segWords

def saveData(data1,data2,data3,path1,path2,path3,stopWordPath):
    stopWords = readStopWords(stopWordPath)
    with open(path1,'w',encoding ='utf-8') as f1:
        count = 0
        for line in data1:
            line = str(line).encode("utf-8")
            if isinstance(line,str):
                segList = segment(line.strip(),cutType='word')
                segList = removeWord(segList)
                segList = removeWord2(segList,stopWords)
                segLine = ' '.join(segList)
                segLine = segLine.encode("utf-8")
                f1.write('%s' % segLine)
            count += 1
            f1.write('\n')

    with open(path2, 'w',encoding ='utf-8') as f2:
        for line in data2:
            line = str(line).encode("utf-8")
            if isinstance(line,str):
                segList = segment(line.strip(),cutType='word')
                segList = removeWord(segList)
                segList = removeWord2(segList,stopWords)
                segLine = ' '.join(segList)
                segLine = segLine.encode("utf-8")
                f2.write('%s' % segLine)
            f2.write('\n')

    with open(path3, 'w',encoding ='utf-8') as f3:
        for line in data3:
            line = str(line).encode("utf-8")
            if isinstance(line,str):
                segList = segment(line.strip(),cutType='word')
                segList = removeWord(segList)
                segList = removeWord2(segList,stopWords)
                segLine = ' '.join(segList)
                segLine = segLine.encode("utf-8")
                f3.write('%s' % segLine)
            f3.write('\n')


if __name__ == '__main__':
    trainListSrc, trainlListTag = parseData('./datasets/AutoMaster_TrainSet.csv')
    testListSrc,_ = parseData('./datasets/AutoMaster_TestSet.csv')

    saveData(trainListSrc, trainlListTag,testListSrc,'./datasets/train_set.seg_x.txt','./datasets/train_set.seg_y.txt','./datasets/test_set.seg_x.txt',stopWordPath='./datasets/stopwords.txt')