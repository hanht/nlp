# -*- coding: utf-8 -*-

import pandas as pd

import jieba
from jieba import posseg

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
    with open(path,mode='r') as f:
        for line in f :
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

def saveData(data1,data2,data3,path1,path2,path3,stopWordPath):
    stopWords = readStopWords(stopWordPath)
    with open(path1,'w') as f1:
        count = 0
        for line in data1:
            if isinstance(line,str):
                segList = segment(line.strip(),cut_type='word')
                segList = removeWord(segList)
                segLine = ' '.join(segList)
                f1.write('%s' % segLine)
            count += 1
            f1.write('\n')

    with open(path2, 'w') as f2:
        for line in data2:
            if isinstance(line,str):
                segList = segment(line.strip(),cut_type='word')
                segList = removeWord(segList)
                segLine = ' '.join(segList)
                f2.write('%s' % segLine)
            f2.write('\n')

    with open(path3, 'w') as f3:
        for line in data3:
            if isinstance(line,str):
                segList = segment(line.strip(),cut_type='word')
                segList = removeWord(segList)
                segLine = ' '.join(segList)
                f3.write('%s' % segLine)
            f3.write('\n')


if __name__ == '__main__' :
    trainListSrc, trainlListTag = parseData('./datasets/AutoMaster_TestSet.csv')
    testListSrc,_ = parseData('./datasets/AutoMaster_TrainSet.csv')

    saveData(trainListSrc, trainlListTag,testListSrc,'./datasets/train_set.seg_x.txt','./datasets/train_set.seg_y.txt','./datasets/test_set.seg_x.txt',stopWordPath='./datasets/stopwords.txt')