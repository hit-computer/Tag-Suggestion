#-*- coding:utf-8 -*-
import cPickle
import numpy
import collections

with open('vec100.txt','r') as fvec, open('testdata.txt','r') as ftest, open('trainingdata.txt','r') as ftrain, open('MT_WordEmb.pkl','w') as fwe, open('ttrain.dialogues.pkl','w') as ftdia, open('tvalid.dialogues.pkl','w') as ftval, open('ttrain.dict.pkl','w') as fdic:
    category = cPickle.load(open('category.pkl','r'))
    wordvec = []
    wordLS = []
    New_voc = []#词表
    New_v = [] #词向量
    fvec.readline()
    
    num_of_word = 50000
    for line in fvec:
        line = line.split()
        wordLS.append(line[0])
        wordvec.append([float(i) for i in line[1:]])
    New_voc.append('<unk>')
    New_v.append(wordvec[0])
    
    idx = wordLS.index('</ss>')
    New_voc.append('</ss>')
    New_v.append(wordvec[idx])
    wordLS.pop(idx)
    wordvec.pop(idx)
    
    idx = wordLS.index('</d>')
    New_voc.append('</d>')
    New_v.append(wordvec[idx])
    wordLS.pop(idx)
    wordvec.pop(idx)
    #WE = [1,0]
    #WE[0] = numpy.array(wordvec)
    #WE[1] = numpy.ones(WE[0].shape)
    #cPickle.dump(WE, fwe, protocol=cPickle.HIGHEST_PROTOCOL)
    
    """
    allwd = []
    for line in fall:
        line = line.split()
        idx = line.index('</d>')
        allwd += line[:idx]
    allwd = set(allwd)
    for wd in allwd:
        if wd in wordLS:
            idx = wordLS.index(wd)
            New_voc.append(wd)
            New_v.append(wordvec[idx])
            wordLS.pop(idx)
            wordvec.pop(idx)
        else:
            #print wd
            pass
    """
    nn = len(New_voc)
    if nn > num_of_word:
        New_voc = New_voc[:num_of_word]
        New_v = New_v[:num_of_word]
    else:
        New_voc += wordLS[1:num_of_word-nn]
        New_v += wordvec[1:num_of_word-nn]
    
    wordvec = New_v
    wordLS = New_voc
    WE = [1,0]
    WE[0] = numpy.array(wordvec)
    WE[1] = numpy.ones(WE[0].shape)
    cPickle.dump(WE, fwe, protocol=cPickle.HIGHEST_PROTOCOL)
    
    wordDic = {}
    for id,wd in enumerate(wordLS):
        wordDic[wd] = id
    Tdial = []
    freqs = collections.defaultdict(lambda: 0)
    df = collections.defaultdict(lambda: 0)
    max_len = 0
    for line in ftrain:
        #line = line.decode('utf-8')
        #print line
        sent, label = line.split('</d>')
        sentLS = sent.split('</ss>')
        #print sentLS
        new_sentLS = []
        for item in sentLS:
            if len(item.split()) > 3:
                new_sentLS.append(item)
        #print new_sentLS
        new_sent = '</ss>' + '</ss>'.join(new_sentLS) + '</ss>'
        line = new_sent + ' </d>' + label
        #print line
        #break
        line = line.split()
        idx = line.index('</d>')
        tmp = []
        for it in line[:idx+1]:
            #print it
            freqs[it] += 1
            if it in wordDic:
                tmp.append(wordDic[it])
            else:
                tmp.append(0)
        #__N = 0
        for it in line[idx+1:-1]:
            tmp.append(category.index(it))
        tmp.append(1)
        max_len = max(max_len,len(tmp))
        Tdial.append(tmp)
        for sit in set(line):
            df[sit] += 1
    cPickle.dump(Tdial, ftdia, protocol=cPickle.HIGHEST_PROTOCOL)
    Tdial = []
    for line in ftest:
        #line = line.decode('utf-8')
        #print line
        sent, label = line.split('</d>')
        sentLS = sent.split('</ss>')
        #print sentLS
        new_sentLS = []
        for item in sentLS:
            if len(item.split()) > 3:
                new_sentLS.append(item)
        #print new_sentLS
        new_sent = '</ss>' + '</ss>'.join(new_sentLS) + '</ss>'
        line = new_sent + ' </d>' + label
        #print line
        line = line.split()
        idx = line.index('</d>')
        tmp = []
        for it in line[:idx+1]:
            #print it
            if it in wordDic:
                tmp.append(wordDic[it])
            else:
                tmp.append(0)
        for it in line[idx+1:-1]:
            tmp.append(category.index(it))
        tmp.append(1)
        max_len = max(max_len,len(tmp))
        Tdial.append(tmp)
    cPickle.dump(Tdial, ftval, protocol=cPickle.HIGHEST_PROTOCOL)
    #print wordDic
    del wordDic['</ss>']
    diction = [(word, word_id, freqs[word], df[word]) for word, word_id in wordDic.items()]
    diction.append(('</s>',1,diction[1][2],diction[1][3]))
    print len(diction)
    #print diction[:100]
    cPickle.dump(diction, fdic, protocol=cPickle.HIGHEST_PROTOCOL)
    print max_len
    #print WE
    #print Tdial[1]
