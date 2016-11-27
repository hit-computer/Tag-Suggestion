#coding:utf-8
import cPickle
import numpy
import collections
from sklearn.cluster import *

category = []
NofC = 5000
with open('alldata.txt', 'r') as fr,open('vec100.txt','r') as fvec:
    for line in fr:
        word = line.split()
        idx = word.index('</d>')
        for lb in word[idx+1:-1]:
            category.append(lb)
    
    C = collections.Counter(category)
    category = set(category)
    print len(category)
    
    wordvec = []
    wordLS = []
    fvec.readline()
    #num_of_word = 50000
    for line in fvec:
        line = line.split()
        wordLS.append(line[0])
        wordvec.append([float(i) for i in line[1:]])
        
    WD_Matrix = []
    cateLS = []
    for wd in category:
        if wd not in wordLS:
            continue
        cateLS.append(wd)
        idx = wordLS.index(wd)
        vector = wordvec[idx]
        WD_Matrix.append(vector)

    WD_Matrix = numpy.array(WD_Matrix)

    km = KMeans(n_clusters = NofC).fit(WD_Matrix)
    labels = km.labels_
    cents = km.cluster_centers_
    
    print 'cluster complte...'
    
    represent = []
    order_word_list = [i[0] for i in C.most_common()]
    for i in range(NofC):
        sore = []
        cent_v = cents[i]
        for j in range(len(cateLS)):
            wd = cateLS[j]
            if labels[j] == i:
                vector = WD_Matrix[j,:]
                s = numpy.sqrt(numpy.sum((vector-cent_v)**2))
                sore.append((wd,s))
        keywords = sorted(sore,key=lambda x:x[1])[:5]
        keywords = [t[0] for t in keywords]
        for kw in order_word_list:
            if kw in keywords:
                represent.append(kw)
                print i,kw
                break
    
    cPickle.dump(represent, open('category.pkl','w'), protocol=cPickle.HIGHEST_PROTOCOL)
    #print ' '.join(represent)
    """
    for lb,wd in zip(labels,category):
		if not W_CL_dic.has_key(lb):
			W_CL_dic[lb] = []
		idx = wdLS.index(wd)
		vector = Vwd[idx]
		cent_v = cents[lb]
		sore = numpy.sqrt(numpy.sum((vector-cent_v)**2))
		W_CL_dic[lb].append((wd,sore))
    """