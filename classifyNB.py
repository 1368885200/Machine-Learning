from numpy import *

def loaddataset():
    postinglist=[['my','dog','has','flea','problems','help','please']]
    classvec=[0]
    return postinglist,classvec

def createvocablist(dataset):
    vocabset=set([])
    for document in dataset:
        vocabset=vocabset|set(document)
    return list(vocabset)

def setofwords2vec(vocablist,inputset):
    returnvec=[0]*len(vocablist)
    for word in inputset:
        if word in vocablist:
            returnvec[vocablist.index(word)]=1
        else:print "the word: %s is not in my vocabulary!" % word
    return returnvec

def trainNBO(trainmatrix,traincategory):
    numtraindocs = len(trainmatrix)
    numwords = len(trainmatrix[0])
    pabusive = sum(traincategory)/float(numtraindocs)
    p0num = ones(numwords);p1num=ones(numwords)
    p0denom=2.0;p1denom=2.0
    for i in range(numtraindocs):
        if traincategory[i]==1:
            p1num+=trainmatrix[i]
            p1denom+=sum(trainmatrix[i])
        else:
            p0num+=trainmatrix[i]
            p0denom+=sum(trainmatrix[i])
    p1vect=log(p1num/p1denom)
    p0vect=log(p0num/p0denom)
    return p0vect,p1vect,pabusive

def classifyNB(vec2classify,p0vec,p1vec,pclass1):
    p1=sum(vec2classify*p1vec)+log(pclass1)
    p0=sum(vec2classify*p0vec)+log(1.0-pclass1)
    if p1>p0:
        return 1
    else :
        return 0

def testingNB():
    listoposts,listclasses=loaddataset()
    myvocablist=createvocablist(listoposts)
    trainmat=[]
    for postindoc in listoposts:
        trainmat.append(setofwords2vec(myvocablist,postindoc))
    p0v,p1v,pab=trainNBO(array(trainmat),array(listclasses))
    testentry=['love','my','dalmation']
    thisdoc=array(setofwords2vec(myvocablist,testentry))
    print testentry,'classified as:',classifyNB(thisdoc,p0v,p1v,pab)
    testentry=['stupid','garbage']
    thisdoc=array(setofwords2vec(myvocablist,testentry))
    print testentry,'classified as:',classifyNB(thisdoc,p0v,p1v,pab)

testingNB()






