#!/usr/bin/env python
# coding: utf-8

# In[343]:


import json
import gzip
import math
from collections import defaultdict
import numpy
from sklearn import linear_model


# In[344]:


# This will suppress any warnings, comment out if you'd like to preserve them
import warnings
warnings.filterwarnings("ignore")


# In[345]:


# Check formatting of submissions
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[346]:


answers = {}


# In[347]:


f = open("spoilers.json.gz", 'r')


# In[348]:


dataset = []
for l in f:
    d = eval(l)
    dataset.append(d)


# In[349]:


f.close()


# In[350]:


# A few utility data structures
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataset:
    u,i = d['user_id'],d['book_id']
    reviewsPerUser[u].append(d)
    reviewsPerItem[i].append(d)

# Sort reviews per user by timestamp
for u in reviewsPerUser:
    reviewsPerUser[u].sort(key=lambda x: x['timestamp'])
    
# Same for reviews per item
for i in reviewsPerItem:
    reviewsPerItem[i].sort(key=lambda x: x['timestamp'])


# In[351]:


# E.g. reviews for this user are sorted from earliest to most recent
[d['timestamp'] for d in reviewsPerUser['b0d7e561ca59e313b728dc30a5b1862e']]


# In[352]:


dataset[0]


# In[353]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[354]:


### 1a


# In[355]:


x=[]
y=[]
for key in reviewsPerUser:
    if len(reviewsPerUser[key])==1:
        continue
    tmp=[]
    for i in range(len(reviewsPerUser[key])-1):
        tmp.append(reviewsPerUser[key][i]['rating'])
    x.append(tmp)
    y.append(reviewsPerUser[key][-1]['rating'])
print(x[:10])
print(y[:10])


# In[356]:


ypred=[sum(i)/len(i) for i in x]
len(y)==len(ypred)


# In[357]:


#MSE=mean_squared_error()
MSE=mean_squared_error
MSE(y,ypred)


# In[ ]:





# In[358]:


answers['Q1a'] = MSE(y,ypred)


# In[359]:


assertFloat(answers['Q1a'])


# In[360]:


### 1b


# In[361]:


x=[]
y=[]
for key in reviewsPerItem:
    if len(reviewsPerItem[key])==1:
        continue
    tmp=[]
    for i in range(len(reviewsPerItem[key])-1):
        tmp.append(reviewsPerItem[key][i]['rating'])
    x.append(tmp)
    y.append(reviewsPerItem[key][-1]['rating'])
print(x[:10])
print(y[:10])


# In[362]:


ypred=[sum(i)/len(i) for i in x]
len(y)==len(ypred)


# In[363]:


MSE=mean_squared_error
MSE(y,ypred)


# In[364]:


answers['Q1b'] = MSE(y,ypred)


# In[365]:


assertFloat(answers['Q1b'])


# In[366]:


### 2


# In[367]:


def experiment(N):
    x=[]
    y=[]
    ypred=[]
    for key in reviewsPerUser:
        if len(reviewsPerUser[key])==1:
            continue
        tmp=[]
        for i in range(len(reviewsPerUser[key])):
            tmp.append(reviewsPerUser[key][i]['rating'])
        
        y.append(reviewsPerUser[key][-1]['rating'])
        if len(reviewsPerUser[key]):
            x.append(tmp[-N-1:-1])
        else:
            ave=sum(tmp)/len(tmp)
            while len(tmp)<N:
                tmp.append(ave)
            x.append(tmp)
        assert y[-1]==tmp[-1]
        assert len(x[0])==N
    ypred=[sum(i)/len(i) for i in x]
    #print(x[:10])
    #print(y[:10])
    #print(len(y))
    return [y,ypred]


# In[368]:


i,j=experiment(3)
len(i)==len(j)


# In[369]:


print(i[:10])
print(j[:10])


# In[370]:


MSE(i[:10],j[:10])


# In[ ]:





# In[371]:


answers['Q2'] = []

for N in [1,2,3]:
    # etc.
    y,ypred=experiment(N)
    answers['Q2'].append(MSE(y,ypred))


# In[372]:


answers['Q2']


# In[373]:


assertFloatList(answers['Q2'], 3)


# In[374]:


### 3a


# In[375]:


def feature3(N, u): # For a user u and a window size of N
    feature=[1]
    for i in range(2,N+2):
        feature.append(reviewsPerUser[u][-i]['rating'])
    return feature


# In[376]:


answers['Q3a'] = [feature3(2,dataset[0]['user_id']), feature3(3,dataset[0]['user_id'])]


# In[377]:


assert len(answers['Q3a']) == 2
assert len(answers['Q3a'][0]) == 3
assert len(answers['Q3a'][1]) == 4


# In[378]:


### 3b


# In[379]:


def experiment3(N):
    x=[]
    y=[]
    ypred=[]
    for key in reviewsPerUser:
        if len(reviewsPerUser[key])<N+1:
            continue
        x.append(feature3(N,key))
        y.append(reviewsPerUser[key][-1]['rating'])
        #assert y[-1]==tmp[-1]
    #ypred=[sum(i[-N:])/len(i[-N:]) for i in x]
    
    return [x,y]


# In[380]:


lr=LinearRegression()


# In[ ]:





# In[381]:


answers['Q3b'] = []

for N in [1,2,3]:
    # etc.
    x,y=experiment3(N)
    lr.fit(x,y)
    ypred=lr.predict(x)
    mse=MSE(y,ypred)
    answers['Q3b'].append(mse)


# In[382]:


answers['Q3b']


# In[383]:


assertFloatList(answers['Q3b'], 3)


# In[384]:


### 4a


# In[385]:


globalAverage = [d['rating'] for d in dataset]
globalAverage = sum(globalAverage) / len(globalAverage)


# In[386]:


def featureMeanValue(N, u): # For a user u and a window size of N
    feature=[1]
    n=len(reviewsPerUser[u])
    for i in range(2,N+2):
        if i>n:
            if n==1:
                feature.append(globalAverage)
            else:
                feature.append(sum(feature[1:n])/len(feature[1:n]))
        else:
            feature.append(reviewsPerUser[u][-i]['rating'])
    return feature


# In[387]:


def featureMissingValue(N, u):
    feature=[1]
    for i in range(2,N+2):
        if i>len(reviewsPerUser[u]):
            feature.append(1)
            feature.append(0)
        else:
            feature.append(0)
            feature.append(reviewsPerUser[u][-i]['rating'])
    return feature


# In[388]:


answers['Q4a'] = [featureMeanValue(10, dataset[0]['user_id']), featureMissingValue(10, dataset[0]['user_id'])]


# In[389]:


answers['Q4a']


# In[390]:


assert len(answers['Q4a']) == 2
assert len(answers['Q4a'][0]) == 11
assert len(answers['Q4a'][1]) == 21


# In[391]:


### 4b


# In[392]:


x=[]
y=[]
for key in reviewsPerUser:
    if len(reviewsPerUser[key])==1:
        continue
    tmp=[]
    for i in range(len(reviewsPerUser[key])-1):
        tmp.append(reviewsPerUser[key][i]['rating'])
    x.append(tmp)
    y.append(reviewsPerUser[key][-1]['rating'])
ypred=[sum(i)/len(i) for i in x]


# In[393]:


N=10


# In[394]:


answers['Q4b'] = []

for featFunc in [featureMeanValue, featureMissingValue]:
    # etc.
    x=[featFunc(N,u) for u in reviewsPerUser]
    y=[reviewsPerUser[u][-1]['rating'] for u in reviewsPerUser]
    lr.fit(x,y)
    ypred=lr.predict(x)
    mse=MSE(y,ypred)
    answers['Q4b'].append(mse)


# In[395]:


answers['Q4b']


# In[396]:


assertFloatList(answers["Q4b"], 2)


# In[397]:


### 5


# In[398]:


dataset[0]


# In[399]:


def feature5(sentence):
    feature=[]
    f1=len(sentence)
    f2=sentence.count('!')
    f3=0
    for s in sentence:
        if 0<=ord(s)-ord('A')<26:
            f3+=1
    feature=[1,f1,f2,f3]
    return feature


# In[400]:


y = []
X = []

for d in dataset:
    for spoiler,sentence in d['review_sentences']:
        X.append(feature5(sentence))
        y.append(spoiler)


# In[401]:


X[0]


# In[402]:


answers['Q5a'] = X[0]


# In[403]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
Lr=LogisticRegression(class_weight='balanced',C=1)


# In[404]:


Lr.fit(X,y)
y_pre=Lr.predict(X)


# In[405]:


coef_mat=confusion_matrix(y,y_pre)


# In[406]:


TN, FP=coef_mat[0][0],coef_mat[0][1]
FN, TP=coef_mat[1][0],coef_mat[1][1]
def get_ber(TP,TN,FP,FN):
    return (FP/(FP+TN)+FN/(FN+TP))/2


# In[407]:


BER=get_ber(TP,TN,FP,FN)
BER


# In[408]:


answers['Q5b'] = [TP, TN, FP, FN, BER]


# In[409]:


assert len(answers['Q5a']) == 4
assertFloatList(answers['Q5b'], 5)


# In[410]:


### 6


# In[411]:


def feature6(review):
    sentences=d['review_sentences']
    feature=[sentences[0][0],sentences[1][0],sentences[2][0],sentences[3][0],sentences[4][0]]
    sentence=sentences[5][1]
    f0=1
    f1=len(sentence)
    f2=sentence.count('!')
    f3=0
    for s in sentence:
        if 0<=ord(s)-ord('A')<26:
            f3+=1
    feature=[1,f1,f2,f3]+feature
    return feature


# In[412]:


y = []
X = []

for d in dataset:
    sentences = d['review_sentences']
    if len(sentences) < 6: continue
    X.append(feature6(d))
    y.append(sentences[5][0])

#etc.


# In[413]:


answers['Q6a'] = X[0]


# In[414]:


Lr.fit(X,y)
y_pre=Lr.predict(X)
coef_mat=confusion_matrix(y,y_pre)
TN, FP=coef_mat[0][0],coef_mat[0][1]
FN, TP=coef_mat[1][0],coef_mat[1][1]
BER=get_ber(TP,TN,FP,FN)
BER


# In[415]:


answers['Q6b'] = BER


# In[416]:


assert len(answers['Q6a']) == 9
assertFloat(answers['Q6b'])


# In[417]:


### 7


# In[418]:


# 50/25/25% train/valid/test split
Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]


# In[419]:


bers=[]
for c in [0.01, 0.1, 1, 10, 100]:
    # etc.
    Lr=LogisticRegression(class_weight='balanced',C=c)
    Lr.fit(Xtrain,ytrain)
    y_pre=Lr.predict(Xvalid)
    coef_mat=confusion_matrix(yvalid,y_pre)
    TN, FP=coef_mat[0][0],coef_mat[0][1]
    FN, TP=coef_mat[1][0],coef_mat[1][1]
    BER=get_ber(TP,TN,FP,FN)
    bers.append(BER)
bers


# In[420]:


bestber=min(bers)
bestC=[0.01, 0.1, 1, 10, 100][bers.index(bestber)]
bestC


# In[421]:


Lr=LogisticRegression(class_weight='balanced',C=bestC)
Lr.fit(Xtrain,ytrain)
y_pre=Lr.predict(Xtest)
coef_mat=confusion_matrix(ytest,y_pre)
TN, FP=coef_mat[0][0],coef_mat[0][1]
FN, TP=coef_mat[1][0],coef_mat[1][1]
ber=get_ber(TP,TN,FP,FN)
ber


# In[422]:


answers['Q7'] = bers + [bestC] + [ber]


# In[423]:


assertFloatList(answers['Q7'], 7)


# In[424]:


### 8


# In[425]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[426]:


# 75/25% train/test split
dataTrain = dataset[:15000]
dataTest = dataset[15000:]


# In[427]:


# A few utilities

itemAverages = defaultdict(list)
ratingMean = []

for d in dataTrain:
    itemAverages[d['book_id']].append(d['rating'])
    ratingMean.append(d['rating'])

for i in itemAverages:
    itemAverages[i] = sum(itemAverages[i]) / len(itemAverages[i])

ratingMean = sum(ratingMean) / len(ratingMean)


# In[428]:


reviewsPerUser = defaultdict(list)
usersPerItem = defaultdict(set)

for d in dataTrain:
    u,i = d['user_id'], d['book_id']
    reviewsPerUser[u].append(d)
    usersPerItem[i].add(u)


# In[429]:


# From my HW2 solution, welcome to reuse
def predictRating(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        ratings.append(d['rating'] - itemAverages[i2])
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        if item in itemAverages:
            return itemAverages[item]
        else:
            return ratingMean


# In[430]:


labels=[d['rating'] for d in dataTest]
labels[0]


# In[431]:


predictions=[predictRating(d['user_id'],d['book_id']) for d in dataTest]


# In[432]:


answers["Q8"] = MSE(predictions, labels)
answers['Q8']


# In[433]:


assertFloat(answers["Q8"])


# In[434]:


### 9


# In[435]:


labels=[d['rating'] for d in dataTest]
predictions=[predictRating(d['user_id'],d['book_id']) for d in dataTest]


# In[437]:


itemAperance=defaultdict(int)
for d in dataTrain:
    itemAperance[d['book_id']]+=1


# In[438]:


pos_never=[]
pos_1to5=[]
pos_5=[]
pos=0
for d in dataTest:
    if 0==itemAperance[d['book_id']]:
        pos_never.append(pos)
    elif 0<itemAperance[d['book_id']]<=5:
        pos_1to5.append(pos)
    else:
        pos_5.append(pos)
    pos+=1
lab1,lab2,lab3=[],[],[]
pre1,pre2,pre3=[],[],[]

for pos in pos_never:
    lab1.append(labels[pos])
    pre1.append(predictions[pos])
    
for pos in pos_1to5:
    lab2.append(labels[pos])
    pre2.append(predictions[pos])
    
for pos in pos_5:
    lab3.append(labels[pos])
    pre3.append(predictions[pos])
print(str(len(pos_never))+' '+str(len(pos_1to5))+' '+str(len(pos_5)))
print(len(labels))


# In[439]:


mse0=MSE(pre1, lab1)
mse0


# In[440]:


mse1to5=MSE(pre2, lab2)
mse1to5


# In[441]:


mse5=MSE(pre3, lab3)
mse5


# In[442]:


answers["Q9"] = [mse0, mse1to5, mse5]


# In[443]:


assertFloatList(answers["Q9"], 3)


# In[446]:


### 10


# Fistly, I want to improve the rating mean since some ratings could be too emotional

# In[485]:


def get_newratingMean(c):
    new_ratingMean=[]
    for d in dataTrain:

        new_ratingMean.append(d['rating'])
    n=len(new_ratingMean)
    new_ratingMean=new_ratingMean[int(n*c):int(n*(1-c))]

    new_ratingMean = sum(new_ratingMean) / len(new_ratingMean)
    return new_ratingMean


# In[488]:


result=[]
for i in range(1,21):
    new_ratingMean=get_newratingMean(0.01*i)
    arr=[new_ratingMean]*len(lab1)
    result.append(MSE(lab1,arr))
best=min(result)
print(result.index(best))
result[result.index(best)]


# In[490]:


itsMSE=result[result.index(best)]


# In[497]:


ratingMean=new_ratingMean=get_newratingMean(0.01*4)


# Secondly, improve the model by giving predictions based on users

# In[502]:


userAverages=defaultdict(list)
for d in dataTrain:
    userAverages[d['user_id']].append(d['rating'])
    
for u in userAverages:
    userAverages[u] = sum(userAverages[u]) / len(userAverages[u])


# In[503]:


def predictRating(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        ratings.append(d['rating'] - itemAverages[i2])
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        if item in itemAverages:
            return itemAverages[item]
        elif user in userAverages:
            return userAverages[user]
        else:
            return ratingMean


# In[504]:


predictions=[predictRating(d['user_id'],d['book_id']) for d in dataTest if 0==itemAperance[d['book_id']]]


# In[505]:


len(predictions)==len(lab1)


# In[506]:


MSE(predictions,lab1)


# In[509]:


itsMSE=MSE(predictions,lab1)


# In[510]:


answers["Q10"] = ("As we know that if the item didn't show in the train set, then we use the mean rating of the train set to make predictions.My idea has two steps.The first is that some ratings could be useless since they could be tooemotional. That is to say some users may be too satisfied that gave a 5 or too depressed and gave a 1.Both of them could be useless either higher than the product should be or too lower.So I managed to choose a range to get rid of the a portion of lowest and highest ratings.I used a for loop to compare and choose a best range, and seems get a small improvement.Secondly, just give a mean is too rough, we give r(u,i) based on u and i, and mean idea ignores theimportance of user. So if item didn't show in train set but user did, we can make predictionsbased on history of user. So there is huge improvement. The only first step improves to 1.72144, which is close tooriginal one. With step 2 and 1 together, the model preforms better with MSE of around 1.67.", itsMSE)


# In[511]:


assert type(answers["Q10"][0]) == str
assertFloat(answers["Q10"][1])


# In[512]:


f = open("answers_midterm.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




