#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
import urllib
import scipy.optimize
import random
from sklearn import linear_model
import gzip
from collections import defaultdict


# In[2]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[4]:


f = open("5year.arff", 'r')


# In[5]:


# Read and parse the data
while not '@data' in f.readline():
    pass

dataset = []
for l in f:
    if '?' in l: # Missing entry
        continue
    l = l.split(',')
    values = [1] + [float(x) for x in l]
    values[-1] = values[-1] > 0 # Convert to bool
    dataset.append(values)


# In[6]:


X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]


# In[7]:


answers = {} # Your answers


# In[8]:


def accuracy(predictions, y):
    cnt,n=0,len(y)
    for i in range(n):
        cnt+=predictions[i]==y[i]
    return cnt/n


# In[9]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[11]:


def BER(predictions, y):
    coef_mat=confusion_matrix(y,predictions)
    TN, FP=coef_mat[0][0],coef_mat[0][1]
    FN, TP=coef_mat[1][0],coef_mat[1][1]
    return (FP/(FP+TN)+FN/(FN+TP))/2


# In[12]:


### Question 1


# In[13]:


mod = linear_model.LogisticRegression(C=1)
mod.fit(X,y)

pred = mod.predict(X)


# In[14]:


acc1=accuracy(pred,y)
acc1


# In[15]:


ber1=BER(pred,y)
ber1


# In[16]:


answers['Q1'] = [acc1, ber1] # Accuracy and balanced error rate


# In[17]:


assertFloatList(answers['Q1'], 2)


# In[18]:


### Question 2


# In[19]:


mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(X,y)

pred = mod.predict(X)


# In[20]:


acc2=accuracy(pred,y)
acc2


# In[21]:


ber2=BER(pred,y)
ber2


# In[22]:


answers['Q2'] = [acc2, ber2]


# In[23]:


assertFloatList(answers['Q2'], 2)


# In[24]:


### Question 3


# In[25]:


random.seed(3)
random.shuffle(dataset)


# In[26]:


X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]


# In[27]:


Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]


# In[28]:


len(Xtrain), len(Xvalid), len(Xtest)


# In[38]:


mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(Xtrain,ytrain)

predTrain = mod.predict(Xtrain)
predValid = mod.predict(Xvalid)
predTest  = mod.predict(Xtest)


# In[ ]:





# In[39]:


berTrain=BER(predTrain,ytrain)
berTrain


# In[40]:


berValid=BER(predValid,yvalid)
berValid


# In[41]:


berTest=BER(predTest,ytest)
berTest


# In[42]:


answers['Q3'] = [berTrain, berValid, berTest]


# In[43]:


assertFloatList(answers['Q3'], 3)


# In[37]:


### Question 4


# In[50]:


C_list=[0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
len(C_list)


# In[51]:


berList=[]
for i in C_list:
    mod = linear_model.LogisticRegression(C=i, class_weight='balanced')
    mod.fit(Xtrain,ytrain)
    predValid = mod.predict(Xvalid)
    berValid=BER(predValid,yvalid)
    berList.append(berValid)


# In[52]:


berList


# In[53]:


answers['Q4'] = berList


# In[54]:


assertFloatList(answers['Q4'], 9)


# In[55]:


### Question 5


# In[56]:


berbest=min(berList)
bestC=C_list[berList.index(berbest)]
bestC


# In[57]:


mod = linear_model.LogisticRegression(C=bestC, class_weight='balanced')
mod.fit(Xtrain,ytrain)
predTest  = mod.predict(Xtest)
ber5=BER(predTest,ytest)
ber5


# In[58]:


answers['Q5'] = [bestC, ber5]


# In[59]:


assertFloatList(answers['Q5'], 2)


# In[63]:


### Question 6


# In[64]:


f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(eval(l))


# In[65]:


dataTrain = dataset[:9000]
dataTest = dataset[9000:]


# In[66]:


dataTrain[0]


# In[109]:


# Some data structures you might want

usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
ratingDict = {} # To retrieve a rating for a specific user/item pair

for d in dataTrain:
    user,item=d['user_id'],d['book_id']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    reviewsPerUser[user].append(d)
    reviewsPerItem[item].append(d)


# In[79]:


def Jaccard(s1, s2):
    nu=len(s1.intersection(s2))
    de=len(s1.union(s2))
    return nu/de


# In[80]:


def mostSimilar(i, N,usersPerItem):
    simList=[]
    users=usersPerItem[i]
    for i2 in usersPerItem:
        if i==i2:continue
        sim=Jaccard(users,usersPerItem[i2])
        simList.append((sim,i2))
    simList.sort(reverse=True)
    return simList[:N]


# In[81]:


answers['Q6'] = mostSimilar('2767052', 10,usersPerItem)
answers['Q6']


# In[82]:


assert len(answers['Q6']) == 10
assertFloatList([x[0] for x in answers['Q6']], 10)


# In[83]:


### Question 7


# In[147]:


mean_rating=0
for d in dataTrain:
    mean_rating+=d['rating']
mean_rating/=len(dataTrain)


# In[154]:


def itemAverages(item):
    return sum(datum['rating'] for datum in reviewsPerItem[item])/len(reviewsPerItem[item])
        
def predictRating(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        ratings.append(d['rating'] - itemAverages(i2))
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages(item) + sum(weightedRatings) / sum(similarities)
    else:
        return mean_rating


# In[155]:


rating_lable=[d['rating'] for d in dataTest]
rating_lable[0]


# In[156]:


rating_predict=[predictRating(d['user_id'],d['book_id']) for d in dataTest]


# In[157]:


from sklearn.metrics import mean_squared_error
mse7=mean_squared_error(rating_lable,rating_predict)
mse7


# In[158]:


answers['Q7'] = mse7


# In[159]:


assertFloat(answers['Q7'])


# In[160]:


### Question 8


# In[161]:


def userAverages(user):
    return sum(datum['rating'] for datum in reviewsPerUser[user])/len(reviewsPerUser[user])
        
def predictRating(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        u = d['user_id']
        if u == user: continue
        ratings.append(d['rating'] - itemAverages(i2))
        similarities.append(Jaccard(userAverages[user],userAverages[u]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages(item) + sum(weightedRatings) / sum(similarities)
    else: 
        return mean_rating


# In[162]:


rating_predict=[predictRating(d['user_id'],d['book_id']) for d in dataTest]


# In[163]:


mse8=mean_squared_error(rating_lable,rating_predict)
mse8


# In[164]:


answers['Q8'] = mse8


# In[165]:


assertFloat(answers['Q8'])


# In[166]:


f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




