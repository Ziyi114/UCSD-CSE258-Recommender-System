#!/usr/bin/env python
# coding: utf-8

# In[264]:


import json
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn import linear_model
import numpy
import random
import gzip
import math


# In[265]:


def assertFloat(x): # Checks that an answer is a float
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[266]:


f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))


# In[267]:


len(dataset)


# In[561]:


answers = {"Q1":[3.688533040832031, 0.07109019019954244, 1.5231747404538287],
           "Q2":[3.7175128077972013, 0.07527591733232629, -4.121506529487976e-05, 1.5214029246165832],
           "Q3":[1.5231747404538287, 1.5046686106250917, 1.496684551517923, 1.490447730223069, 1.4896106953961648],
           "Q4":[1.5248743859866292,1.4977199259322445,1.4856632190311088,1.4767337440080983,1.4809577272893133],
           "Q5":0.907,
           "Q6":[0, 20095, 0, 308, 0.5],
           "Q7":[88, 16332, 3763, 220, 0.4507731134255145],
           "Q8":[0.0, 0.0, 0.03, 0.033, 0.0308]} # Put your answers to each question in this dictionary


# In[268]:


dataset[0]


# In[287]:


### Question 1


# In[288]:


def feature(datum):
    # your implementation
    feat = datum['review_text'].count('!')
    return [feat]


# In[289]:


X = [feature(data) for data in dataset]
Y = [data['rating'] for data in dataset]


# In[290]:


print(X[:40])


# In[291]:


print(Y[0:40])


# In[292]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[293]:


lr.fit(X,Y)


# In[294]:


lr.coef_


# In[295]:


lr.intercept_


# In[300]:


[theta0,theta1]=lr.intercept_,lr.coef_[0]


# In[301]:


Y_pred=lr.predict(X)
Y_pred[:10]


# In[302]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
mse


# In[303]:


answers['Q1'] = [theta0, theta1, mse]
answers['Q1']


# In[304]:


assertFloatList(answers['Q1'], 3) # Check the format of your answer (three floats)


# In[305]:


### Question 2


# In[306]:


def feature(datum):
    feat1 = datum['review_text'].count('!')
    feat2 = len(datum['review_text'])
    return [feat1]+[feat2]
    


# In[307]:


X = [feature(data) for data in dataset]
Y = [data['rating'] for data in dataset]


# In[308]:


X[:10]


# In[309]:


lr=LinearRegression()
lr.fit(X,Y)


# In[311]:


theta0, [theta1, theta2]=lr.intercept_,lr.coef_
[theta0, theta1, theta2]


# In[312]:


Y_pre=lr.predict(X)
Y_pre


# In[313]:


mse=mean_squared_error(Y,Y_pre)
mse


# In[315]:


answers['Q2'] = [theta0, theta1, theta2, mse]
print(answers["Q2"])


# In[316]:


assertFloatList(answers['Q2'], 4)


# In[38]:


### Question 3


# In[39]:


def feature(datum, deg):
    # feature for a specific polynomial degree
    feat = datum['review_text'].count('!')
    ans=[]
    ans.append(feat)
    for i in range(deg-1):
        tmp=ans[-1]
        ans.append(tmp*feat)
    return [1] + ans


# In[40]:


#degree 1
X = [feature(data,1) for data in dataset]
Y = [data['rating'] for data in dataset]


# In[41]:


X[:10]


# In[42]:


lr=LinearRegression()
lr.fit(X,Y)
Y_pre=lr.predict(X)
mse=mean_squared_error(Y,Y_pre)
mse


# In[45]:


mses=[]


# In[46]:


mses.append(mse)


# In[49]:


#degree 2
X = [feature(data,2) for data in dataset]
lr=LinearRegression()
lr.fit(X,Y)
Y_pre=lr.predict(X)
mse=mean_squared_error(Y,Y_pre)
mse


# In[50]:


mses.append(mse)


# In[51]:


#degree 3
X = [feature(data,3) for data in dataset]
lr=LinearRegression()
lr.fit(X,Y)
Y_pre=lr.predict(X)
mse=mean_squared_error(Y,Y_pre)
mse


# In[52]:


mses.append(mse)


# In[53]:


#degree 4
X = [feature(data,4) for data in dataset]
lr=LinearRegression()
lr.fit(X,Y)
Y_pre=lr.predict(X)
mse=mean_squared_error(Y,Y_pre)
mse


# In[54]:


mses.append(mse)


# In[55]:


#degree 5
X = [feature(data,5) for data in dataset]
lr=LinearRegression()
lr.fit(X,Y)
Y_pre=lr.predict(X)
mse=mean_squared_error(Y,Y_pre)
mse


# In[56]:


mses.append(mse)


# In[65]:


print(mses)


# In[70]:


answers['Q3']=mses


# In[71]:


assertFloatList(answers['Q3'], 5)# List of length 5


# In[121]:


### Question 4


# In[167]:


def feature(datum):
    # your implementation
    feat = datum['review_text'].count('!')
    return [1] + [feat]
X = [feature(data) for data in dataset]
Y = [data['rating'] for data in dataset]


# In[168]:


X_train=X[:len(X)//2]
X_test=X[len(X)//2:]
Y_train=Y[:len(Y)//2]
Y_test=Y[len(Y)//2:]


# In[170]:


#degree 1
lr=LinearRegression()
lr.fit(X_train,Y_train)
Y_pre=lr.predict(X_test)
mse=mean_squared_error(Y_test,Y_pre)
mse


# In[171]:


mses=[]
mses.append(mse)


# In[172]:


def moredegree(data):
    for i in range(len(data)):
        tmp=data[i]+[data[i][1]*data[i][-1]]
        data[i]=tmp
    return data


# In[173]:


#degree 2
X_train=moredegree(X_train)
X_test=moredegree(X_test)
lr.fit(X_train,Y_train)
Y_pre=lr.predict(X_test)
mse=mean_squared_error(Y_test,Y_pre)
mse


# In[174]:


mses.append(mse)


# In[175]:


#degree 3
X_train=moredegree(X_train)
X_test=moredegree(X_test)
lr.fit(X_train,Y_train)
Y_pre=lr.predict(X_test)
mse=mean_squared_error(Y_test,Y_pre)
mse


# In[176]:


mses.append(mse)


# In[177]:


#degree 4
X_train=moredegree(X_train)
X_test=moredegree(X_test)
lr.fit(X_train,Y_train)
Y_pre=lr.predict(X_test)
mse=mean_squared_error(Y_test,Y_pre)
mse


# In[178]:


mses.append(mse)


# In[179]:


#degree 5
X_train=moredegree(X_train)
X_test=moredegree(X_test)
lr.fit(X_train,Y_train)
Y_pre=lr.predict(X_test)
mse=mean_squared_error(Y_test,Y_pre)
mse


# In[180]:


mses.append(mse)


# In[181]:


mses


# In[182]:


answers['Q4'] = mses


# In[183]:


assertFloatList(answers['Q4'], 5)


# In[184]:


### Question 5


# In[185]:


def median(arr):
    arr.sort()
    lenth=len(arr)
    return arr[lenth//2] if lenth%2==1 else (arr[lenth//2]+arr[lenth//2-1])/2


# In[186]:


best_theta=median(Y_train)
best_theta


# In[187]:


from sklearn.metrics import mean_absolute_error


# In[188]:


mae=mean_absolute_error(Y_test,[best_theta]*len(Y_test))
mae


# In[189]:


answers['Q5'] = mae


# In[190]:


assertFloat(answers['Q5'])


# In[534]:


### Question 6


# In[535]:


f = open("beer_50000.json")
dataset = []
for l in f:
    if 'user/gender' in l:
        dataset.append(eval(l))


# In[536]:


len(dataset)


# In[537]:


dataset[0]


# In[538]:


def feature(datum):
    # your implementation
    feat = datum['review/text'].count('!')
    return [feat]


# In[539]:


X = [[1]+feature(data) for data in dataset]
Y = [1 if data['user/gender']=='Female' else 0 for data in dataset]


# In[540]:


print(X[:40])


# In[541]:


print(Y[:40])


# In[542]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[543]:


Lr=LogisticRegression()
Lr.fit(X,Y)


# In[544]:


y_pre=Lr.predict(X)


# In[545]:


coef_mat=confusion_matrix(Y,y_pre)
coef_mat


# In[546]:


TN, FP=coef_mat[0][0],coef_mat[0][1]
FN, TP=coef_mat[1][0],coef_mat[1][1]


# In[547]:


def get_ber(TP,TN,FP,FN):
    return (FP/(FP+TN)+FN/(FN+TP))/2


# In[548]:


BER=get_ber(TP,TN,FP,FN)
BER


# In[549]:


answers['Q6'] = [TP, TN, FP, FN, BER]
print(answers['Q6'])


# In[550]:


assertFloatList(answers['Q6'], 5)


# In[551]:


### Question 7


# In[552]:


X = [[1]+feature(data) for data in dataset]
Y = [data['user/gender']=='Female' for data in dataset]


# In[553]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
Lr=LogisticRegression(class_weight='balanced')
Lr.fit(X,Y)
y_pre=Lr.predict(X)


# In[554]:


y_pre=Lr.predict(X)
coef_mat=confusion_matrix(Y,y_pre)
coef_mat


# In[558]:


TN, FP=coef_mat[0][0],coef_mat[0][1]
FN, TP=coef_mat[1][0],coef_mat[1][1]


# In[559]:


BER=get_ber(TP,TN,FP,FN)
BER


# In[560]:


answers["Q7"] = [TP, TN, FP, FN, BER]
print(answers["Q7"])


# In[501]:


assertFloatList(answers['Q7'], 5)


# In[244]:


### Question 8


# In[245]:


precisionList=[]


# In[249]:


Lr.coef_


# In[253]:


import numpy as np


# In[254]:


conf=[np.dot(Lr.coef_,x) for x in X]
conf


# In[255]:


len(conf)


# In[256]:


sbf=list(zip(conf,Y))
sbf.sort(reverse=True)
sbf[:10]


# In[258]:


def get_pre_k(sbf,k):
    num=0
    for i in range(k):
        if sbf[i][1]==True:
            num+=1
    return num/k


# In[259]:


for k in [1, 10, 100, 1000, 10000]:
    precisionList.append(get_pre_k(sbf,k))
precisionList


# In[ ]:





# In[ ]:





# In[260]:


answers['Q8'] = precisionList


# In[261]:


assertFloatList(answers['Q8'], 5) #List of five floats


# In[562]:


f = open("answers_hw1.txt", 'w') # Write your answers to a file
f.write(str(answers) + '\n')
f.close()


# In[ ]:




