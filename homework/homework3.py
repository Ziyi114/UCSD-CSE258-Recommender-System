#!/usr/bin/env python
# coding: utf-8

# In[45]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model
import random


# In[46]:


import warnings
warnings.filterwarnings("ignore")


# In[47]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[48]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[49]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# In[50]:


answers = {}


# In[51]:


# Some data structures that will be useful


# In[52]:


allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)


# In[53]:


len(allRatings)


# In[64]:


ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))


# In[161]:


##################################################
# Rating prediction (CSE258 only)                #
##################################################


# In[ ]:





# In[162]:


### Question 9


# In[163]:


ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))


# In[165]:


ratingMean=0
for u,b,r in ratingsTrain:
    ratingMean+=r
ratingMean/=len(ratingsTrain)
ratingMean


# In[166]:


alpha = ratingMean


# In[167]:


N = len(ratingsTrain)
nUsers = len(ratingsPerUser)
nItems = len(ratingsPerItem)
users = list(ratingsPerUser.keys())
items = list(ratingsPerItem.keys())


# In[169]:


def unpack(theta):
    global alpha
    global userBiases
    global itemBiases
    alpha = theta[0]
    userBiases = dict(zip(users, theta[1:nUsers+1]))
    itemBiases = dict(zip(items, theta[1+nUsers:]))


# In[186]:


def derivative(theta, labels, lamb):
    unpack(theta)
    N = len(ratingsTrain)
    dalpha = 0
    dUserBiases = defaultdict(float)
    dItemBiases = defaultdict(float)
    for u,b,r in ratingsTrain:
        
        pred = prediction(u, b)
        diff = pred - r
        dalpha += 2/N*diff
        dUserBiases[u] += 2/N*diff
        dItemBiases[b] += 2/N*diff
    for u in userBiases:
        dUserBiases[u] += 2*lamb*userBiases[u]
    for i in itemBiases:
        dItemBiases[b] += 2*lamb*itemBiases[b]
    dtheta = [dalpha] + [dUserBiases[u] for u in users] + [dItemBiases[b] for b in items]
    return numpy.array(dtheta)


# In[187]:


def cost(theta, labels, lamb):
    unpack(theta)
    predictions = [prediction(u, b) for u,b,_ in ratingsTrain]
    cost = MSE(predictions, labels)
    print("MSE = " + str(cost))
    for u in userBiases:
        cost += lamb*userBiases[u]**2
    for i in itemBiases:
        cost += lamb*itemBiases[i]**2
    return cost


# In[200]:


def prediction(user, item):
    return alpha + userBiases[user] + itemBiases[item]


# In[189]:


labels = [r for _,_,r in ratingsTrain]


# In[190]:


from sklearn.metrics import mean_squared_error
MSE=mean_squared_error


# In[210]:


scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems),
                             derivative, args = (labels, 1))


# In[211]:


u,b,r=ratingsValid[0]
userBiases[u]
itemBiases[b]


# In[212]:


def prediction(user, item):
    if user in userBiases and item in itemBiases:
        return alpha + userBiases[user] + itemBiases[item]
    elif item in itemBiases:
        return alpha + itemBiases[item]
    elif user in userBiases:
        return alpha + userBiases[user]
    else:
        return alpha


# In[213]:


predictions=[]
for u,b,r in ratingsValid:
    predict=prediction(u, b)
    predictions.append(predict)
len(predictions)==len(ratingsValid)


# In[214]:


validlabels = [r for _,_,r in ratingsValid]
validMSE=MSE(predictions,validlabels)
validMSE


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[215]:


answers['Q9'] = validMSE


# In[216]:


assertFloat(answers['Q9'])


# In[217]:


### Question 10


# In[218]:


maxpair=[ratingsTrain[0][0],userBiases[ratingsTrain[0][0]]]
minpair=[ratingsTrain[0][0],userBiases[ratingsTrain[0][0]]]
for u,b,r in ratingsTrain:
    if userBiases[u]>maxpair[1]:
        maxpair[0]=u
        maxpair[1]=userBiases[u]
    if userBiases[u]<minpair[1]:
        minpair[0]=u
        maxpair[1]=userBiases[u]
maxpair


# In[219]:


minpair


# In[220]:


maxUser, maxBeta=maxpair


# In[221]:


minUser, minBeta=minpair


# In[228]:


minBeta=float(minBeta)


# In[229]:


maxBeta=float(maxBeta)


# In[230]:


answers['Q10'] = [maxUser, minUser, maxBeta, minBeta]
answers['Q10']


# In[231]:


type(answers['Q10'][2])


# In[232]:


assert [type(x) for x in answers['Q10']] == [str, str, float, float]


# In[233]:


### Question 11


# In[239]:


MSES=[float('inf')]
bestlamb=float('inf')
for i in range(50):
    lamb=0.1*i+0.001
    scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems),derivative, args = (labels, lamb))
    predictions=[]
    for u,b,r in ratingsValid:
        predict=prediction(u, b)
        predictions.append(predict)
    validMSE=MSE(predictions,validlabels)
    if validMSE<min(MSES):
        bestlamb=lamb
    MSES.append(validMSE)


# In[240]:


bestlamb


# In[241]:


min(MSES)


# In[244]:


lamb,validMSE=bestlamb,min(MSES)


# In[ ]:





# In[245]:


answers['Q11'] = (lamb, validMSE)


# In[246]:


assertFloat(answers['Q11'][0])
assertFloat(answers['Q11'][1])


# In[247]:


predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"): # header
        predictions.write(l)
        continue
    u,b = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
    # (etc.)
    
predictions.close()


# In[55]:


##################################################
# Read prediction                                #
##################################################


# In[56]:


# Copied from baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/2: break


# In[57]:


totalRead


# In[58]:


len(return1)


# In[59]:


len(mostPopular)


# In[60]:


### Question 1


# In[61]:


allRatings[0]


# In[62]:


ratingsValid[0]


# In[65]:


bookPeruser=defaultdict(set)
allbook=set()
for u,b,r in ratingsTrain:
    bookPeruser[u].add(b)
    allbook.add(b)
allbook=list(allbook)
i=0
n=len(ratingsValid)
while i<n:
    #print(n)
    u,b,_=ratingsValid[i]
    nb=random.choice(allbook)
    while nb in bookPeruser[u]:
        nb=random.choice(allbook)
    ratingsValid.append((u,nb,None))
    i+=1
len(ratingsValid)


# In[66]:


ratingsValid[-1]


# In[72]:


predictions=[]
for u,b,_ in ratingsValid:
    if b in return1:
        predictions.append(True)
    else:
        predictions.append(False)
len(predictions)


# In[73]:


def get_acc(predictions,ratingsValid):
    n=len(predictions)
    pos=0
    for i in range(n):
        if (predictions[i]==True and ratingsValid[i][2]!=None) or (predictions[i]==False and ratingsValid[i][2]==None):
            pos+=1
    return pos/n


# In[74]:


acc1=get_acc(predictions,ratingsValid)
acc1


# In[75]:


answers['Q1'] = acc1


# In[76]:


assertFloat(answers['Q1'])


# In[78]:


### Question 2


# In[86]:


def model_on_different_threshold(thre):
    bookCount = defaultdict(int)
    totalRead = 0

    for user,book,_ in readCSV("train_Interactions.csv.gz"):
        bookCount[book] += 1
        totalRead += 1

    mostPopular = [(bookCount[x], x) for x in bookCount]
    mostPopular.sort()
    mostPopular.reverse()

    returnthre = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        returnthre.add(i)
        if count > totalRead*thre: break
    predictions=[]
    for u,b,_ in ratingsValid:
        if b in returnthre:
            predictions.append(True)
        else:
            predictions.append(False)
    return predictions


# In[89]:


acclist=[]
for i in range(1,200):
    thre=0.005*i
    predictions=model_on_different_threshold(thre)
    acc=get_acc(predictions,ratingsValid)
    acclist.append(acc)
max(acclist)


# In[91]:


acclist.index(max(acclist))


# In[95]:


threshold=0.005*(acclist.index(max(acclist))+1)
acc2=max(acclist)


# In[96]:


answers['Q2'] = [threshold, acc2]


# In[97]:


assertFloat(answers['Q2'][0])
assertFloat(answers['Q2'][1])


# In[98]:


### Question 3/4


# In[114]:


usersPerbook = defaultdict(set) 
booksPeruser = defaultdict(set) 

for u,b,_ in ratingsTrain:
    usersPerbook[b].add(u)
    booksPeruser[u].add(b)


# In[115]:


def Jaccard(s1, s2):
    nu=len(s1.intersection(s2))
    de=len(s1.union(s2))
    return nu/de


# In[129]:


def predict(u,b,threshold):
    similarities=[]
    for ob in booksPeruser[u]:
        if ob==b:continue
        similarities.append(Jaccard(usersPerbook[ob],usersPerbook[b]))
    if max(similarities,default=0)>threshold:
        return True
    else:
        return False


# In[130]:


u,b,_=ratingsValid[0]
predict(u,b,threshold)


# In[131]:


def model(threshold):
    predictions=0
    for u,b,r in ratingsValid:
        if predict(u,b,threshold)==True and r!=None:
            predictions+=1
        elif predict(u,b,threshold)==False and r==None:
            predictions+=1
        else:
            pass
    return predictions/len(ratingsValid)


# In[134]:


acclist=[]
for i in range(1,99):
    threshold=i*0.01
    acclist.append(model(threshold))
    #print(len(acclist))
max(acclist)


# In[138]:


acclist.index(max(acclist))


# In[139]:


threshold=0.01


# In[140]:


acc3=max(acclist)


# In[141]:


answers['Q3'] = acc3


# In[149]:


def model_on_different_threshold2(thre1,thre2):
    bookCount = defaultdict(int)
    totalRead = 0

    for user,book,_ in readCSV("train_Interactions.csv.gz"):
        bookCount[book] += 1
        totalRead += 1

    mostPopular = [(bookCount[x], x) for x in bookCount]
    mostPopular.sort()
    mostPopular.reverse()

    returnthre = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        returnthre.add(i)
        if count > totalRead*thre1: break
    predictions=[]
    for u,b,_ in ratingsValid:
        if b in returnthre or predict(u,b,thre2)==True:
            predictions.append(True)
        else:
            predictions.append(False)
    return predictions

#select the best pair of thresholds
#this procedure takes at least ten minutes since pairs are too many
acclist2=[]
for i in range(100,200):
    for j in range(1,9):
        thre1=i*0.005
        thre2=j*0.01
        predictions=model_on_different_threshold2(thre1,thre2)
        acc=get_acc(predictions,ratingsValid)
        acclist2.append(acc)
        #print(acc)
max(acclist2)


# In[150]:


acc4=max(acclist2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[151]:


answers['Q3'] = acc3
answers['Q4'] = acc4


# In[152]:


assertFloat(answers['Q3'])
assertFloat(answers['Q4'])


# In[154]:


predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    # (etc.)

predictions.close()


# In[155]:


answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"


# In[156]:


assert type(answers['Q5']) == str


# In[ ]:


##################################################
# Category prediction (CSE158 only)              #
##################################################


# In[ ]:


### Question 6


# In[ ]:


data = []

for d in readGz("train_Category.json.gz"):
    data.append(d)


# In[ ]:


data[0]


# In[ ]:





# In[ ]:


answers['Q6'] = counts[:10]


# In[ ]:


assert [type(x[0]) for x in answers['Q6']] == [int]*10
assert [type(x[1]) for x in answers['Q6']] == [str]*10


# In[ ]:


### Question 7


# In[ ]:





# In[ ]:


Xtrain = X[:9*len(X)//10]
ytrain = y[:9*len(y)//10]
Xvalid = X[9*len(X)//10:]
yvalid = y[9*len(y)//10:]


# In[ ]:





# In[ ]:


answers['Q7'] = acc7


# In[ ]:


assertFloat(answers['Q7'])


# In[ ]:


### Question 8


# In[ ]:





# In[ ]:


answers['Q8'] = acc8


# In[ ]:


assertFloat(answers['Q8'])


# In[ ]:


# Run on test set


# In[ ]:


predictions = open("predictions_Category.csv", 'w')
pos = 0

for l in open("pairs_Category.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    # (etc.)

predictions.close()


# In[248]:


f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




