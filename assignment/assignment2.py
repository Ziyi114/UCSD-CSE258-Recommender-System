#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
import json
from collections import defaultdict


# problem 1
# 
# In our project, we use the dataset of Amazon Product Reviews. Because this dataset is so large 
# and it has already divided into many groups based on product categories,we decide to choose one category of product to analyze in our project
# The category we decide to choose is "Video Games".

# In[2]:


f = gzip.open('Video_Games.json.gz')
dataset = []
for l in f:
    dataset.append(json.loads(l))


# There are 2565349 product reviews for "Video Games" in total.Therefore, we think this dataset is large enough.

# In[3]:


len(dataset)


# For every review, all the information of the review is stored in a dict which is shown like this:
# 
# the overall is the rating of the product.
# 
# the verified is whether this review is verified or not
# 
# the reviewTime is the time when the review was made
# 
# the reviewerID is the ID of the reviewer who made this review to diff from others
# 
# the asin is the ID of the product
# 
# the reviewerName is the name of the reviewer who made this review
# 
# the reviewText is the comments made by the reviewers to the product in detail
# 
# the summary is the summary of the comments made by the reviewers to the product in short
# 
# the unixReviewTime is time of the review (unix time)

# In[4]:


dataset[0]


# the overall is a float which range from 1.0 to 5.0 (rating of product from low to high)
# 
# the verified is a boolean which is whether true or false
# 
# the reviewTime is a string to represent time
# 
# the reviewerID is a string to represent the reviewer
# 
# the asin is a string to represent the product
# 
# the reviewerName is a string to represent the reviewer
# 
# the reviewText is a string to represent comments
# 
# the summary is a string to represent summary of comments
# 
# the unixReviewTime is a integer to represent time

# In[5]:


overall_list = []
verified_list = []
for i in dataset:
    if i['overall'] not in overall_list:
        overall_list.append(i['overall'])
    if i['verified'] not in verified_list:
        verified_list.append(i['verified'])
print(overall_list)
print(verified_list)


# some interesting finding in our exploratory analysis
# 
# (1)
# the reviews with "true" verified attribute is relativly trustable than the reviews with "false" verified attribute. For example, for one product,
#  if a lot of reviews give low rating to it, another review with "true" verified attribute will always give low rating to it while another review with
#  false" verified attribute may give high rating to it. Threfore, we think to make our prediction task more trustable, in our training process,
#  we will pick up the "true" verified reviews to train our model instead of the whole dataset to make our model more available.
# 
# (2)
# we find that the rating for one product will change with time goes, which I means the later reviews to one product may be a little different from the earlier ones.
# Therefore, we think in our prediction task, we prefer to use the lateset reviews which is relative more suitable.
# 
# (3)
# we find that there is a positive relationship between the comments and the rating of the reivews. In the comment of a review, if the feedback in the comment is positive,
# the rating is relatively high. When the feedback is negative, the rating is relatively low. Therefore, we can make our prediction task to be predict rating based on the sentiment
# analysis of the reviewText

# problem 2
# 
# In our project, we will make two predictive task.
# We will divide the first 2300000 reviews to train, 200000 reviews to valid and the last 65349 to test
# (1)We will predict the rating of new reviews based on the past reviews. Here we will use three features: overall, reviewerID and asin and we will process the dataset in the following way to
# get this data
# 

# In[7]:


allRatings = []
for i in dataset:
    allRatings.append([i['reviewerID'],i['asin'],i['overall']])
allRatings[1]


# In[8]:


#we set a part of the dataset as the vaild dataset to assess the validity of your model’s predictions (avoid overfitting and test the accuracy in the same time)
ratingsTrain = allRatings[:2300000]
ratingsValid = allRatings[2300000:2500000]
ratingsTest = allRatings[2500000:]


# Our model for this predictive task will be SVD, Simple (bias only) latent factor-based recommender and Complete latent factor model. We will
# try to modify the parameters of these models to compare which model is better by drawing the figure of accuracy relative to different parameters(in the meantime, we will
# avoid overfitting by using the ratingValid)
# We will use the MSE to judge the accuracy of rating prediction.

# In[10]:


#for the baseline
### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

all_kinds_ratings = []
userRatings = defaultdict(list)
'''
for i in allRatings:
    user = i[0]
    item = i[1]
    rating = i[2]
    all_kinds_ratings.append(rating)
    userRatings[user].append(rating)
'''

for i in ratingsTrain:
    user = i[0]
    item = i[1]
    rating = i[2]
    all_kinds_ratings.append(rating)
    userRatings[user].append(rating)

globalAverage = sum(all_kinds_ratings) / len(all_kinds_ratings)
userAverage = {}
for u in userRatings:
    userAverage[u] = sum(userRatings[u]) / len(userRatings[u])


# In[3]:


#problem 2
#(2)
#We will make prediction of rating based on the comments of reviews and summary of comments of reviews
#(1)We will predict the rating of new reviews based on the past reviews. Here we will use three features: overall, reviewerID, asin, reviewText and summary and we will process the dataset in the following way to
#get this data
#some reviews do not have reviewText or summary, we should clean them
allRatings_2 = []
for i in dataset:
    if 'reviewText' in i and 'summary' in i:
        allRatings_2.append([i['reviewerID'],i['asin'],i['overall'],i['reviewText'],i['summary']])
len(allRatings_2)


# In[12]:


#we set a part of the dataset as the vaild dataset to assess the validity of your model’s predictions (avoid overfitting and test the accuracy in the same time)
ratingsTrain_2 = allRatings_2[:2300000]
ratingValid_2 = allRatings_2[2300000:2500000]
ratingTest_2 = allRatings_2[2500000:]


# Our model for this task is rating prediction based on sentiment analysis. We will pick up some critical words from comments and give different weight
# to different words, then based on the calculation of combination of weights and compare the result to the theshold which is set by ourself, we
# will give a predictive rating.
# We will use the MSE to judge the accuracy of rating prediction.

# In[14]:


#for the baseline
#we will also use the rating baseline based on the review history of the user, then compare the effect of our model to the baseline
### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

all_kinds_ratings_2 = []
userRatings_2 = defaultdict(list)
'''
for i in allRatings_2:
    user = i[0]
    item = i[1]
    rating = i[2]
    all_kinds_ratings_2.append(rating)
    userRatings_2[user].append(rating)
'''
for i in ratingsTrain_2:
    user = i[0]
    item = i[1]
    rating = i[2]
    all_kinds_ratings_2.append(rating)
    userRatings_2[user].append(rating)

globalAverage_2 = sum(all_kinds_ratings_2) / len(all_kinds_ratings_2)
userAverage_2 = {}
for u in userRatings_2:
    userAverage_2[u] = sum(userRatings_2[u]) / len(userRatings_2[u])


# problem 3-1

# In[15]:


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
import warnings
from sklearn.metrics import mean_squared_error

MSE=mean_squared_error
warnings.filterwarnings("ignore")


# baseline

# In[16]:


predictions_bs=[]
valid_bs=[]
for u,i,r in ratingsValid:
    rp=userAverage[u] if u in userAverage else globalAverage
    predictions_bs.append(rp)
    valid_bs.append(r)
mse=0
for i in range(len(valid_bs)):
    mse+=abs(valid_bs[i]**2-predictions_bs[i]**2)
mse/=len(valid_bs)
mse


# In[88]:


MSE(predictions_bs,valid_bs)


# svd

# In[163]:


allRatings[:10]


# In[89]:


import surprise
from surprise import SVD, Reader, Dataset
from surprise.model_selection import train_test_split


# In[90]:


fname="amazon_data"
with open(fname, 'w', newline='') as file:
    #file.write('user item rating'+'\n')
    for datum in allRatings:
        case=[str(t) for t in datum]
        file.write(' '.join(case)+'\n')


# In[91]:


reader = Reader(line_format='user item rating', sep=' ')
data = Dataset.load_from_file("train_data", reader=reader)


# In[92]:


trainset, testset = train_test_split(data, test_size=.1)


# In[113]:


model_svd=surprise.SVD(n_factors=256,n_epochs=2000,lr_all=0.001,reg_pu=0.001,reg_qi=0.001)
model_svd.fit(trainset)


# In[114]:


predictions_svd = model_svd.test(testset)
sse = 0
for p in predictions_svd:
    sse += (p.r_ui - p.est)**2
sse/=len(predictions_svd)
sse


# In[204]:


mses=[]
for i in range(30,50):
    model_svd=surprise.SVD(n_factors=i*10,n_epochs=2000,lr_all=0.001,reg_pu=0.001,reg_qi=0.001)
    model_svd.fit(trainset)
    predictions_svd = model_svd.test(testset)
    sse = 0
    for p in predictions_svd:
        sse += (p.r_ui - p.est)**2
    sse/=len(predictions_svd)
    mses.append(sse)
    print(i)


# In[207]:


mses


# In[208]:


[i*10 for i in range(30,50)]


# simple latent factor

# In[23]:


ratingsTrain = allRatings[:2300000]
ratingsValid = allRatings[2300000:2500000]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))


# In[24]:


ratingMean=0
for u,b,r in ratingsTrain:
    ratingMean+=r
ratingMean/=len(ratingsTrain)
ratingMean


# In[25]:


alpha = ratingMean


# In[26]:


N = len(ratingsTrain)
nUsers = len(ratingsPerUser)
nItems = len(ratingsPerItem)
users = list(ratingsPerUser.keys())
items = list(ratingsPerItem.keys())


# In[27]:


#all funcs for model
def unpack(theta):
    global alpha
    global userBiases
    global itemBiases
    alpha = theta[0]
    userBiases = dict(zip(users, theta[1:nUsers+1]))
    itemBiases = dict(zip(items, theta[1+nUsers:]))

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

def prediction(user, item):
    if user in userBiases and item in itemBiases:
        return alpha + userBiases[user] + itemBiases[item]
    elif item in itemBiases:
        return alpha + itemBiases[item]
    elif user in userBiases:
        return alpha + userBiases[user]
    else:
        return alpha


# In[28]:


train_labels = [r for _,_,r in ratingsTrain]
valid_labels = [r for _,_,r in ratingsValid]


# In[29]:


scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems),derivative, args = (train_labels, 0.001))
predictions_sim=[]
for u,b,r in ratingsValid:
    predict=prediction(u, b)
    predictions_sim.append(predict)
MSE(predictions_sim,valid_labels)


# In[170]:


scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems),derivative, args = (train_labels, 0.001))
predictions_sim2=[]
for u,b,r in ratingsValid:
    predict=prediction(u, b)
    predictions_sim2.append(predict)
MSE(predictions_sim2,valid_labels)


# Latent Factor

# Try smaller dataset

# In[167]:


random.shuffle(allRatings)
ratingsTrain = allRatings[:300000]
ratingsValid = allRatings[2300000:2400000]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))


# In[168]:


ratingMean=0
for u,b,r in ratingsTrain:
    ratingMean+=r
ratingMean/=len(ratingsTrain)
alpha = ratingMean
N = len(ratingsTrain)
nUsers = len(ratingsPerUser)
nItems = len(ratingsPerItem)
users = list(ratingsPerUser.keys())
items = list(ratingsPerItem.keys())
train_labels = [r for _,_,r in ratingsTrain]
valid_labels = [r for _,_,r in ratingsValid]


# In[169]:


def prediction(user, item):
    if user in userBiases and item in itemBiases:
        return alpha + userBiases[user] + itemBiases[item] + inner(userGamma[user], itemGamma[item])
    elif item in itemBiases:
        return alpha + itemBiases[item]+sum(itemGamma[item])
    elif user in userBiases:
        return alpha + userBiases[user]+sum(userGamma[user])
    else:
        return alpha
    
def inner(x, y):
    return sum([a*b for a,b in zip(x,y)])
userGamma = {}
itemGamma = {}

def train_model(K,l):
   
    for u,b,r in ratingsTrain:
        userGamma[u] = [random.random() * 0.1 - 0.05 for k in range(K)]
        itemGamma[b] = [random.random() * 0.1 - 0.05 for k in range(K)]
    def unpack(theta):
        global alpha
        global userBiases
        global itemBiases
        global userGamma
        global itemGamma
        index = 0
        alpha = theta[index]
        index += 1
        userBiases = dict(zip(users, theta[index:index+nUsers]))
        index += nUsers
        itemBiases = dict(zip(items, theta[index:index+nItems]))
        index += nItems
        for u in users:
            userGamma[u] = theta[index:index+K]
            index += K
        for i in items:
            itemGamma[i] = theta[index:index+K]
            index += K
    
    def cost(theta, labels, lamb):
        unpack(theta)
        predictions = [prediction(u, b) for u,b,_ in ratingsTrain]
        cost = MSE(predictions, labels)
        print("MSE = " + str(cost))
        for u in users:
            cost += lamb*userBiases[u]**2
            for k in range(K):
                cost += lamb*userGamma[u][k]**2
        for i in items:
            cost += lamb*itemBiases[i]**2
            for k in range(K):
                cost += lamb*itemGamma[i][k]**2
        return cost
    
    def derivative(theta, labels, lamb):
        unpack(theta)
        N = len(ratingsTrain)
        dalpha = 0
        dUserBiases = defaultdict(float)
        dItemBiases = defaultdict(float)
        dUserGamma = {}
        dItemGamma = {}
        for u in ratingsPerUser:
            dUserGamma[u] = [0.0 for k in range(K)]
        for i in ratingsPerItem:
            dItemGamma[i] = [0.0 for k in range(K)]
        for u,b,r in ratingsTrain:
            pred = prediction(u, i)
            diff = pred - r
            dalpha += 2/N*diff
            dUserBiases[u] += 2/N*diff
            dItemBiases[i] += 2/N*diff
            for k in range(K):
                dUserGamma[u][k] += 2/N*itemGamma[i][k]*diff
                dItemGamma[i][k] += 2/N*userGamma[u][k]*diff
        for u in userBiases:
            dUserBiases[u] += 2*lamb*userBiases[u]
            for k in range(K):
                dUserGamma[u][k] += 2*lamb*userGamma[u][k]
        for i in itemBiases:
            dItemBiases[i] += 2*lamb*itemBiases[i]
            for k in range(K):
                dItemGamma[i][k] += 2*lamb*itemGamma[i][k]
        dtheta = [dalpha] + [dUserBiases[u] for u in users] + [dItemBiases[i] for i in items]
        for u in users:
            dtheta += dUserGamma[u]
        for i in items:
            dtheta += dItemGamma[i]
        return numpy.array(dtheta)
    
    scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + # Initialize alpha
                                   [0.0]*(nUsers+nItems) + # Initialize beta
                                   [random.random() * 0.1 - 0.05 for k in range(K*(nUsers+nItems))], # Gamma
                             derivative, args = (train_labels, l))
train_model(8,0.005)
predictions_lf=[]

for u,b,r in ratingsValid:
    predict=prediction(u, b)
    predictions_lf.append(predict)

MSE(predictions_lf,valid_labels)


# problem 3-2

# NLP metheod

# item2vec

# In[4]:


import gzip
import math
import matplotlib.pyplot as plt
import numpy
import random
import sklearn
import string
from collections import defaultdict
from gensim.models import Word2Vec
from nltk.stem.porter import *
from sklearn import linear_model
from sklearn.manifold import TSNE
import random


# In[6]:


random.shuffle(allRatings_2)
ratingsTrain2 = allRatings_2[:2300000]
ratingsValid2 = allRatings_2[2300000:2500000]
ratingsPerUser2 = defaultdict(list)
ratingsPerItem2 = defaultdict(list)
for u,b,r,re,su in ratingsTrain2:
    ratingsPerUser2[u].append((b,r))
    ratingsPerItem2[b].append((u,r))


# In[7]:


itemAverages = defaultdict(list)
reviewsPerUser = defaultdict(list)
    
for u,i,r,re,su in ratingsTrain2:
    
    itemAverages[i].append(r)
    reviewsPerUser[u].append((u,i,r,re,su))
    
for i in itemAverages:
    itemAverages[i] = sum(itemAverages[i]) / len(itemAverages[i])


# focus on review

# In[176]:


reviewLists = []
for u in reviewsPerUser:
    rl = list(reviewsPerUser[u])
    rl.sort()
    reviewLists.append([x[1] for x in rl])


# In[161]:


list(reviewsPerUser)
reviewLists


# In[182]:


model10 = Word2Vec(reviewLists,
                 min_count=10, # Words/items with fewer instances are discarded
                 vector_size=100,  # Model dimensionality
                 window=3, # Window size
                 sg=1) # Skip-gram model


# In[183]:


notin=[]
def predictRating(user,item):
    ratings = []
    similarities = []
    if not str(item) in model10.wv:
        notin.append(1)
        return ratingMean
    for u,i,r,re,su in reviewsPerUser[user]:
        i2 = i
        if i2 == item: continue
        ratings.append(r - itemAverages[i2])
        if str(i2) in model10.wv:
            similarities.append(model10.wv.distance(str(item), str(i2)))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        return ratingMean


# In[184]:


'0439381673' in model10.wv


# In[185]:


predictions_i2=[predictRating(u,i) for u,i,r,re,su in ratingsValid2]
valid_labels2 = [r for _,_,r,re,su in ratingsValid2]
MSE(predictions_i2,valid_labels2)


# In[181]:


len(notin)


# In[186]:



def predictRating2(user,item):
    ratings = []
    similarities = []
    if not str(item) in model10.wv:
        notin.append(1)
        return ratingMean
    for u,i,r,re,su in reviewsPerUser[user]:
        i2 = i
        if i2 == item: continue
        ratings.append(r - itemAverages[i2])
        if str(i2) in model10.wv:
            similarities.append(model10.wv.similarity(str(item), str(i2)))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        return ratingMean


# In[187]:


predictions_i22=[predictRating2(u,i) for u,i,r,re,su in ratingsValid2]
valid_labels2 = [r for _,_,r,re,su in ratingsValid2]
MSE(predictions_i22,valid_labels2)


# word2vec

# In[8]:


gameStyles = {} # Style of each item
categories = set() # Set of item categories
reviewsPerUser = defaultdict(list)
gameIdToName = {} # Map an ID to the name of the product


# In[9]:


reviews = []
reviewDicts = []

for u,i,r,re,su in ratingsTrain2:
    reviews.append(re)
    reviewsPerUser[u].append((re, i))
    reviewDicts.append((u,i,r,re,su))
    if len(reviews) == 50000:
        break


# In[10]:


reviewTokens = []
punctuation = set(string.punctuation)
for re in reviews:
    re = ''.join([c for c in re.lower() if not c in punctuation])
    tokens = []
    for w in re.split():
        tokens.append(w)
    reviewTokens.append(tokens)


# In[49]:


modelw2v = Word2Vec(reviewTokens,
                 min_count=40, # Words/items with fewer instances are discarded
                 vector_size=100,# Model dimensionality
                 window=3, # Window size
                 sg=1) # Skip-gram model


# In[56]:


X=[]
y=[]
import numpy as np
for u,i,r,re,su in ratingsTrain2:
    y.append(r)
    tmp=None
    word = ''.join([c for c in re.lower() if not c in punctuation])
    tokens = []
    for w in word.split():
        tokens.append(w)
    cont=0
    for token in tokens:
        if isinstance(tmp,np.ndarray)==False:
            if token not in modelw2v.wv:
                continue
            tmp=modelw2v.wv[token].copy()
            cont+=1
            #print(tmp)
        else:
            if token not in modelw2v.wv:
                continue
            new=modelw2v.wv[token].copy()
            #print(new)
            tmp+=new
            cont+=1
    if type(tmp)==type(None):
        tmp=[0]*100
    else:
        tmp/=cont
    X.append(tmp)


# In[53]:


a=np.array([1,2,3])/2
a


# In[57]:



X_train=[]
for x in X:
    if type(x)==type(None):
        x=[0]*20
    #print(x)
    X_train.append(list(x))
X_train[:10]


# In[59]:


clf = linear_model.Ridge(1.0, fit_intercept=False) # MSE + 1.0 l2
clf.fit(X_train, y)


# In[60]:


y=[]
for u,i,r,re,su in ratingsValid2:
    y.append(r)


# In[61]:


X_valid=[]
for u,i,r,re,su in ratingsValid2:
    y.append(r)
    tmp=None
    word = ''.join([c for c in re.lower() if not c in punctuation])
    tokens = []
    for w in word.split():
        tokens.append(w)
    cont=0
    for token in tokens:
        if type(tmp)==type(None):
            if token not in modelw2v.wv:
                continue
            tmp=modelw2v.wv[token].copy()
            cont+=1
        else:
            if token not in modelw2v.wv:
                continue
            new=modelw2v.wv[token].copy()
            tmp+=new
            cont+=1
    if type(tmp)==type(None):
        tmp=[0]*100
    else:
        tmp/=cont
    X_valid.append(tmp)


# In[62]:


X_va=[]
for x in X_valid:
    if type(x)==type(None):
        x=[0]*20
    #print(x)
    X_va.append(list(x))
pre=clf.predict(X_va)
from sklearn.metrics import mean_squared_error
valid_labels2 = [r for _,_,r,re,su in ratingsValid2]
MSE=mean_squared_error
MSE(pre,valid_labels2)

