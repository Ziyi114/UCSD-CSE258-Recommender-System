#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# In[5]:


import warnings
warnings.filterwarnings("ignore")


# In[6]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[7]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[8]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# In[9]:


allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)


# In[10]:


len(allRatings)


# In[11]:


ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))


# # Read prediction                                #

# In[10]:


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


# In[11]:


totalRead


# In[12]:


len(return1)


# In[13]:


len(mostPopular)


# In[14]:


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


# In[15]:


ratingsValid[-1]


# In[18]:


def get_acc(predictions,ratingsValid):
    n=len(predictions)
    pos=0
    for i in range(n):
        if (predictions[i]==True and ratingsValid[i][2]!=None) or (predictions[i]==False and ratingsValid[i][2]==None):
            pos+=1
    return pos/n


# In[19]:


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


# In[20]:


acclist=[]
for i in range(1,200):
    thre=0.005*i
    predictions=model_on_different_threshold(thre)
    acc=get_acc(predictions,ratingsValid)
    acclist.append(acc)
max(acclist)


# In[21]:


acclist.index(max(acclist))


# In[22]:


threshold=0.005*(acclist.index(max(acclist))+1)
acc2=max(acclist)


# In[23]:


usersPerbook = defaultdict(set) 
booksPeruser = defaultdict(set) 

for u,b,_ in ratingsTrain:
    usersPerbook[b].add(u)
    booksPeruser[u].add(b)


# In[24]:


def Jaccard(s1, s2):
    nu=len(s1.intersection(s2))
    de=len(s1.union(s2))
    return nu/de


# In[26]:


def predict(u,b,threshold):
    similarities=[]
    for ob in booksPeruser[u]:
        if ob==b:continue
        similarities.append(Jaccard(usersPerbook[ob],usersPerbook[b]))
    if max(similarities,default=0)>threshold:
        return True
    else:
        return False


# In[27]:


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


# In[32]:


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
thresholds=[]
for i in range(100,200):
    for j in range(1,9):
        thre1=i*0.005
        thre2=j*0.01
        predictions=model_on_different_threshold2(thre1,thre2)
        acc=get_acc(predictions,ratingsValid)
        thresholds.append((thre1,thre2))
        acclist2.append(acc)
        #print(acc)
max(acclist2)


# In[33]:


acc=max(acclist2)
acc


# In[34]:


index=acclist2.index(acc)
index


# In[35]:


thre1,thre2=thresholds[index]
(thre1,thre2)


# In[36]:


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


# In[37]:


predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    
    if b in returnthre or predict(u,b,thre2)==True:
        predictions.write(u + ',' + b + ",1\n")
    else:
        predictions.write(u + ',' + b + ",0\n")
    '''
    
    if b in return1:
        predictions.write(u + ',' + b + ",1\n")
    else:
        predictions.write(u + ',' + b + ",0\n")
    '''
predictions.close()


# # Rating Prediction

# In[60]:


ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))


# In[61]:


ratingMean=0
for u,b,r in ratingsTrain:
    ratingMean+=r
ratingMean/=len(ratingsTrain)
ratingMean


# In[126]:


alpha = ratingMean


# In[127]:


N = len(ratingsTrain)
nUsers = len(ratingsPerUser)
nItems = len(ratingsPerItem)
users = list(ratingsPerUser.keys())
items = list(ratingsPerItem.keys())


# In[128]:


def unpack(theta):
    global alpha
    global userBiases
    global itemBiases
    alpha = theta[0]
    userBiases = dict(zip(users, theta[1:nUsers+1]))
    itemBiases = dict(zip(items, theta[1+nUsers:]))


# In[129]:


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


# In[130]:


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


# In[131]:


labels = [r for _,_,r in ratingsTrain]


# In[132]:


from sklearn.metrics import mean_squared_error
MSE=mean_squared_error


# In[133]:


def prediction(user, item):
    if user in userBiases and item in itemBiases:
        return alpha + userBiases[user] + itemBiases[item]
    elif item in itemBiases:
        return alpha + itemBiases[item]
    elif user in userBiases:
        return alpha + userBiases[user]
    else:
        return alpha


# In[134]:


validlabels = [r for _,_,r in ratingsValid]


# In[137]:


MSES=[float('inf')]
bestlamb=float('inf')
for i in range(50):
    lamb=1e-9*i
    scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems),derivative, args = (labels, lamb))
    predictions=[]
    for u,b,r in ratingsValid:
        predict=prediction(u, b)
        predictions.append(predict)
    validMSE=MSE(predictions,validlabels)
    if validMSE<min(MSES):
        bestlamb=lamb
    MSES.append(validMSE)


# In[139]:


bestlamb


# In[140]:


min(MSES)


# In[160]:


scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems),derivative, args = (labels, 1e-5))


# In[161]:


predictions=[]
for u,b,r in ratingsValid:
    predict=prediction(u, b)
    predictions.append(predict)
validMSE=MSE(predictions,validlabels)
validMSE


# In[ ]:


predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"): # header
        predictions.write(l)
        continue
    u,b = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
    # (etc.)
    
    predictions.write(u + ',' + b + ',' + str(prediction(u,b)) + '\n')
    
predictions.close()


# Try latent factor model with gamma instead of simple alpha

# In[49]:


alpha = ratingMean
userBiases = defaultdict(float)
itemBiases = defaultdict(float)


# In[50]:


ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))


# In[118]:


K=4
userGamma = {}
itemGamma = {}
for u,b,r in ratingsTrain:
    userGamma[u] = [random.random() * 0.1 - 0.05 for k in range(K)]
    itemGamma[b] = [random.random() * 0.1 - 0.05 for k in range(K)]


# In[119]:


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


# In[120]:


def inner(x, y):
    return sum([a*b for a,b in zip(x,y)])


# In[141]:


def prediction(user, item):
    if user in userBiases and item in itemBiases:
        return alpha + userBiases[user] + itemBiases[item] + inner(userGamma[user], itemGamma[item])
    elif item in itemBiases:
        return alpha + itemBiases[item]+sum(itemGamma[item])
    elif user in userBiases:
        return alpha + userBiases[user]+sum(userGamma[user])
    else:
        return alpha


# In[122]:


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


# In[123]:


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


# In[57]:


def training_with_l(l):
    scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + # Initialize alpha
                                   [0.0]*(nUsers+nItems) + # Initialize beta
                                   [random.random() * 0.1 - 0.05 for k in range(K*(nUsers+nItems))], # Gamma
                             derivative, args = (labels, l))


# In[66]:


bestmse=float('inf')
bestl=0
for i in range(20):
    l=i*1e-5
    predictions=[]
    training_with_l(l)
    for u,b,r in ratingsValid:
        predict=prediction(u, b)
        predictions.append(predict)
    validMSE=MSE(predictions,validlabels)
    if validMSE<bestmse:
        bestmse=validMSE
        bestl=l


# In[74]:


bestmse


# In[69]:


bestl


# In[124]:


training_with_l(bestl)


# In[125]:


predictions=[]
for u,b,r in ratingsValid:
        predict=prediction(u, b)
        predictions.append(predict)
validMSE=MSE(predictions,validlabels)
validMSE


# In[77]:


predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"): # header
        predictions.write(l)
        continue
    u,b = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
    # (etc.)
    
    predictions.write(u + ',' + b + ',' + str(prediction(u,b)) + '\n')
    
predictions.close()


# In[ ]:


bestl=0
bestmse=float('inf')
bestK=0
def train_model(K,l):
    userGamma = {}
    itemGamma = {}
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
                             derivative, args = (labels, l))
    
for i in range(1,21):
    for j in range(1,21):
        K=i
        l=1e-5*j
        train_model(K,l)
        predictions=[]
        
        for u,b,r in ratingsValid:
            predict=prediction(u, b)
            predictions.append(predict)
        validMSE=MSE(predictions,validlabels)
        if validMSE<bestmse:
            bestmse=validMSE
            bestl=l
            bestK=K
(bestl,bestK)


# In[144]:


(bestl,bestK)


# In[147]:


train_model(1,1e-4)
predictions=[]

for u,b,r in ratingsValid:
    predict=prediction(u, b)
    predictions.append(predict)
validMSE=MSE(predictions,validlabels)
validMSE


# In[ ]:


predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"): # header
        predictions.write(l)
        continue
    u,b = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
    # (etc.)
    
    predictions.write(u + ',' + b + ',' + str(prediction(u,b)) + '\n')
    
predictions.close()


# In[ ]:





# Try SVD from surprise

# SVD1

# In[12]:


import surprise
from surprise import SVD, Reader, Dataset
from surprise.model_selection import train_test_split


# In[13]:


f = gzip.GzipFile("train_Interactions.csv.gz")
f


# In[14]:


allRatings[0]


# In[15]:


fname="train_data"


with open(fname, 'w', newline='') as file:
    #file.write('user item rating'+'\n')
    for datum in allRatings:
        case=[str(t) for t in datum]
        
        file.write(' '.join(case)+'\n')


# In[16]:


reader = Reader(line_format='user item rating', sep=' ')
data = Dataset.load_from_file("train_data", reader=reader)


# In[167]:


trainset, testset = train_test_split(data, test_size=.05)


# In[85]:


model1=surprise.SVD(n_factors=1024,n_epochs=2048,lr_all=0.01)
model1.fit(trainset)


# In[111]:


#model.fit(trainset)
predictions = model1.test(testset)


# In[112]:


sse = 0
for p in predictions:
    sse += (p.r_ui - p.est)**2

print(sse / len(predictions))


# In[113]:


s=model1.predict('u67805239', 'b61372131')
s.est


# In[115]:


predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"): # header
        predictions.write(l)
        continue
    u,b = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
    # (etc.)
    
    predictions.write(u + ',' + b + ',' + str(model1.predict(u,b).est) + '\n')
    
predictions.close()


# SVD exp

# In[108]:


model=surprise.SVD(n_factors=2048,n_epochs=10000,lr_all=0.01)
model.fit(trainset)


# In[109]:


#model.fit(trainset)
predictions = model.test(testset)


# In[110]:


sse = 0
for p in predictions:
    sse += (p.r_ui - p.est)**2

print(sse / len(predictions))


# In[ ]:





# In[152]:


modele=surprise.SVD(n_factors=1,n_epochs=130,lr_all=0.001,reg_pu=1e-3,reg_qi=1e-3)
modele.fit(trainset)


# In[153]:


predictions = modele.test(testset)


# In[154]:


sse = 0
for p in predictions:
    sse += (p.r_ui - p.est)**2


# In[155]:


print(sse / len(predictions))


# In[171]:


model=surprise.SVD(n_factors=2,n_epochs=120,lr_all=0.001,reg_pu=1e-4,reg_qi=1e-4)
model.fit(trainset)
predictions = model.test(testset)
sse = 0
for p in predictions:
    sse += (p.r_ui - p.est)**2

see=(sse / len(predictions))
see


# In[174]:


predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"): # header
        predictions.write(l)
        continue
    u,b = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
    # (etc.)
    
    predictions.write(u + ',' + b + ',' + str(model.predict(u,b).est) + '\n')
    
predictions.close()


# In[ ]:




