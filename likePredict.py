import pandas as pd
import numpy as np
import os
import random
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df=pd.read_csv('dataFinal.csv', usecols = ['categoryId','channelSubscriberCount','dislikeCount','viewCount','commentCount','viewCount/channel_month_old','viewCount/video_month_old','viewCount/NoOfTags'])
df.head()

# df.to_csv('youtubeData.csv',index=False)

df=pd.read_csv('dataFinal.csv', usecols = ['likeCount'])

# df.to_csv('likeCount.csv',index=False)

data=[]
with open('youtubeData.csv') as csvFile:
    reader=csv.reader(csvFile,quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        data.append(row)

data=np.array(data)

# print(data)

likes=[]
with open('likeCount.csv') as csvFile:
    reader=csv.reader(csvFile,quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        likes.append(row)

likes=np.array(likes)

# print(likes)

testData=[]
testDataResults=[]

with open("testData.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: 
        testData.append(row)

with open("testDataResults.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: 
        testDataResults.append(row)

count=0

###############################################################

dt=tree.DecisionTreeClassifier(max_depth=5)
dt.fit(data,likes)

predictedValues=dt.predict(testData)
# print(predictedValues)

for i in range(len(predictedValues)):
    if(testDataResults[i][0]-predictedValues[i]>0):
        if(testDataResults[i][0]-predictedValues[i]<1000):
            count+=1
    else:
        if(testDataResults[i][0]-predictedValues[i]>-1000):
            count+=1

# print(count)

print("Accuracy of Decision Tree : ",(count/190)*100,"%")

###############################################################

rf=RandomForestClassifier(n_estimators=100,max_depth=2,random_state=0)
rf.fit(data,np.ravel(likes,order='C'))

predictedValues=rf.predict(testData)
# print(predictedValues)

count=0

for i in range(len(predictedValues)):
    if(testDataResults[i][0]-predictedValues[i]>0):
        if(testDataResults[i][0]-predictedValues[i]<1000):
            count+=1
    else:
        if(testDataResults[i][0]-predictedValues[i]>-1000):
            count+=1

print("Accuracy of Random Forest : ",(count/190)*100,"%")

###############################################################

lin=SGDClassifier()
lin.fit(data,np.ravel(likes,order='C'))

predictedValues=lin.predict(testData)
# print(predictedValues)

count=0

for i in range(len(predictedValues)):
    if(testDataResults[i][0]-predictedValues[i]>0):
        if(testDataResults[i][0]-predictedValues[i]<1000):
            count+=1
    else:
        if(testDataResults[i][0]-predictedValues[i]>-1000):
            count+=1

# print(count)

print("Accuracy of Linear Classifier : ",(count/190)*100,"%")

###############################################################

nb=GaussianNB()
nb.fit(data,np.ravel(likes,order='C'))

predictedValues=nb.predict(testData)
# print(predictedValues)

count=0

for i in range(len(predictedValues)):
    if(testDataResults[i][0]-predictedValues[i]>0):
        if(testDataResults[i][0]-predictedValues[i]<1000):
            count+=1
    else:
        if(testDataResults[i][0]-predictedValues[i]>-1000):
            count+=1

# print(count)

print("Accuracy of Naive Bayes : ",(count/190)*100,"%")

###############################################################

categoryID=0
channelSubscriberCount=float(input("Enter the channel subscriber count : "))
dislikeCount=float(input("Enter the dislike count : "))
viewCount=float(input("Enter the total number of views : "))
commentCount=float(input("Enter the total number of comments : "))
channelAge=float(input("Enter the age of channel in months : "))
videoAge=float(input("Enter the age of video in months : "))
tagCount=float(input("Enter the number of tags for the video : "))

arr=[categoryID,channelSubscriberCount,dislikeCount,viewCount,commentCount,channelAge,videoAge,tagCount]
arr=np.array(arr)
arr=arr.transpose()
arr=arr.reshape(1,-1)

temp=nb.predict(arr)
print("The given video is going to get ",int(temp[0])," likes")

