# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:21:36 2019

@author: mt01034
"""


import pandas as pd  
import numpy as np 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
from KNNImplement import MyKNeighborsClassifier as myKNeighborsClassifier
from sklearn.model_selection import train_test_split
import atexit
import sys


result_table=pd.DataFrame(columns=['distance','weight','k','accuracy','featurenames'])

def sens_preci(mc):
    m = []
    for i in range(mc.shape[0]):
        sensitivity=(mc[i,i]/float(sum(mc[i,:])+0.000000000000001))
        precision=(mc[i,i]/float(sum(mc[:,i])+0.000000000000001))
        m += [[sensitivity,precision]]
    return m


def MSKNN(X,y,distance,weight,kMax=10,bfeatures=1,fNum=15):
    if (bfeatures == 1):
        MCKNNSelectFeatures(X,y,distance,weight,kMax,fNum)
    else:
        KNN(X,y,kMax)
    
def MCKNNSelectFeatures(X,y,dtype,w,kMax,fNum):
    features= []
    featurenames=[]
    selectedscorelist = []
    
    count=0 
    distance = ['manhattan','euclidean','minkowski','cosine','canberra','lorentzian','sorensen','hamming','JaccardModif']
    weight = ['uniform','distance','distancesquare']
   
    print ("dtype : "+str(distance[dtype])+"weight : "+str([w]))
    for k in range(1,kMax):
        knnclassifier = myKNeighborsClassifier(n_neighbors=k,weights=weight[w],distancetype =distance[dtype])
        for j in range(fNum): 
            scoref=np.zeros(len(X.columns))
            for i, feature in enumerate(X.columns):
                cv = LeaveOneOut()
                if i in features:
                    #print ("jump")
                    continue
                nF = [feature] + list(X.columns[features])
                scores = cross_val_score(knnclassifier, np.array(X[nF]), np.array(y), cv=cv)  #scoring='f1_macro' cv=12
                #print ("nF value :" + str(nF)+ "\n i="+str(i)+" \n score:"+str(scores))
                scoref[i] = np.mean(scores)
               
            ms = max(scoref)
            if ms ==1:
                selectedscorelist += [max(scoref)]
                features += [np.argmax(scoref)]
                featurenames.insert(j,X.columns[np.argmax(scoref)]) 
                result_table.loc[count,['distance','weight','k','accuracy','featurenames']]=[dtype,w,k,max(scoref),featurenames]
                break
            
        selectedscorelist += [max(scoref)]
        features += [np.argmax(scoref)]
        featurenames.insert(j,X.columns[np.argmax(scoref)]) 
        result_table.loc[count,['distance','weight','k','accuracy','featurenames']]=[dtype,w,k,max(scoref),featurenames]
        count=count+1      
        print_full (result_table)
        
        
    print_full (result_table)
    
   

def KNN(X,Y,kmax=10):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    
    count=0 
    distance = ['manhattan','euclidean','minkowski','cosine','canberra','lorentzian','sorensen','hamming','JaccardModif']
    weight = ['uniform','distance','distancesquare']
    for i, dtype in enumerate(distance):
        for j,w in enumerate(weight):
            for k in range(1,kmax):
                classifier = myKNeighborsClassifier(n_neighbors=k,weights=w,distancetype =dtype)
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                
                accuracy = sum(y_pred==y_test)/float(len(y_test))
        
                mc=confusion_matrix(y_test, y_pred)
                sp=(np.array(sens_preci(mc)))
                count=count+1
                #print(classification_report(y_test, y_pred)) #F1 Score = 2*(Recall * Precision) / (Recall + Precision) #Precison==sensitivity
    print_full (result_table)


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    f= open("KNNResult.txt","w+")
    f.write(str(x))
    f.close()
    pd.reset_option('display.max_rows')
    
def printonExit():
    print_full (result_table)
    f= open("KNNResult.txt","w+")
    f.write("final")
    f.write(str(result_table))
    f.close()

#####################################################


pathogendata=pd.read_csv("output_filename.csv")

X = pathogendata.drop(['Experiment', 'Paper', 'Method', 'Pathogens'], axis=1)
y = pathogendata['Pathogens']  

     
atexit.register(printonExit)

print ("Find accuracy for different distance type and k value by selecting  features")
print ("Enter one of following the distance matrix to be used:  0 for manhattan','1 for euclidean','2 for minkowski','3 for cosine','4 for canberra','5 for lorentzian','6 for sorensen','7 for hamming','8 for JaccardModif")
for line in sys.stdin:
    distance= int(line);
    break;
    
print ("Enter one of following the weight method to be used: '0 for uniform','1 for distance','2 for distancesquare'")
for line in sys.stdin:
    weight=int (line);
    break;

#_features = [53, 332, 572, 32, 333, 618, 629, 102, 234, 601, 393, 600, 531, 676, 523, 392, 0, 1]
#X.columns = range(X.shape[1])
#y.columns = range(y.shape[0])
MSKNN(X,y,distance,weight,kMax=11,bfeatures=1,fNum=20)


#### K 3-10 and different distance for Entrire Feature (Binary)
#print ("Find accuracy fro different distance type and k value for binary class")
#for uy in np.unique(y):
#     nY = (y==uy)*1  #Make it Binary class
#     print(uy+":"+str(sum(nY))+"Sample")
#     KNN(np.array(X),np.array(y),12)      
#    
#
### For 15 features find best distance, Feature, K , Time, Accuracy, precision (Binary Class)
#print ("Find accuracy for different distance type and k value by selecting 15 features for binary class")
#for uy in np.unique(y):
#     nY = (y==uy)*1  #Make it Binary class
#     print(uy+":"+str(sum(nY))+"Sample")
#     MSKNN(X,y,kMax=10,bfeatures=1,fNum=20)
#     
#start = time. time()
#end = time. time()
#General KNN method which select best k, Classfier, 