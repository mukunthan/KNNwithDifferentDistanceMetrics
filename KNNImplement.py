
import numpy as np
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier

#Minkowski distance  manhattan=1, euclidean=2
class MyKNeighborsClassifier(KNeighborsClassifier):
    def __init__(self, n_neighbors=5,weights="uniform",distancetype="euclidean",p=4):
        self.delta= 0.000000000000000000001
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.distancetype=distancetype;
        self.p= p


    def fit(self, X, y):
        self.X = X
        self.y = y
        return self


    def predict(self, X):
       # print ("predict")
        return [self._predict_oneset(i) for i in X]

    def _predict_oneset(self, test):
            distances = sorted((self._distance(x, test), y) for x, y in zip(self.X, self.y))
            neighbors = distances[:self.n_neighbors]
            weights = self._compute_weights(neighbors)
            weights_by_class = defaultdict(list)
            for d, c in weights:
                weights_by_class[c].append(d)
            
            MaxKey=max((sum(val), key) for key, val in weights_by_class.items())[1]
            return MaxKey
        
    
    
    def _distance(self, data1, data2):
        if self.distancetype == "manhattan": #manhattan
            #print ("manhattan")
            return sum(abs(data1 - data2))
            
        elif self.distancetype == "euclidean":
            #print ("Euclidean")
            return np.sqrt(sum((data1 - data2) ** 2))
            
        elif self.distancetype == "minkowski":
            return self._nth_root((sum((data1 - data2) ** self.p)),self.p)
        
        elif self.distancetype == "cosine":
            numerator = sum(a*b for a,b in zip(data1,data2))
            denominator = self._squareroot_of_squaresum(data1)*self._squareroot_of_squaresum(data2) +self.delta
            return (1 - round(numerator/float(denominator),3))
        

        elif self.distancetype == "jaccard":
            intersection_cardinality = len(set.intersection(*[set(data1), set(data2)]))
            union_cardinality = len(set.union(*[set(data1), set(data2)])) +self.delta
            return 1-(intersection_cardinality/float(union_cardinality))
        
        elif self.distancetype == "canberra":
            numerator = abs(data1 - data2)
            denominator = abs(data1) +abs(data2) +self.delta
            return round(sum(numerator/denominator),3)
            
        elif self.distancetype == "lorentzian":
             return sum(np.log1p(1+abs(data1 - data2)))
        
        elif self.distancetype == "sorensen":
            numerator = sum(abs(data1 - data2))
            denominator = sum(data1 + data2) +self.delta
            return round((numerator/float(denominator)),3)
        
        elif self.distancetype == "dice":
            numerator = 2*sum(a*b for a,b in zip(data1,data2))
            denominator = sum(data1 **2) + sum(data2 **2) +self.delta
            return (1 - round(numerator/float(denominator),3))
        
        elif self.distancetype == "hamming": 
            #print ("hamming")
            return (sum(data1 != data2))/float(len(data1))
        
        elif self.distancetype == "JaccardModif": 
            perfectmatch = sum(data1 == data2)
            union_cardinality = (2*len(data1)) - perfectmatch
            return 1-(perfectmatch/float(union_cardinality))
            
        raise ValueError("distancetype not recognized: should be  'manhattan' or 'euclidean' or 'minkowski' or 'cosine' or 'jaccard' or 'canberra' or 'lorentzian' or 'sorensen' or hamming or 'JaccardModif'")

    def _nth_root(self,value, n_root):
        root_value = 1/float(n_root)
        return round ((value ** root_value),3)  
        
    def _squareroot_of_squaresum(self,x):
        return round(np.sqrt(sum([a*a for a in x])),3) 
        
        
    def _compute_weights(self, distances):
        if self.weights == 'uniform':
             return [(1, y) for d, y in distances]
             
        elif self.weights == 'distance':
            matches = [(1, y) for d, y in distances if d == 0]
            return matches if matches else [(1/d, y) for d, y in distances]
            
        elif self.weights == 'distancesquare':
            matches = [(1, y) for d, y in distances if d == 0]
            return matches if matches else [(1/(d **2), y) for d, y in distances]
        
        raise ValueError("weights not recognized: should be 'uniform' or 'distance'")
    
    
    def score(self, X, y):
       # print ("score")
        return sum(self.predict(X) == y) / len(y)


#X_train = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 0], [1, 0, 1]])
#y_train = np.array([1,1,1,0,0,0])
#neighbor = MyKNeighborsClassifier(n_neighbors=3,weights='distance',distancetype ='JaccardModif',p=1)
#neighbor.fit(X_train, y_train)
#X_test = np.array([[1, 0, 1], [-2, -2, 1]])
#print(neighbor.predict(X_test))
#
#X = np.array([[1, 1], [4, 4], [5, 5]])
#y = np.array([1,0,0])
#neighbor = KNeighborsClassifier(n_neighbors=3,weights='distance',p=2).fit(X, y)

