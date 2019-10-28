# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 17:25:25 2019

@author: mt01034
"""

import pandas as pd
import numpy as np

data = pd.read_csv('input_for_classifiers.tsv', delimiter='\\t')

X=data[data.columns[0:-1]] #Without last column
Y=data[data.columns[-1]] #Getting Last Column Pathogen


uY = np.unique(Y)
select =  [len(data[Y==path])>5 for path in uY]
sel = uY[find(select)]

sY = Y[Y.isin(sel)]
sX = X[Y.isin(sel)]

for i, colname in enumerate(sX.columns):
    if i<4: #skipping first 3 columns
        continue;
    sum = 0;
    for j, value in enumerate(sX[colname]):
        if value==1:
            sum = value
            break
    if (sum == 0):
        sX.drop([colname], axis = 1, inplace = True)
 

#print ('Pathogens with more than 4 experiments:')
#print (unique(sY)) # Pathogens with more than 4 experiments
#print ('Number of pathogens with more than 4 experiments:')
#print (len(unique(sY)))
sX.insert(len(sX.columns),"Pathogens",sY, allow_duplicates = False)

sX.to_csv("output_filename.csv", index=False, encoding='utf8')
