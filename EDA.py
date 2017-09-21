# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 20:59:32 2016
Python EDA for the kaggle allstate competition to look at correlations and uniques

@author: Brian
"""

import os
import pandas as pd
import numpy as np

### charting
from matplotlib import pyplot as plt
import seaborn as sns # to make sharts prettier

#### standard functions
def Check_Uniques(frame):
    sizes = []
    low_var_col = []
    for column in frame:
        data_sets = frame[column]
        uniques = data_sets.unique() # note the brackets
        # print the pivot to show distributions
        # will error
        if (column != 'id'):
            pd_Table = pd.pivot_table(train, values = 'id', index=column, aggfunc='count').apply(lambda x: np.round(np.float(x)/len(train)*100, 2))
            print (pd_Table )
            if max(pd_Table) > 99:
                low_var_col.append(column)
        #variance = np.var(data_sets)
        #print(uniques.size)
        #print(column)
        
        sizes.append(uniques.size)
    #print(len(uniques))
    return(sizes, low_var_col)

def Categorical_Processing(frame):
    
    for column in frame:
        frame[column] = frame[column].astype('category')
        print(frame[column].describe())
    print(frame.describe())
    return frame
    
### check dist of the continuous
def Check_Distrib(frame):
    
    list_of_hist = []
    for column in frame:
        hist, bins = np.histogram(frame[column], bins = 50)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        plt.show()
        list_of_hist.append([hist, bins])
        
    return list_of_hist

def correlation_plot(pandas_correlation_frame, max_Shade = 0.95):
    mask = np.zeros_like(pandas_correlation_frame, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
    map_obj = sns.heatmap(pandas_correlation_frame, annot=True, fmt="0.2f", mask=mask, cmap=cmap, vmax=max_Shade,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    
    return map_obj
    
data_path = 'Data'
os.listdir('./Data/')
train = pd.read_csv('./Data/train.csv')

### 2 tyoes if columns the continuous and the categorical
# lets separate these

cate_cols = [col for col in train.columns if 'cat' in col]
continuous_cols = [col for col in train.columns if 'cont' in col]

categorical = train[cate_cols]

### make into categories

continuous = train[continuous_cols]

## id and target columns
id_target = train[['id', 'loss']]


sizes, low_var_cols_to_drop = Check_Uniques(categorical)
categorical_to_keep = categorical.drop(low_var_cols_to_drop, axis = 1)

Check_Uniques(continuous)
## corr plot
cp = continuous.corr()
correlation_plot(cp, 1)
### correlation plot
# Generate a mask for the upper triangle





#### check the loss func
### check the column that we are trying to predict
hist, bins = np.histogram(np.log(train['loss']), bins = 50)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()

np.var([0,0,0,0,0,0,1])

