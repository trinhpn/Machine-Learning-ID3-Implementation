# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 02:06:05 2018

@author: Trinh
"""
import pandas as pd
import sys
import math
import operator
from collections import Counter
def main():
    train = pd.read_csv('playtennis_train.csv')
    test = pd.read_csv('playtennis_test.csv')
    target = 'playtennis'
    attributes = train.columns.tolist()
    attributes.remove(target)
    print(choose_best_attribute(train, attributes, target))
    
    
    
def id3(examples, target, attributes):
    # Checking if end cases:
    class_list = [example[-1] for example in examples]
    # Case 1: all remaining examples belong to one class => perfectly classified
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # Case 2: mixed classes but no more attribute to split on
    if len(dataSet[0]) == 1:  
        return count_majority(class_list)
    
    
    # Else recursively build the tree
    best_attribute = choose_best_feature(train, attributes)
    best_attribute_label = attributes[best_attribute]
    
    myTree = DecisionNode(best_attribute_label)
    del (attributes[best_attribute])
    children = [examples[best_attribute] for example in examples]
    uniqChildren = set(children)
    for children in uniqChildren:
        sub_attributes = attributes[:]
        myTree.children[children] = id3(split_set(examples, best_attribute, children)\
                       , target, sub_attribute)
    return myTree
    
# Function to calculate entropy of a given dataset
# GOOD TO GO
def entropy(examples, target):
    N = len(examples)
    counter = Counter(examples.loc[:,target])
    return sum(-1.0*(counter[k] / N)*math.log(counter[k] / N, 2) for k in counter)

# Function to extract subset from examples, where the best attribute has 'val' value
# GOOD TO GO
def split_on_val(examples, attributes, best, val, target):
    new_data = [[]]
    i = attributes.index(best)

    for row in examples.itertuples(index=False, name = 'Panda') :
        if (row[i] == val):
            new_row = []
            for k in range(0,len(row)):
                if(k != i):
                    new_row.append(row[k])
            new_data.append(new_row)

    new_data.remove([])
    new_data = pd.DataFrame(new_data)    
    header = attributes[:]
    del header[i]
    header.append(target)
    new_data.columns = header
    return new_data


# Calculate the info gain by splitting on a chosen attribute
# GOOD TO GO
def info_gain(examples, attributes, attr, target):
    base_entropy = entropy(examples, target)
    # Find all unique values of the chosen attribute and their number of occurence
    counter = Counter(examples.loc[:, attr])
    new_entropy = 0.0
    for key in counter:
        prob = counter[key] / sum(counter.values())

        new_entropy += prob * entropy(split_on_val(examples, attributes, attr, key, target)\
                    , target)
    return (base_entropy - new_entropy)
    
def choose_best_attribute(examples, attributes, target):
    best_info_gain = 0.0
    best = -1
    # List thru all the features to find the best feat
    # based on info gain
    for i in range(len(attributes)):
        new_gain = info_gain(examples, attributes, attributes[i], target)
        print(new_gain)
        if new_gain > best_info_gain:
            best_info_gain = new_gain
            best = i
    return best
    
def count_majority(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # return the highest count aka the majority
    return sortedClassCount[0][0]
    
    
if __name__ == "__main__":
    main()   