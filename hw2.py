# course: TCSS455
# Homework 2
# date: 01/22/2018
# name: Martine De Cock
# description: Training and testing decision trees with discrete-values attributes

import sys
import math
import pandas as pd
import operator

class DecisionNode:

    # A DecisionNode contains an attribute and a dictionary of children. 
    # The attribute is either the attribute being split on, or the predicted label if the node has no children.
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}

    # Visualizes the tree
    def display(self, level = 0):
        if self.children == {}: # reached leaf level
            print(": ", self.attribute, end="")
        else:
            for value in self.children.keys():
                prefix = "\n" + " " * level * 4
                print(prefix, self.attribute, "=", value, end="")
                self.children[value].display(level + 1)
     
    # Predicts the target label for instance x
    def predicts(self, x):
        if self.children == {}: # reached leaf level
            return self.attribute
        value = x[self.attribute]
        subtree = self.children[value]
        return subtree.predicts(x)


# Illustration of functionality of DecisionNode class
def funTree():
    myLeftTree = DecisionNode('humidity')
    myLeftTree.children['normal'] = DecisionNode('no')
    myLeftTree.children['high'] = DecisionNode('yes')
    myTree = DecisionNode('wind')
    myTree.children['weak'] = myLeftTree
    myTree.children['strong'] = DecisionNode('no')
    return myTree


def id3(examples, target, attributes):
    # Checking if end cases:
    class_list = (pd.unique(examples.iloc[:,-1])).tolist()
    # Case 1: all remaining examples belong to one class => perfectly classified
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # Case 2: mixed classes but no more attribute to split on
    if (len(attributes) == 1):   
        return count_majority(class_list)
    
    # Else recursively build the tree
    best_attribute = choose_best_feature(train, attributes, target)
    
    best_attribute_label = attributes[best_attribute]
    print(best_attribute_label)
    myTree = DecisionNode(best_attribute_label)
    
    uniqChildren = examples[best_attribute_label].unique()
    del (attributes[best_attribute])

    for children in uniqChildren:
        sub_attributes = attributes[:]
        myTree.children[children] = id3(split_set(examples, best_attribute, children), target, sub_attributes)
    
    return myTree
    
# Function to calculate entropy of a given dataset:
def entropy(examples, axis):
    freq = {}
    dataEntropy = 0.0

    for example in examples:
        # If the label's value exist, update
        if example[-1] in freq:
            example[entry[-1]] += 1.0
        # Else, add to the list
        else:
            example[entry[-1]]  = 1.0
    # Calculate entropy for each label's value and add them up
    for freq in freq.values():
        dataEntropy += (-freq/len(examples)) * math.log(freq/len(examples), 2) 
    return dataEntropy

# Split tree based on a value
def split_set(examples, axis, val):
    splitted = []
    for feature in examples:
        # remove the row:
        if feature[axis] == val:
            reduced = feature[ : axis]
            reduced = feature[axis + 1 :]
            splitted.append(reduced)
    df = pd.DataFrame(splitted)      
    return df
    
def choose_best_feature(examples, attributes, target):
    baseEntropy =  entropy(examples, -1)
    best_info_gain = 0.0
    best_feature = -1
    # List thru all the features to find the best feat
    # based on info gain
    for i in range(len(attributes)):
        feature_list = [example[i] for example in examples]
        # get all possible values of the feature
        uniqVals = set(feature_list)
        newEntropy = 0.0
        # calculate entropy if split on that feature
        for val in uniqVals:
            subset = split_set(examples, i, val)
            prob = len(subset)/len(examples)
            newEntropy += prob * entropy(subset, -1)
        info_gain =  baseEntropy - newEntropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature
    
def count_majority(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # return the highest count aka the majority
    return sortedClassCount[0][0]


####################   MAIN PROGRAM ######################

# Reading input data
'''
train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])
target = sys.argv[3]
attributes = train.columns.tolist()
attributes.remove(target)
'''
train = pd.read_csv('playtennis_train.csv')
test = pd.read_csv('playtennis_test.csv')
target = 'playtennis'
attributes = train.columns.tolist()
attributes.remove(target)

# Learning and visualizing the tree
tree = id3(train,target,attributes)
tree.display()
# Evaluating the tree on the test data
correct = 0
for i in range(0,len(test)):
    if str(tree.predicts(test.loc[i])) == str(test.loc[i,target]):
        correct += 1
print("\nThe accuracy is: ", correct/len(test))