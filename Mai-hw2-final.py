 # course: TCSS455
# Homework 2
# date: 01/22/2018
# name: Martine De Cock
# description: Training and testing decision trees with discrete-values attributes

import sys
import math
import pandas as pd
import operator
from collections import Counter
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
    class_list = examples.loc[:,target].tolist()
    # Case 1: all remaining examples belong to one class => perfectly classified
    if class_list.count(class_list[0]) == len(class_list):
        return DecisionNode(class_list[0])
    # Case 2: mixed classes but no more attribute to split on
    # Then we return the major class left
    elif len(attributes) == 0:  
        return DecisionNode(count_majority(class_list))
    
    # Case 3: else, build the tree recursively
    else:
        best = choose_best_attribute(examples, attributes, target)
        best_attribute_label = attributes[best]
        tree = DecisionNode(best_attribute_label)
        vals = get_values(examples, attributes, best_attribute_label)
        for val in vals:
            new_examples = split_on_val(examples, best_attribute_label, val)
            newAttr = attributes[:]
            newAttr.remove(best_attribute_label)
            subtree = id3(new_examples, target, newAttr)
            tree.children[val] = subtree
            
    return tree
    
# Function to calculate entropy of a given dataset
# GOOD TO GO
def entropy(examples, target):
    N = len(examples)
    # Count all the unique values of the target and their occurences
    counter = Counter(examples.loc[:,target])
    return sum(-1.0*(counter[k] / N)*math.log(counter[k] / N, 2) for k in counter)

# Function to extract subset from examples, where the best attribute has 'val' value
# GOOD TO GO
def split_on_val(examples, best, val):
    new_data = examples.loc[examples[best] == val]
    del new_data[best]
    return new_data

# Calculate the info gain by splitting on a chosen attribute
# GOOD TO GO
def info_gain(examples, attributes, attr, target):
    base_entropy = entropy(examples, target)
    # Find all unique values of the chosen attribute and their number of occurence
    counter = Counter(examples.loc[:, attr])
    new_entropy = 0.0
    for key in counter:
        # Probability of a unique value
        prob = counter[key] / sum(counter.values())
        new_entropy += prob * entropy(split_on_val(examples, attr, key), target)
    return (base_entropy - new_entropy)

# Return an integer index of the best attribute to split on
def choose_best_attribute(examples, attributes, target):
    best_info_gain = 0.0
    best = -1
    # List thru all the features to find the best feat
    # based on info gain
    for i in range(len(attributes)):
        new_gain = info_gain(examples, attributes, attributes[i], target)
        if new_gain > best_info_gain:
            best_info_gain = new_gain
            best = i
    # Return an integer index of the attribute
    return best
    
def count_majority(classes):
    classCount = {}
    for cla in classes:
        if cla not in classCount.keys(): classCount[cla] = 1
        classCount[cla] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # return the highest count a.k.a the majority
    return sortedClassCount[0][0]

# Function to get all unique values of an attribute
def get_values(examples, attributes, attr):
    counter = Counter(examples.loc[:, attr])
    values = []
    for key in counter:
        if key not in values:
            values.append(key)
    return values
####################   MAIN PROGRAM ######################

# Reading input data

train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])
target = sys.argv[3]
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
