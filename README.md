# Machine-Learning-ID3-Implementation
# This a exercise in a Machine learning course.
One classical tool for classification in Machine Learning is using the ID3 algorithm to build a decision tree.
Tree is built with a greedy aproach by continously calculate the Entropy and Info gain to choose which Attribute/Feature will be the decision node at a certain level. The algorithm will stop when all the training examples belong to one class (perfectly classified), or when there's no more attribute to split. In the second case, the leaf node decision is the majority class in the remaining example.
