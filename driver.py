from DecisionTree import *
import pandas as pd
from sklearn import model_selection

""" header for the iris dataset"""
header = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']

""" header for the car dataset"""
# header = ['Buying', 'Maint', 'Doors', 'Persons', 'Lug_boot', 'Safety', 'Class']

"""local car dataset"""
# df = pd.read_csv('/Users/aditya/Downloads/FW__Report_/car.csv', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])
""" local Iris dataset"""
# df = pd.read_csv('/Users/aditya/Downloads/iris.csv', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])

"""
*****UCI IRIS DATASET*****
"""
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])
lst = df.values.tolist()
t = build_tree(lst, header)
print_tree(t)


print("********** Leaf nodes ****************")
leaves = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t)
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))

trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header)
print("*************Tree before pruning*******")
print_tree(t)
acc = computeAccuracy(test, t)
beforePruning = acc
print("Accuracy on test = " + str(acc))


## TODO: You have to decide on a pruning strategy
# t_pruned = prune_tree(t, [26, 11, 5])
pruneList = [64]
t_pruned = prune_tree(t, pruneList)
print("*************Tree after pruning*******")
print_tree(t_pruned)

pleaves = getLeafNodes(t_pruned)

pinnerNodes = getInnerNodes(t_pruned)


acc = computeAccuracy(test, t)
afterPruning =acc
print("Accuracy on test = " + str(acc))

print("********** Tree Before Pruning ****************")

print("**********  Accuracy on test = " + str(beforePruning))


print("**********  Accuracy on test before Pruning = " + str(beforePruning))

print("********** pruneNodes ------> "+ str(pruneList))

print("********** Tree After Pruning ****************")

print("**********  Accuracy on test = " + str(afterPruning))


print("**********  Accuracy on test after Pruning = " + str(afterPruning))