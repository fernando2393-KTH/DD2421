import monkdata as m
import dtree as d
import random
from matplotlib import pyplot as plt
import numpy as np

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

# From a tree and a validation set, returns the best tree pruned of 1 node and the correspondent 1-error
def best_prune(tree, validation):
    maxCorrect = 0
    bestPruned = tree
    for prunedTree in d.allPruned(tree):
        correct = d.check(prunedTree, validation)
        if correct >= maxCorrect:
            maxCorrect = correct
            bestPruned = prunedTree
    return bestPruned, maxCorrect

# returns the complete pruning of the tree and the correspondent 1-error (correct)
def prune_tree(training_dataset, testing_dataset, fraction):
    test, validation = partition(testing_dataset, fraction)
    t=d.buildTree(training_dataset, m.attributes);
    correctDefaultTree = d.check(t, test)
    pruned = t
    iteration = 0
    repeat = True
    lastCorrectPruned = 0
    while repeat:
        bestPruned, maxCorrect = best_prune(pruned, validation)
        if lastCorrectPruned < correctDefaultTree:
            repeat = False
        if lastCorrectPruned >= maxCorrect:
            repeat = False
        else:
            pruned = bestPruned
            lastCorrectPruned = maxCorrect
    return pruned, lastCorrectPruned


fractions = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
iterations = 300
bestErrorForFractions = np.zeros((fractions.shape[0], iterations))
for i in range(iterations):
    for index, fraction in enumerate(fractions):
        pruned, correct = prune_tree(m.monk1, m.monk1test, fraction)
        bestErrorForFractions[index][i] += (1-correct)

print(bestErrorForFractions)
errorVector = [sum(bestErrorForFractions[row,:])/iterations for row in range(len(fractions))]
print(errorVector)
plt.plot(fractions, errorVector)
plt.show()
