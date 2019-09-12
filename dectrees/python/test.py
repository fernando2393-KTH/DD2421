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
    allPruned = d.allPruned(tree)
    maxCorrect = 0
    for prunedTree in allPruned:
        correct = d.check(prunedTree, validation)
        print("correct: {}".format(correct))
        if correct >= maxCorrect:
            maxCorrect = correct
            bestPruned = prunedTree
    return bestPruned, maxCorrect

# returns the complete pruning of the tree and the correspondent 1-error (correct)
def prune_tree(dataset, fraction):
    test, validation = partition(dataset, fraction)
    t=d.buildTree(dataset, m.attributes);
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


#fractions = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
fractions = np.array([0.3])
iterations = 1
bestErrorForFractions = np.zeros((fractions.shape[0], iterations))
for i in range(iterations):
    for index, fraction in enumerate(fractions):
        pruned, correct = prune_tree(m.monk1, fraction)
        print("Correct {}".format(correct))
        bestErrorForFractions[index][i] += (1-correct)

bestErrorForFractions /= iterations
print(bestErrorForFractions)
plt.plot(fractions, bestErrorForFractions)
#plt.show()
