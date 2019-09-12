import monkdata as m
import dtree as d
import random
from matplotlib import pyplot as plt

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
bestErrorForFractions = []
for fraction in fractions:
    test, validation = partition(m.monk1test, fraction)
    t=d.buildTree(m.monk1, m.attributes);
    pruned = t
    iteration = 0
    repeat = True
    lastCorrectPruned = 0
    while repeat:
        print(iteration)
        allPruned = d.allPruned(pruned)
        maxCorrect = 0

        for prunedTree in allPruned:
            correct = d.check(prunedTree, validation)
            if correct >= maxCorrect:
                maxCorrect = correct
                bestPruned = prunedTree
        correctNormal = d.check(t, test)
        correctPruned = d.check(bestPruned, validation)
        print("Iteration {} default tree error: {}, pruned error: {}".format(iteration, 1-correctNormal, 1-correctPruned))
        iteration = iteration + 1
        if lastCorrectPruned >= correctPruned:
            repeat = False
        else:
            pruned = bestPruned
            lastCorrectPruned = correctPruned
    bestErrorForFractions.append(1-lastCorrectPruned)

print(fractions)
print(bestErrorForFractions)
plt.plot(fractions, bestErrorForFractions)
plt.show()

#print("Monk1: ")
#for attribute in m.attributes:
#    print("{} {}".format(attribute.name, d.averageGain(m.monk1, attribute)))
#
#print("Monk2: ")
#for attribute in m.attributes:
#    print("{} {}".format(attribute.name, d.averageGain(m.monk2, attribute)))
#
#print("Monk3: ")
#for attribute in m.attributes:
#    print("{} {}".format(attribute.name, d.averageGain(m.monk3, attribute)))
