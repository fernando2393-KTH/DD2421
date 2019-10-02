import numpy, random , math
from scipy . optimize import minimize
import matplotlib . pyplot as plt

numpy.random.seed(100)

classA = numpy.concatenate (
    (numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5],
    numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
classB = numpy.random.randn(20, 2) * 0.2 + [0.0, -0.5]

inputs = numpy.concatenate((classA, classB))
targets = numpy.concatenate(
    (numpy.ones(classA.shape[0]), 
    -numpy.ones(classB.shape[0])))

N = inputs.shape[0] # Number of rows (samples)

permute = list (range(N))
random.shuffle(permute) 
inputs = inputs[permute, :]
targets = targets[permute]

# PLOTTING

plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
plt.axis('equal') # Force same scale in both axis
plt.savefig('svmplot.pdf') # Save copy in pdf file
plt.show() # Show the plot



# CONSTANTS

dimension = 2
C = 1
dataset = [] # List of dataset elements

def kernel(x, y):
    p = 2
    return (numpy.dot(x,y)+1)**p

for i in range(N):
    dataset.append([inputs[i], targets[i]])

kernel_matrix = numpy.zeros([N,N])

for i in range(N):
    for j in range (N):
        kernel_matrix[i][j] = kernel(dataset[i][0], dataset[j][0])

# FUNCTIONS

def zeroFun(alphas):

    result = 0

    for i, alpha in enumerate(alphas):
        result += alpha * dataset[i][1]

    return result

def objective(vector):

    result = 0

    for i, alphai in enumerate(vector):
        for j, alphaj in enumerate(vector):
            result += alphai * alphaj * dataset[i][1] * dataset[j][1] * kernel_matrix[i][j]

    result = result/2
    result -= sum(vector)

    return result

def nonZeroAlpha(alphas):

    nonzeroalphas = []
    
    for i, alpha in enumerate(alphas):
        if (alpha > (10**(-5))):
            nonzeroalphas.append([alpha, i])

    return nonzeroalphas

def bComputation(alphas):

    b = 0
    support_vector = []

    for i, alpha in enumerate(alphas):
        if (alpha < C):
            support_vector = [dataset[i], i]
            break

    for i, alpha in enumerate(alphas):
        b += alpha * dataset[i].y * kernel_matrix[support_vector[1]][i] - dataset[support_vector[1]][1]

    return b

def indicatorFunction(alphas, s):

    result = 0
    b = bComputation(alphas)
    
    for i, alpha in enumerate(alphas):
        result += alpha * dataset[i].y * kernel_matrix[s.y][i] - b

    return result
        
# MAIN CODE

ret = minimize(objective, numpy.zeros(N), bounds=[(0, C) for b in range(N)], constraints={'type':'eq', 'fun':zeroFun})

alpha = nonZeroAlpha(ret['x'])
    