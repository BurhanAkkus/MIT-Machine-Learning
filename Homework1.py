import numpy as np
'''A = np.array([[1,2],[3,4],[5,6],[7,8]])

print(A[:,1:2])


def positive(x, th, th0):
    if np.matmul(np.transpose(th),x) + th0 > 0:
        return  1
    elif np.matmul(np.transpose(th),x) + th0 == 0:
        return 0
    return -1
    pass

print(positive([1,-6],[1,1],5))

data = np.transpose(np.array([[1, 2], [1, 3], [2, 1], [1, -1], [2, -1]]))
labels = np.array([-1, -1, +1, +1, +1])

print(data)

positive = [[1, 1, 1, -1, -1]]

result = labels == positive

print(result)
'''

ths = np.array([[0.98645534, -0.02061321, -0.30421124, -0.62960452, 0.61617711, 0.17344772, -0.21804797, 0.26093651, 0.47179699, 0.32548657], [0.87953335, 0.39605039, -0.1105264, 0.71212565, -0.39195678, 0.00999743, -0.88220145, -0.73546501, -0.7769778, -0.83807759]])
th0s = np.array([[0.65043158, 0.61626967, 0.84632592, -0.43047804, -0.91768579, -0.3214327, 0.0682113, -0.20678004, -0.33963784, 0.74308104]])
data = np.array([[1, 1, 2, 1, 2], [2, 3, 1, -1, -1]])
labels = np.array([[-1, -1, 1, 1, 1]])

def interpret(x, th, th0):
    return np.matmul(np.transpose(th),x) + np.transpose(th0)
def sign(x,th,th0):
    return np.where(interpret(x, th, th0) >= 0, 1, -1)


def best_separator(data, labels, ths, th0s):
    bestSeperatorIndex = np.argmax(np.sum(labels == np.sign(interpret(data, ths, th0s)), axis=1))

    best_th = ths[:, bestSeperatorIndex].reshape(-1, 1)
    best_th0 = th0s[:, bestSeperatorIndex].reshape(-1, 1)
    combined = [best_th, best_th0]
    return combined
