import numpy as np

def sd(x,th,th0):
    return (np.dot(np.transpose(th),x) + th0) / np.sum(np.dot(np.transpose(th),th)) **0.5

def margin(x,y,th,th0):
    return y * sd(x,th,th0)

def margin_sum(x,y,th,th0):
    return np.sum(margin(x,y,th,th0))
def margin_min(x,y,th,th0):
    return np.min(margin(x,y,th,th0))
def margin_max(x,y,th,th0):
    return np.max(margin(x,y,th,th0))

data = np.array([[1, 2, 1, 2, 10, 10.3, 10.5, 10.7],
                 [1, 1, 2, 2,  2,  2,  2, 2]])
labels = np.array([[-1, -1, 1, 1, 1, 1, 1, 1]])
blue_th = np.array([[0, 1]]).T
blue_th0 = -1.5
red_th = np.array([[1, 0]]).T
red_th0 = -2.5

seperators = [[blue_th,blue_th0],[red_th,red_th0]]
for seperator in seperators:
    print(margin_sum(data,labels,seperator[0],seperator[1]))
    print(margin_min(data,labels,seperator[0],seperator[1]))
    print(margin_max(data,labels,seperator[0],seperator[1]))
