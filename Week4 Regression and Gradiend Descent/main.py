import numpy as np

def cv(value_list):
    return np.transpose(rv(value_list))

# Takes a list of numbers and returns a row vector: 1 x n
def rv(value_list):
    return np.array([value_list])

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

def hingeLoss(x,y,th,th0,margin_ref):
    return np.maximum(0,1 - (margin(x,y,th,th0) / margin_ref))

data = np.array([[1, 2, 1, 2, 10, 10.3, 10.5, 10.7],
                 [1, 1, 2, 2,  2,  2,  2, 2]])
labels = np.array([[-1, -1, 1, 1, 1, 1, 1, 1]])
blue_th = np.array([[0, 1]]).T
blue_th0 = -1.5
red_th = np.array([[1, 0]]).T
red_th0 = -2.5

seperators = [[blue_th,blue_th0],[red_th,red_th0]]
'''for seperator in seperators:
    print(margin_sum(data,labels,seperator[0],seperator[1]))
    print(margin_min(data,labels,seperator[0],seperator[1]))
    print(margin_max(data,labels,seperator[0],seperator[1]))
'''

data = np.array([[1.1, 1, 4],[3.1, 1, 2]])
labels = np.array([[1, -1, -1]])
th = np.array([[1, 1]]).T
th0 = -4

#print(hingeLoss(data,labels,th,th0,2**-0.5))


def gradient_descent(f,df,x0,stepsize_fn,max_iter):
    xs = [x0]
    x = x0
    fs = [f(x)]
    for iter in range(max_iter):
        gradient = df(x)
        x = x - gradient * stepsize_fn(iter + 1)
        xs.append(x)
        fs.append(f(x))

    return x,fs,xs

def f1(x):
    return float((2 * x + 3)**2)

def df1(x):
    return 2 * 2 * (2 * x + 3)

def f2(v):
    x = float(v[0]); y = float(v[1])
    return (x - 2.) * (x - 3.) * (x + 3.) * (x + 1.) + (x + y -1)**2

def df2(v):
    x = float(v[0]); y = float(v[1])
    return cv([(-3. + x) * (-2. + x) * (1. + x) + \
               (-3. + x) * (-2. + x) * (3. + x) + \
               (-3. + x) * (1. + x) * (3. + x) + \
               (-2. + x) * (1. + x) * (3. + x) + \
               2 * (-1. + x + y),
               2 * (-1. + x + y)])
def package_ans(gd_vals):
    x, fs, xs = gd_vals
    return [x.tolist(), [fs[0], fs[-1]], [xs[0].tolist(), xs[-1].tolist()]]

# Test case 1
ans=package_ans(gradient_descent(f1, df1, cv([0.]), lambda i: 0.1, 1000))
print(ans)
# Test case 2
ans=package_ans(gradient_descent(f2, df2, cv([0., 0.]), lambda i: 0.01, 1000))
print(ans)

