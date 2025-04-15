import numpy
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


def num_grad(f, delta=0.001):
    """
    Returns a function that calculates the numerical gradient of f using
    central differences. The function f should take a column vector (a 2D numpy array
    with shape (n, 1)) and return a scalar value.

    Parameters:
      f     : The objective function.
      delta : The small constant for finite differences (default is 0.001).

    Returns:
      A function that takes a column vector x and returns the estimated gradient,
      also as a column vector.
    """

    def grad(x):
        # Create an array to store the gradient estimates, preserving the shape of x.
        grad_est = np.zeros_like(x)
        n = x.shape[0]  # number of dimensions

        # Compute the gradient for each component using central differences.
        for i in range(n):
            # Create a perturbation vector that is all zeros except in the i-th position.
            perturb = np.zeros_like(x)
            perturb[i, 0] = delta  # increment only the i-th component by delta

            # Calculate the function values at x+perturb and x-perturb.
            f_plus = f(x + perturb)
            f_minus = f(x - perturb)

            # Estimate the i-th component of the gradient.
            grad_est[i, 0] = (f_plus - f_minus) / (2 * delta)

        return grad_est

    return grad

def minimize(f, x0, step_size_fn, max_iter):
    xs = [x0]
    x = x0
    fs = [f(x)]
    df = num_grad(f)
    for iter in range(max_iter):
        gradient = df(x)
        x = x - gradient * step_size_fn(iter + 1)
        xs.append(x)
        fs.append(f(x))

    return x, fs, xs

def hinge(v):
    return np.maximum(0,1-v)

# x is dxn, y is 1xn, th is dx1, th0 is 1x1
def hinge_loss(x, y, th, th0):
    return hinge(y* (np.dot(np.transpose(th),x) + th0))
    pass

# x is dxn, y is 1xn, th is dx1, th0 is 1x1, lam is a scalar
def svm_obj(x, y, th, th0, lam):
    average = np.average(hinge_loss(x,y,th,th0))
    regularizor = lam * np.linalg.norm(th) ** 2
    return average + regularizor

def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

sep_e_separator = np.array([[-0.40338351], [1.1849563]]), np.array([[-2.26910091]])

# Test case 1
x_1, y_1 = super_simple_separable()
th1, th1_0 = sep_e_separator
ans = svm_obj(x_1, y_1, th1, th1_0, .1)
print(ans)
# Test case 2
ans = svm_obj(x_1, y_1, th1, th1_0, 0.0)
print(ans)


################################################

# Returns the gradient of hinge(v) with respect to v.
def d_hinge(v):
    return np.where(v < 1, -1, 0)

# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th
def d_hinge_loss_th(x, y, th, th0):
    return d_hinge(y * (np.dot(np.transpose(th),x) + th0)) * (y * x)


# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th0
def d_hinge_loss_th0(x, y, th, th0):
    return d_hinge(y * (np.dot(np.transpose(th),x) + th0)) * y

# Returns the gradient of svm_obj(x, y, th, th0) with respect to th
def d_svm_obj_th(x, y, th, th0, lam):
    average = np.vstack(np.average(d_hinge_loss_th(x,y,th,th0),axis=1))
    regularizor = lam * 2 * th
    return average + regularizor

# Returns the gradient of svm_obj(x, y, th, th0) with respect to th0
def d_svm_obj_th0(x, y, th, th0, lam):
    average = np.average(d_hinge_loss_th0(x,y,th,th0))
    return np.array([average]).reshape(-1, 1)

# Returns the full gradient as a single vector (which includes both th, th0)
def svm_obj_grad(X, y, th, th0, lam):
    return np.vstack([d_svm_obj_th(X,y,th,th0,lam),d_svm_obj_th0(X,y,th,th0,lam)])

X1 = np.array([[1, 2, 3, 9, 10]])
y1 = np.array([[1, 1, 1, -1, -1]])
th1, th10 = np.array([[-0.31202807]]), np.array([[1.834     ]])
X2 = np.array([[2, 3, 9, 12],
               [5, 2, 6, 5]])
y2 = np.array([[1, -1, 1, -1]])
th2, th20=np.array([[ -3.,  15.]]).T, np.array([[ 2.]])

print(d_hinge(np.array([[ 71.]])).tolist())
print(d_hinge(np.array([[ -23.]])).tolist())
print(d_hinge(np.array([[ 71, -23.]])).tolist())

print(d_hinge_loss_th(X2[:,0:1], y2[:,0:1], th2, th20).tolist())
print(d_hinge_loss_th(X2, y2, th2, th20).tolist())
print(d_hinge_loss_th0(X2[:,0:1], y2[:,0:1], th2, th20).tolist())
print(d_hinge_loss_th0(X2, y2, th2, th20).tolist())

print(d_svm_obj_th(X2[:,0:1], y2[:,0:1], th2, th20, 0.01).tolist())
print(d_svm_obj_th(X2, y2, th2, th20, 0.01).tolist())
print(d_svm_obj_th0(X2[:,0:1], y2[:,0:1], th2, th20, 0.01).tolist())
print(d_svm_obj_th0(X2, y2, th2, th20, 0.01).tolist())

print(svm_obj_grad(X2, y2, th2, th20, 0.01).tolist())
print(svm_obj_grad(X2[:,0:1], y2[:,0:1], th2, th20, 0.01).tolist())