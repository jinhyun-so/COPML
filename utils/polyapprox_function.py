import numpy as np

def sigmoid(z): # calculates the sigmoid function
    sigma = 1./(1+np.exp(-z)) # exp operates elementwise on vectors
    is_largerthanone = (sigma>0.9999)
    is_zero = (sigma<0.0001)
    sigma = is_largerthanone * 0.9999 + (1-is_largerthanone)*sigma
    sigma = is_zero*0.0001 + (1-is_zero)*sigma
    
    return sigma
    
def polyapp(samples): # polynomial (degree-2) approximation for the sigmoid function
    rangepoly = 10 # fit values bw [-range, range]  
    xpoly = - rangepoly * np.ones((samples,1)) + 2*rangepoly * np.random.rand(samples, 1)
    A = np.empty((samples,3)) # for degree 2 polynomial
    ypoly = np.empty((samples,1)) # y values for the polynomial
        
    for j in range(samples):
        A[j] = [1, xpoly[j], (xpoly[j])**2]
    ypoly = sigmoid(xpoly)
        
    Apinv = np.linalg.pinv(A) # pseudoinverse of A
    coeffs = Apinv.dot(ypoly)
        
    return coeffs


def computepoly(x, coeffs): # compute the degree-2 polynomial for the given coefficients, takes a vector as input
        
    out  = coeffs[0] + coeffs[1] * x + coeffs[2] * (x**2) # specific for the degree-2 polynomial

    return out

def test_function(y_hat, test_image):
    
    flag = y_hat - 0.5
    y_label = (abs(np.sign(flag)) + np.sign(flag))/2
    tmp = test_image.reshape(len(test_image),1)
    num_Error = np.sum(abs(y_label - tmp))

    accuracy = 1 - float(num_Error)/float(len(test_image))

    return accuracy*100

