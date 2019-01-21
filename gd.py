from numpy import *
import util
import matplotlib.pyplot as plot
import pylab

def gd(func, grad, x0, numIter, stepSize):
    """
    func(x) = funtion on which gradient decesnt is to be performed.
    grad(x) = derivative of func(x)
    x0 = starting position
    numIter = number of iterations.
    stepSize = initial step size.

    returns final solution and the trajectory
    """
    x = x0
    trajectory = zeros(numIter + 1)
    trajectory[0] = func(x)

    for iter in range(numIter):
        g = grad(x) 
        eta = stepSize/(sqrt(iter+1))
        x = x - g*eta 
        trajectory[iter+1] = func(x)

    return (x, trajectory)