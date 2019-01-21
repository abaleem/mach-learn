"""
Implementation of *regularized* linear classification/regression by
plug-and-play loss functions
"""

from numpy import *
from pylab import *

from gd import *

class LossFunction:

    def loss(self, Y, Yhat):
        util.raiseNotDefined()

    def lossGradient(self, X, Y, Yhat):
        util.raiseNotDefined()

class SquaredLoss(LossFunction):

    def loss(self, Y, Yhat):
        return 0.5 * dot(Y - Yhat, Y - Yhat)

    def lossGradient(self, X, Y, Yhat):
        return - sum((Y - Yhat) * X.T, axis=1)

class LogisticLoss(LossFunction):
    
    def loss(self, Y, Yhat):
        return sum(log([item+1 for item in exp([-a*b for a,b in zip(Y,Yhat)])]))

    def lossGradient(self, X, Y, Yhat):
        grad = zeros(size(X,1))
        for i in range(len(X)):
            grad = grad + ( (1/(1+exp(-Y[i]*Yhat[i])) * exp(-Y[i]*Yhat[i]) * -Y[i] * X[i,:]))
        return grad

class HingeLoss(LossFunction):

    def loss(self, Y, Yhat):
        return sum([max(0,1-Y[i]*Yhat[i]) for i in range(len(Y))])

    def lossGradient(self, X, Y, Yhat):
        grad = zeros(size(X,1))
        for i in range(len(Y)):
            if(Y[i]*Yhat[i] <= 1):
                grad = grad - Y[i]*X[i,:]
        return grad


class LinearClassifier():
    """
    Linear classifier parameterized by a loss function and a ||w||^2 regularizer.
    """

    def __init__(self, opts):

        self.opts = opts
        self.reset()

    def reset(self):
        self.weights = np.array([0, 0])

    def online(self):
        return False

    def __repr__(self):
        return    "w=" + repr(self.weights)

    def predict(self, X):

        if type(self.weights) == int: 
            print('Weights not defined yet')
            return 0
        else: 
            return dot(self.weights, X)

    def predictAll(self, X):
        N,D = X.shape
        Y   = zeros(N)
        for n in range(N):
            Y[n] = self.predict(X[n,:])
        return Y

    def getRepresentation(self):
        return self.weights

    def train(self, X, Y):

        lossFn   = self.opts['lossFunction']         # loss function to optimize
        lambd    = self.opts['lambda']               # regularizer is (lambd / 2) * ||w||^2
        numIter  = self.opts['numIter']              # how many iterations of gd to run
        stepSize = self.opts['stepSize']             # what should be our GD step size?

        self.weights = zeros(size(X,1))

        def func(w):
            # obj = loss(w) + (lambd/2) * norm(w)^2
            Yhat = X@w
            obj  = lossFn.loss(Y,Yhat) + (lambd/2) * norm(w)*norm(w)
            return obj

        def grad(w):
            # gr = grad(w) + lambd * w
            Yhat = X@w 
            gr   = lossFn.lossGradient(X,Y,Yhat) + lambd*w
            return gr

        w, trajectory = gd(func, grad, self.weights, numIter, stepSize)
        self.weights = w
        self.trajectory = trajectory
