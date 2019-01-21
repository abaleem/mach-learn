from numpy import *
from pylab import *
import util


class DT():

    def __init__(self, opts):
        self.opts = opts
        self.isLeaf = True
        self.label  = 1

    def online(self):
        return False

    def __repr__(self):
        return self.displayTree(0)

    def displayTree(self, depth):
        if self.isLeaf:
            return (" " * (depth*2)) + "Leaf " + repr(self.label) + "\n"
        else:
            return (" " * (depth*2)) + "Branch " + repr(self.feature) + "\n" + \
                      self.left.displayTree(depth+1) + \
                      self.right.displayTree(depth+1)

    def predict(self, X):
        while self.isLeaf != True:
            if X[self.feature] <= 0.5:
                self = self.left
            else:
                self = self.right

        return self.label

    def predictAll(self, X):
        N,D = X.shape
        Y   = zeros(N)
        for n in range(N):
            Y[n] = self.predict(X[n,:])
        return Y


    def trainDT(self, X, Y, maxDepth, used):
        # size of the data set
        N,D = X.shape

        # Stopping Critera
        if maxDepth <= 0 or len(util.uniq(Y)) <= 1:
            self.isLeaf =  1              
            self.label  = util.mode(Y)

        else:
            bestFeature = -1     # which feature has lowest error
            bestError   = N      # the number of errors for this feature
            
            for d in range(D):

                # have we used this feature yet
                if d in used:
                    continue

                leftY  = Y[X[:,d]<=0.5]
                rightY = Y[X[:,d]>=0.5]

                leftYmode = util.mode(leftY)
                rightYmode = util.mode(rightY)

                leftYerror = (leftY != leftYmode).sum()
                rightYerror = (rightY != rightYmode).sum()

                error = leftYerror + rightYerror
                if error <= bestError:
                    bestFeature = d
                    bestError   = error

            # Error check.
            if bestFeature < 0:
                self.isLeaf = True
                self.label  = util.mode(Y)

            else:
                self.isLeaf  = False
                self.feature = bestFeature
                used.append(bestFeature)

                self.left  = DT({'maxDepth': maxDepth-1})
                self.right = DT({'maxDepth': maxDepth-1})

                self.left.trainDT(X[X[:,bestFeature]<=0.5,:],Y[X[:,bestFeature]<=0.5],maxDepth-1,used)
                self.right.trainDT(X[X[:,bestFeature]>=0.5,:],Y[X[:,bestFeature]>=0.5],maxDepth-1,used)


    def train(self, X, Y):
        self.trainDT(X, Y, self.opts['maxDepth'], [])

    def getRepresentation(self):    
        return self

    def tuneHyperparameter(self, hpName, hpValues, xTrain, yTrain, xVal, yVal):
        
        M = len(hpValues)
        trainAcc = []
        valAcc  = []
        
        for m in range(M):

            # train the classifier
            #self.reset()
            self.opts[hpName] = hpValues[m]
            self.train(xTrain, yTrain)

            trAcc = 100*mean(self.predictAll(xTrain) == yTrain)
            vAcc = 100*mean(self.predictAll(xVal) == yVal)

            # store the results
            trainAcc.append(trAcc)
            valAcc.append(vAcc)        

        plot(hpValues, trainAcc, 'b-', hpValues, valAcc, 'r-')
        legend( ('Train', 'Validation') )
        xlabel(hpName)
        ylabel('Accuracy')
        title('Hyperparameter Tuning')
        show()
