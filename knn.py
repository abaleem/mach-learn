from numpy import *
from pylab import *
from statistics import *


class KNN():

	def __init__(self, opts):
		self.reset()
		self.opts = opts

	def reset(self):
		self.trX = zeros((0,0))
		self.trY = zeros((0))

	def online(self):
		return False

	def __repr__(self):
		return "Not defined for KNN"

	def predict(self, X):

		isKNN = self.opts['isKNN']     # true for KNN, false for epsilon balls
		N     = self.trX.shape[0]      # number of training examples

		if self.trY.size == 0:
			print('here')
			return 0                   # not trained yet
		elif isKNN:
			K = self.opts['K']
			val = []
			distances = []

			for i in range(N):
				eucDist = 0
				for j in range(len(X)):
					eucDist = eucDist +  (X[j] - self.trX[i,j])**2
				eucDist = eucDist**(0.5)
				distances.append(eucDist)

			ind = argsort(distances)

			for i in range(K):
				val.append(self.trY[ind[i]])
			return mode(val)

		else:						   # this is an epsilon ball model
			eps = self.opts['eps']     # how big is our epsilon ball
			val = 0                    # to be done, pos - #neg within and epsilon ball of X
			util.raiseNotDefined()
			return val

	def predictAll(self, X):
		N,D = X.shape
		Y   = zeros(N)
		for n in range(N):
			Y[n] = self.predict(X[n,:])
		return Y

	def getRepresentation(self):
		return (self.trX, self.trY)

	def train(self, X, Y):
		self.trX = X
		self.trY = Y

	def tuneHyperparameter(self, hpName, hpValues, xTrain, yTrain, xVal, yVal):
		
		M = len(hpValues)
		trainAcc = []
		valAcc  = []
		
		for m in range(M):

			# train the classifier
			self.reset()
			self.opts[hpName] = hpValues[m]
			
			self.train(xTrain, yTrain)

			trAcc = 100*(sum(1*((self.predictAll(xTrain)) == yTrain))/len(yTrain))
			vAcc = 100*(sum(1*((self.predictAll(xVal)) == yVal))/len(yVal))

			# store the results
			trainAcc.append(trAcc)
			valAcc.append(vAcc)        

		plot(hpValues, trainAcc, 'b-', hpValues, valAcc, 'r-')
		legend( ('Train', 'Validation') )
		xlabel(hpName)
		ylabel('Accuracy')
		title('Hyperparameter Tuning')
		show()