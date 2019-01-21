from numpy import *
from pylab import *


class Perceptron():
	
	def __init__(self, opts):       
		self.opts = opts
		self.reset()

	def reset(self):
		self.weights = 0    # weight vector
		self.bias    = 0    # bias
		self.numUpd  = 0    # number of updates made

	def online(self):   
		return True

	def __repr__(self):        
		return    "w=" + repr(self.weights)   +  ", b=" + repr(self.bias)

	def train(self, X, Y):

		if self.online():
			for epoch in range(self.opts['numEpoch']):
				for n in range(X.shape[0]):
					self.nextExample(X[n], Y[n])

	def nextExample(self, X, Y):
		if Y * self.predict(X) <= 0:
			self.numUpd  = self.numUpd  + 1
			self.weights = self.weights + Y*X 
			self.bias =  self.bias + Y

	def predict(self, X):        
		if self.numUpd == 0:
			return 0
		else:
			return ((self.weights @ X) + self.bias)


	def predictAll(self, X):
		N,D = X.shape
		Y   = zeros(N)
		for n in range(N):
			Y[n] = self.predict(X[n,:])
		return Y
		
	def getRepresentation(self):
		return (self.numUpd, self.weights, self.bias)

	def tuneHyperparameter(self, hpName, hpValues, xTrain, yTrain, xVal, yVal):
		
		M = len(hpValues)
		trainAcc = []
		valAcc  = []
		
		for m in range(M):
			# train the classifier
			self.reset()
			self.opts[hpName] = hpValues[m]
			self.train(xTrain, yTrain)

			trAcc = 100*mean(1*((self.predictAll(xTrain)>=0) == (yTrain>=0)))
			vAcc = 100*mean(1*((self.predictAll(xVal)>=0) == (yVal>=0)))

			# store the results
			trainAcc.append(trAcc)
			valAcc.append(vAcc)        

		plot(hpValues, trainAcc, 'b-', hpValues, valAcc, 'r-')
		legend( ('Train', 'Validation') )
		xlabel(hpName)
		ylabel('Accuracy')
		title('Hyperparameter Tuning')
		show()