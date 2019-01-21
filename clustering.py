from numpy import *
import numpy as np
from util import *
from pylab import *

def kmeans(X, mu0, doPlot=True):
	'''
	X is an N*D matrix of N data points in D dimensions.

	mu is a K*D matrix of initial cluster centers, K is
	the desired number of clusters.

	this function should return a tuple (mu, z, obj) where mu is the
	final cluster centers, z is the assignment of data points to
	clusters, and obj[i] is the kmeans objective function:
	  (1/N) sum_n || x_n - mu_{z_n} ||^2
	at iteration [i].

	mu[k,:] is the mean of cluster k
	z[n] is the assignment (number in 0...K-1) of data point n
	'''

	mu = mu0.copy()
	N,D = X.shape
	K   = mu.shape[0]

	z   = zeros((N,), dtype=int)
	obj = []

	# run at most 100 iterations
	for it in range(100):
		z_old = z.copy()
		
		for n in range(N):
			bestK    = -1
			bestDist = 0
			for k in range(K):
				d = linalg.norm(X[n,:] - mu[k,:])
				if d < bestDist or bestK == -1:
					bestK = k
					bestDist = d
			z[n] = bestK

		for k in range(K):
			mu[k,:] = mean(X[z==k, :], axis=0)

		currentObjective = 0
		for n in range(N):
			currentObjective = currentObjective + linalg.norm(X[n,:] - mu[z[n],:]) ** 2 / float(N)
		obj.append(currentObjective)

		if doPlot:
			plotDatasetClusters(X, mu, z)
			show(block=False)

		if all(z == z_old):
			print("Converging Iteration {0}, objective={1}".format(it, currentObjective))
			break

	if doPlot and D==2:
		plotDatasetClusters(X, mu, z)
		show(block=False)

	return (mu, z, array(obj))


def initialize_clusters(X, K, method):
	'''
	X is N*D matrix of data
	K is desired number of clusters (>=1)
	method is one of:
	  determ: initialize deterministically (for comparitive reasons)
	  random: just initialize randomly
	  ffh   : use furthest-first heuristic

	returns a matrix K*D of initial means.

	you may assume K <= N
	'''

	N,D = X.shape
	mu = zeros((K,D))

	if method == 'determ':
		# just use the first K points as centers
		mu = X[0:K,:].copy()

	elif method == 'random':
		# pick K random centers
		dataPoints = list(range(N))
		permute(dataPoints)
		mu = X[dataPoints[0:K], :].copy()   # ditto above

	elif method == 'ffh':
		# pick the first center randomly and each subsequent
		# subsequent center according to the furthest first
		# heuristic

		mu[0,:] = X[int(rand() * N), :].copy()    # be sure to copy!
		found = 1

		for k in range(1, K):

			currentBest = 0
			currentBestIndex = 0

			for i in range(0, N):
				temp = 0
				closestCenteroid = float('inf')

				for j in range(0,k):

					temp = linalg.norm(mu[j,:]-X[i,:])  
					if(temp < closestCenteroid):
						closestCenteroid = temp

				if(closestCenteroid > currentBest):
					currentBest = closestCenteroid
					currentBestIndex = i

			
			mu[k,:] = X[currentBestIndex,:].copy() 
			
	elif method == 'km++':
		# pick the first center randomly and each subsequent
		# subsequent center according to the kmeans++ method
		# HINT: see numpy.random.multinomial
		
		mu[0,:] = X[int(rand() * N), :].copy()

		for k in range(1, K):

			currentBest = 0
			currentBestIndex = 0

			d = []

			for i in range(0, N):
				temp = 0
				closestCenteroid = float('inf')

				for j in range(0,k):

					temp = linalg.norm(mu[j,:]-X[i,:])  
					if(temp < closestCenteroid):
			
						closestCenteroid = temp
				d.append(closestCenteroid)

			dSum = sum(d)
			p = [d[i]/dSum for i in range(len(d))]

			index = np.random.choice(N,1,p)

			mu[k,:] = X[index,:].copy() 

	else:
		print("Initialization method not implemented")
		sys.exit(1)

	return mu

def plotDatasetClusters(X, mu, z):
  colors = array(['b','r','m','k','g','c','y','b','r','m','k','g','c','y','b','r','m','k','g','c','y','b','r','m','k','g','c','y','b','r','m','k','g','c','y'])
  plot(X[:,0], X[:,1], 'w.')
  # hold(True)
  for k in range(mu.shape[0]):
    plot(X[z==k,0], X[z==k,1], colors[k] + '.')
    plot(array([mu[k,0]]), array([mu[k,1]]), colors[k] + 'x')
  # hold(False)


