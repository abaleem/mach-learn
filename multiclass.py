from util import *
from numpy import *

class OVA:
    def __init__(self, K, mkClassifier):
        self.f = []
        self.K = K
        for k in range(K):
            self.f.append(mkClassifier())

    def train(self, X, Y):
        for k in range(self.K):
            print("training classifier for {0} versus rest".format(k))
            Yk = 2 * (Y == k) - 1   # +1 if it's k, -1 if it's not k
            self.f[k].fit(X, Yk)  
    
    def predict(self, X, useZeroOne=False):
        vote = zeros((self.K,))
        for k in range(self.K):
            probs = self.f[k].predict_proba(X.reshape(1, -1))
            if useZeroOne:
                vote[k] += 1 if probs[0,1] > 0.5 else 0
            else:
                vote[k] += probs[0,1]   # weighted vote
        return argmax(vote)

    def predictAll(self, X, useZeroOne=False):    
        N,D = X.shape
        Y   = zeros(N, dtype=int)
        for n in range(N):
            Y[n] = self.predict(X[n,:], useZeroOne)
        return Y
        

class AVA:
    def __init__(self, K, mkClassifier):
        self.f = []
        self.K = K
        for i in range(K):
            self.f.append([])
        for j in range(K):
            for i in range(j):
                self.f[j].append(mkClassifier())

    def train(self, X, Y):
        for i in range(self.K):
            for j in range(i):
                print("training classifier for {0} versus {1}".format(i,j))
                
                Xij = []
                Yij = []

                Xij = [X[x,:] for x in range(len(X)) if Y[x] == i or Y[x] == j]
                Yij = [Y[x] for x in range(len(Y)) if Y[x] == i or Y[x] == j]
                Yij = [1 if Yij[x] == i else -1 for x in range(len(Yij))]
                
                self.f[i][j].fit(Xij, Yij)  

    def predict(self, X, useZeroOne=False):
        vote = zeros((self.K,))
        for i in range(self.K):
            for j in range(i):

                probs = self.f[i][j].predict_proba(X.reshape(1, -1))
                p = None

                if useZeroOne:
                    vote[i] += 1 if probs[0,1] >= 0.5 else 0
                    vote[j] += 1 if probs[0,1] < 0.5 else 0
                else:
                    vote[i] += probs[0,1] if probs[0,1] >= 0.5 else 0
                    vote[j] += probs[0,0] if probs[0,1] < 0.5 else 0
        return argmax(vote)

    def predictAll(self, X, useZeroOne=False):
        N,D = X.shape
        Y   = zeros((N,), dtype=int)
        for n in range(N):
            Y[n] = self.predict(X[n,:], useZeroOne)
        return Y
    
class TreeNode:
    def __init__(self):
        self.isLeaf = True
        self.label  = 0
        self.info   = None

    def setLeafLabel(self, label):
        self.isLeaf = True
        self.label  = label

    def setChildren(self, left, right):
        self.isLeaf = False
        self.left   = left
        self.right  = right

    def isLeaf(self): return self.isLeaf
    
    def getLabel(self):
        if self.isLeaf: return self.label
        else: raise Exception("called getLabel on an internal node!")
        
    def getLeft(self):
        if self.isLeaf: raise Exception("called getLeft on a leaf!")
        else: return self.left
        
    def getRight(self):
        if self.isLeaf: raise Exception("called getRight on a leaf!")
        else: return self.right

    def setNodeInfo(self, info):
        self.info = info

    def getNodeInfo(self): return self.info

    def iterAllLabels(self):
        if self.isLeaf:
            yield self.label
        else:
            for l in self.left.iterAllLabels():
                yield l
            for l in self.right.iterAllLabels():
                yield l

    def iterNodes(self):
        yield self
        if not self.isLeaf:
            for n in self.left.iterNodes():
                yield n
            for n in self.right.iterNodes():
                yield n

    def __repr__(self):
        if self.isLeaf:
            return str(self.label)
        l = repr(self.left)
        r = repr(self.right)
        return '[' + l + ' ' + r + ']'
            

def makeBalancedTree(allK):
    if len(allK) == 0:
        raise Exception("makeBalancedTree: cannot make a tree of 0 classes")

    tree = TreeNode()
    
    if len(allK) == 1:
        tree.setLeafLabel(allK[0])
    else:
        split  = len(allK)//2
        leftK  = allK[0:split]
        rightK = allK[split:]
        leftT  = makeBalancedTree(leftK)
        rightT = makeBalancedTree(rightK)
        tree.setChildren(leftT, rightT)

    return tree

class MCTree:
    def __init__(self, tree, mkClassifier):
        self.f = []
        self.tree = tree
        for n in self.tree.iterNodes():
            n.setNodeInfo(   mkClassifier()  )

    def train(self, X, Y):
        for n in self.tree.iterNodes():
            if n.isLeaf:   # don't need to do any training on leaves!
                continue

            # otherwise we're an internal node
            leftLabels  = list(n.getLeft().iterAllLabels())
            rightLabels = list(n.getRight().iterAllLabels())

            print("training classifier for {0} versus {1}".format(leftLabels,rightLabels))

            # compute the training data, store in thisX, thisY
            ### TODO:

            # left = -1, right = = 1
            thisX = []
            thisY = []

            thisX = [X[x,:] for x in range(len(X)) if set([Y[x]]) <= set(leftLabels+rightLabels)]
            thisY = [Y[x] for x in range(len(Y)) if set([Y[x]]) <= set(leftLabels+rightLabels)]
            thisY = [1 if set([thisY[x]]) <= set(rightLabels) else -1 for x in range(len(thisY))]

            n.getNodeInfo().fit(thisX, thisY)

    def predict(self, X):
        ### TODO:

        for n in self.tree.iterNodes():
            break

        while(not n.isLeaf):
            probs  = n.getNodeInfo().predict_proba(X.reshape(1, -1))
            if(probs[0,1] >= 0.5):
                n = n.getRight()
            else: 
                n = n.getLeft()
        
        return n.getLabel()

    def predictAll(self, X):
        N,D = X.shape
        Y   = zeros((N,), dtype=int)
        for n in range(N):
            Y[n] = self.predict(X[n,:])
        return Y
        
def getMyTreeForWine(self, X, Y, allK):
    return makeBalancedTree(20)