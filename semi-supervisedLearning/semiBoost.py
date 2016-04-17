#-*- coding: utf-8 -*-
__author__ = 'qinyuhao'

import numpy as np
from sklearn.neighbors import BallTree
from sklearn import linear_model
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import math


class SBClassifer:
    def __init__(self):
        #初始化基本数据
        self.originData = []
        self.data = []
        self.labeledData = []
        self.labeledLoc = []
        self.unlabeledData = []
        self.sampleWeight = []
        self.H = []

    def loadDataSet(self, filename):
        #读入基本数据
        def lineTransfer(line):
            transferLine = []
            eachline = line.strip().split(",")
            for item in eachline:
                transferLine.append(float(item))
            transferLine[-1] = int(transferLine[-1])
            return transferLine

        with open(filename, 'r') as f:
            self.data = np.array([lineTransfer(eachline) for eachline in f])
        self.originData = self.data.copy()

        totalDataLen = len(self.data)
        self.sampleWeight = [1.0/totalDataLen for loc in xrange(totalDataLen)]

    def splitData(self, ratio = 0.1):
        #在测试中选区一部分数据隐藏标签
        totalDataLen = len(self.data)
        labeledNum = int(ratio*totalDataLen)

        #记录标记数据的位置
        self.labeledLoc = np.random.choice(totalDataLen,
                                           labeledNum, replace= False)

        self.data[self.labeledLoc, -1] = np.zeros((1,labeledNum))

    def reweight(self, weight):
        return map(lambda x:x[0], (weight/sum(weight)).tolist())

    def train(self, sampleRatio = 0.5, nNeighbours = 10, C = 2.0):

        totalDataLen = len(self.data)
        labeledNum = int(sampleRatio*totalDataLen)
        self.H = np.zeros((totalDataLen, 1))

        #数据抽样部分

        #将数据分为label和unlabel两部分
        labelData = self.data[np.where(self.data[:,-1]!=0)[0]]
        unlabelData = self.data[np.where(self.data[:,-1]==0)[0]]

        forLabelNei = BallTree(labelData[:,:-1], metric='euclidean')
        forUnlableNei = BallTree(unlabelData[:,:-1], metric='euclidean')

        labelDis, labelIndex = forLabelNei.query(self.data[:, :-1],
                                                 k=nNeighbours, return_distance= True)

        unlabelDis, unlabelIndex  = forUnlableNei.query(self.data[:, :-1],
                                                 k=nNeighbours, return_distance= True)

        for T in range(30):
            p = np.zeros((totalDataLen,1))
            q = np.zeros((totalDataLen,1))

            #构建人工label
            for loop in xrange(nNeighbours):

                disL = labelDis[:, loop]
                sigmaLp = np.sign(labelData[labelIndex[:, loop], -1]+1).reshape(totalDataLen,1)
                sigmaLq = np.sign(-labelData[labelIndex[:, loop], -1]+1).reshape(totalDataLen,1)
                similarityL = np.exp(-disL**2).reshape(totalDataLen,1)

                disU = unlabelDis[:, loop]
                similarityU = np.exp(-disU**2).reshape(totalDataLen,1)

                p += sigmaLp*similarityL*np.exp(-2*self.H)+\
                     C/2*similarityU*np.exp(self.H[unlabelIndex[:,loop]]-self.H)

                q += sigmaLq*similarityL*np.exp(2*self.H)+\
                     C/2*similarityU*np.exp(self.H-self.H[unlabelIndex[:,loop]])

            #z为人工label
            z = np.sign(p - q)

            data = np.concatenate((self.data[:,:-1], z), axis=1)

            choiceLoc = np.random.choice(totalDataLen, labeledNum,
                        replace= False, p=self.sampleWeight)
            sampleData = data[choiceLoc]

            #
            #clf = linear_model.SGDClassifier(loss="log")
            clf = KNeighborsClassifier(n_neighbors=1)
            X = sampleData[:, :-1]
            Y = sampleData[:, -1]
            clf.fit(X, Y)

            iterationRes = clf.predict(self.data[:,:-1]).reshape(totalDataLen,1)

            errorRate = float(1.0*sum(p*np.sign(-iterationRes+1.0)+
                                q*np.sign(iterationRes+1.0))/sum(p+q))
            if errorRate > 0.5:
                alpha = 0.0
            else:
                alpha = 1.0/4*math.log((1-errorRate)/errorRate)

            self.H += alpha*iterationRes

            self.sampleWeight = self.reweight(np.abs(p-q))

if __name__ == "__main__":
    #程序的核心运行部分
    myClassifer = SBClassifer()
    myClassifer.loadDataSet("titanic.txt")
    myClassifer.splitData(0.01)
    myClassifer.train()
    print sum(np.abs(np.sign(myClassifer.H)-myClassifer.data[:,-1].reshape(len(myClassifer.data), 1)))/4402


