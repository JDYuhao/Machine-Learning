#-*- coding: utf-8 -*-
from sklearn import datasets
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

#导入数据集
class adaboostClassifer:
    def __init__(self):
        self.weight = []
        self.H = []
        self.data = []
        self.trainData = []
        self.testData = []
        self.baseClassifers = []
        self.alphas = []
        self.numberOfClass = 0

    def loadDataSet(self):
        #导入相关的数据
        iris = datasets.load_digits()
        iris_feature = iris["data"]
        iris_target = iris["target"].reshape(
            len(iris_feature),1)
        
        #原始数据
        self.data = np.concatenate(
            (iris_feature,iris_target), axis=1)

        #所有的类别数
        self.numberOfClass = len(np.unique(iris_target))

    def splitData(self, ratio = 0.4):
        totalNum = len(self.data)
        trainLoc = np.random.choice(totalNum,
                    int(ratio*totalNum), replace=False)
        self.trainData = self.data[trainLoc, :]
        self.testData = self.data[-trainLoc, :]


        #赋予训练数据训练的初始权重
        trainLen = len(self.trainData)
        self.weight = 1.0*np.ones((trainLen, self.numberOfClass)
                      )/(trainLen*(self.numberOfClass-1))
        
        for loc in xrange(trainLen):
            self.weight[loc, int(self.trainData[loc, -1])] = 0.0

        self.H = np.zeros((trainLen, self.numberOfClass))

    def normalize(self, weight):
        return weight/sum(weight)
        

    def train(self, sampleRatio = 0.5, numIter = 20, errMin = 1e-5, rescaleSize = 1000):

        for iteration in xrange(numIter):
            weight = np.sum(self.weight,axis =1)
            q = self.weight.copy()

            nCol = len(q[0,:])

            for col in xrange(nCol):
                q[:, col] = 1.0*q[:, col]/weight

            #权重归一化，方便抽样
            dataDis = self.normalize(weight)

            #抽样数据
            totalTrainNum = len(self.trainData)
            sampleLoc = np.random.choice(totalTrainNum, int(sampleRatio*totalTrainNum),
                                         p = list(dataDis), replace = False)
            sampleData = self.trainData[sampleLoc, :]


            #利用weakLearner进行预测
            X = sampleData[:, :-1]
            Y = sampleData[:, -1]
            clf = KNeighborsClassifier(n_neighbors=1, algorithm = "ball_tree")
            clf = clf.fit(X, Y)

            allPredictRes = clf.predict(self.trainData[:,:-1])


            predictArr = np.zeros((len(self.trainData), self.numberOfClass))

            #转化为预测矩阵
            lineNum = 0
            for predictRes in allPredictRes:
                predictArr[lineNum, int(predictRes)] = 1
                lineNum += 1

            predGap = np.abs(np.sign(allPredictRes-self.trainData[:,-1]))
            predInfo = np.abs(predGap-1).reshape((len(self.trainData),1))
                    
            #基于样本的错误率估计
            err = 0.5*sum(dataDis.reshape((len(self.trainData),1))*
                 (1-predInfo+
                  np.sum(1.0/(self.numberOfClass-1)*q*predictArr, axis=1
                  ).reshape((len(self.trainData),1))))


            #错误估计的各种情形

            #当误差满足要求时，直接退出
            if err < errMin:
                break

            if err > 0.5:
                alpha = 0.0
            else:
                alpha = float(np.log(1.0/err-1))

            #分类器叠加
            self.H += alpha*predictArr
            
            


            #权重更新
            self.weight = self.weight*np.exp(-alpha/rescaleSize*
                         (1-np.kron(np.ones((1,self.numberOfClass)), predInfo)+predictArr))

            self.baseClassifers.append(clf)
            self.alphas.append(alpha)


    def predict(self):
        predictMat = np.zeros((len(self.data), self.numberOfClass))
        
        for loop in xrange(len(self.alphas)):
            tempMat = np.zeros((len(self.data), self.numberOfClass))
            clf = self.baseClassifers[loop]
            
            allPredictRes = clf.predict(self.testData[:,:-1])

            lineNum = 0
            for predictRes in allPredictRes:
                tempMat[lineNum, int(predictRes)] = 1
                lineNum += 1
                
            predictMat += self.alphas[loop]*tempMat
        print sum(np.abs(np.sign(
            np.array(map(lambda x:np.argmax(x), predictMat))
            -self.testData[:,-1])))*1.0/len(self.testData)

                     
if __name__=="__main__":
    myClassifer = adaboostClassifer()
    myClassifer.loadDataSet()
    myClassifer.splitData()
    myClassifer.train()
    myClassifer.predict()
    
        
        
