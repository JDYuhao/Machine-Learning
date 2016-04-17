#-*- coding: utf-8 -*-
__author__ = 'qinyuhao'
'''
－－－－－－－－－－－－－－－－－－－－－－－－－
@文件描述：
该文件时决策树的的实现，可以针对连续变量和分类变量进行树划分
该文件不仅具有预测功能，同时可以对变量重要性进行排序
在确定是否为分类变量时，该程序处理方法比较简单，认为可以转化为
浮点函数的变量是连续变量，不可以的就是分类变量。
－－－－－－－－－－－－－－－－－－－－－－－－－
@使用方法：
1. 初始化树：
    myTree = DTree(featureName, parameter)
    %fetureName传入一个list,记录不同feature的名字，而不用加入flag
    %parameter是一个字典，记录树训练的参数：
     目前实现的key有:
        parameter = {"imPurityMtd": "ent", "maxRatio": 0.9, "max_depth":3}  
        ＃imPurityMtd -> 不纯度计算方法，ent代表熵，gini代表Gini系数
        ＃maxRatio -> 每片枝叶最大类占比的最小值
        ＃max_depth -> 树的最大深度
2. 读入文件数据：
    myTree.loadDataSet(filename) #文件以\t分隔，最后一列是flag
3. 训练集测试集切分：
    myTree.trainTestSplit(ratio = 0.5) ratio ->  分隔比例
    可以通过myTree.trainData, myTree.testData 查看结果
4. 训练树:
    myTree.train()
    训练结果可以通过myTree.tree查看
    同时变量重要性可以通过myTree.importance查看
5. 预测:
    myTree.predict(measure="acc")
    目前只实现了acc的计算，返回acc指标
    同时预测结果可以通过myTree.predictFlag查看
－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－
'''


import math
import random
from collections import Counter
from copy import deepcopy

#该函数对读入数据的每一行做处理
def lineTransfer(line):
    transferLine = []
    eachline = line.strip().split("\t")
    for item in eachline:
        try:
            transferLine.append(float(item))
        except:
            transferLine.append(item)
    transferLine[-1] = str(transferLine[-1])
    return transferLine

class DTree:
    def __init__(self, featureName,
                 parameter =  {"imPurityMtd": "ent", "maxRatio": 0.8, "max_depth":2}):
        self.dataset = [] #存储数据集
        self.tree = {} #存储训练的树
        self.trainData = [] #训练集
        self.testData = [] #测试集
        self.parameter = parameter #决策数前剪枝时用到的参数
        self.labels = featureName #存储feature的名字
        self.importance = dict(zip(self.labels,[0.0 for item in self.labels]))#存取变量的重要性
        self.predictFlag = []#存储预测的结果

    #读入数据集
    def loadDataSet(self, filename):
        with open(filename, "r") as f:
         self.dataset = [lineTransfer(eachline) for eachline in f]

    #将数据集分为训练和测试集
    def trainTestSplit(self, ratio):
        trainLen = int(ratio*len(self.dataset))
        dataset = deepcopy(self.dataset)
        random.shuffle(dataset)
        self.trainData = dataset[:trainLen]
        self.testData = dataset[trainLen+1:]

    #计算不纯度
    def calcImpurity(self, dataset):
        if len(dataset)==0:
            return 0.0

        dataDis = {}
        totalLen = len(dataset)

        #概率分布计数
        cateList = [line[-1] for line in dataset]
        dataDis = Counter(cateList)

        impurity = 0.0

        #不纯度的计算方法
        if self.parameter["imPurityMtd"] == "ent":
            for value in dataDis.values():
                impurity -= math.log(float(value)/totalLen)*(float(value)/totalLen)
        if self.parameter["imPurityMtd"] == "gini":
            for value in dataDis.values():
                impurity += (1-float(value)/totalLen)*(float(value)/totalLen)

        return impurity

    #分裂分类数据
    def splitDatasetForCate(self, dataset, colIdx, value):
        returnList = []
        for line in dataset:
            if line[colIdx] == value:
                rfectVec = line[:colIdx]
                rfectVec.extend(line[colIdx+1:])
                returnList.append(rfectVec)
        return returnList

    #分裂连续变量,寻找使得不纯度最小的变量进行分裂
    def splitDatasetForCon(self, dataset, colIdx):
        baseImpurityValue = 1e3
        baseSplitValue = 0.0
        for line in dataset:
            splitValue = line[colIdx]
            leftList = []
            rightList = []
            for line in dataset:
                rfectVec = line[:colIdx]
                rfectVec.extend(line[colIdx+1:])
                if line[colIdx] <= splitValue:
                    leftList.append(rfectVec)
                else:
                    rightList.append(rfectVec)

            impurityValue = 1.0*len(leftList)/len(dataset)*self.calcImpurity(leftList)+\
                    1.0*len(rightList)/len(dataset)*self.calcImpurity(rightList)
            if baseImpurityValue > impurityValue:
                baseImpurityValue = impurityValue
                baseSplitValue = splitValue

        leftList = []
        rightList = []
        for line in dataset:
            rfectVec = line[:colIdx]
            rfectVec.extend(line[colIdx+1:])
            if line[colIdx] <= baseSplitValue:
                leftList.append(rfectVec)
            else:
                rightList.append(rfectVec)

        return baseSplitValue, leftList, rightList

    #确定最好的分类标签
    def findBestSplitFeature(self, dataset):
        baseImpurity = self.calcImpurity(dataset)
        baseGain = 0

        for col in range(len(dataset[0])-1):
            splitInfo = 0.0
            if isinstance(col, str):
                newImpurity = 0.0
                allValues = set(line[col] for line in dataset)
                for value in allValues:
                    prob = 1.0*len(splitDataset)/len(dataset)
                    splitDataset = self.splitDatasetForCate(dataset, col, value)
                    newImpurity += prob*self.calcImpurity(splitDataset)
                    splitInfo -= prob*math.log(prob)
            else:
                baseSplitValue, leftList, rightList = self.splitDatasetForCon(dataset, col)
                leftProb = 1.0*len(leftList)/len(dataset)
                rightProb = 1.0*len(rightList)/len(dataset)
                newImpurity = leftProb*self.calcImpurity(leftList)+\
                    rightProb*self.calcImpurity(rightList)
                splitInfo -= (leftProb*math.log(leftProb)+rightProb*math.log(rightProb))
            gain = (baseImpurity - newImpurity)/splitInfo
            if gain > baseGain:
                baseGain = gain
                selectCol = col
        if baseGain == 0:
            selectCol = -1
        return selectCol, baseGain

    #确定预测标签
    def maxCate(self, dataset):
        cateList = [line[-1] for line in dataset]
        labelCounter = Counter(cateList)

        baseValue = 0

        label = "NULL"
        #条件1：当某一类大于某个比例时，树停止生长
        for key, value in labelCounter.items():
            if value > baseValue:
                baseValue = value
                label = key
        return label, baseValue

    #构建决策树，利用python的字典进行构建
    def buildTree(self, dataset, labels, depth):
        totalLen = len(dataset)

        label, baseValue = self.maxCate(dataset)

        if baseValue*1.0/totalLen >= self.parameter["maxRatio"]:
            return label

        if len(dataset[0])==1:
            return label

        if depth >= self.parameter["max_depth"]:
            return label

        bestFeat, baseGain = self.findBestSplitFeature(dataset)
        bestFeatLabel = labels[bestFeat]
        self.importance[bestFeatLabel] += len(dataset)*1.0/len(self.trainData)*baseGain

        tree = {bestFeatLabel: {}}
        del(labels[bestFeat])

        if isinstance(dataset[0][bestFeat], str):
            allValues = set(line[bestFeat] for line in dataset)
            for value in allValues:
                splitData = self.splitDatasetForCate(dataset, bestFeat, value)

                subLabels = labels[:]
                subTree = self.buildTree(splitData, subLabels, depth+1)
                tree[bestFeatLabel][value] = subTree
        else:
            baseSplitValue, leftList, rightList = self.splitDatasetForCon(dataset, bestFeat)
            subLabels = labels[:]
            subLeftTree = self.buildTree(leftList, subLabels, depth+1)
            tree[bestFeatLabel][(baseSplitValue, -1)] = subLeftTree

            subLabels = labels[:]
            subRightTree = self.buildTree(rightList, subLabels, depth+1)
            tree[bestFeatLabel][(baseSplitValue, 1)] = subRightTree
        return tree

    #根据训练集训练决策树
    def train(self):
        labels = deepcopy(self.labels)
        self.tree = self.buildTree(self.trainData, labels, 0)

    #预测单一向量的决策树
    def predictOneVec(self, tree, testVec):
        root = tree.keys()[0]
        secondDict = tree[root]
        featIndex = self.labels.index(root)
        key = testVec[featIndex]
        if isinstance(key, str):
            valueOfFeat = secondDict[key]
        else:
            cmpValue = secondDict.keys()[0][0]
            if key <= cmpValue:
                valueOfFeat = secondDict[(cmpValue, -1)]
            else:
                valueOfFeat = secondDict[(cmpValue, 1)]

        if isinstance(valueOfFeat, dict):
            calassLabel = self.predictOneVec(valueOfFeat, testVec)
        else:
            calassLabel = valueOfFeat
        return calassLabel

    #对所有的测试集进行预测，这里我们仅仅选择准确率作为统计指标，用户可以根据定义自己拓展
    def predict(self, measure="acc"):
        predLabels = []
        for testVec in self.testData:
            label = self.predictOneVec(self.tree, testVec)
            predLabels.append(label)
        
        self.predictFlag = preLabels
        
        if measure == "acc":
            totalTestLen = len(self.testData)
            accCount = 0
            for loc in range(totalTestLen):
                if predLabels[loc] == self.testData[loc][-1]:
                    accCount += 1.0
            return accCount/totalTestLen

#运行主函数
if __name__ == "__main__":
    featureName = ["v1", "v2", "v3", "v4"]#设定feaute的名称, 不需要设置flag

    #设置训练决策树的参数
    parameter =  {"imPurityMtd": "ent", "maxRatio": 0.9, "max_depth":3}


    myTree = DTree(featureName, parameter)
    myTree.loadDataSet("titanic.txt")#读入数据
    myTree.trainTestSplit(0.5)#分裂训练集和测试集
    myTree.train()#训练决策树
    print myTree.tree#打印训练后的树
    print myTree.importance#打印变量重要性指标
    print myTree.predict()#决策树预测与评估



