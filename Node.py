import math
import random
import numpy as np


def SigmaEnt(dataSet={}):
    res = 0
    for key in dataSet.keys():
        classDistribution = {}
        tot = 0
        for element in dataSet[key]:
            if element not in classDistribution.keys():
                classDistribution[element] = 1
            else:
                classDistribution[element] += 1
            tot+=1
        for ky in classDistribution.keys():
            pk = classDistribution[ky]/tot
            res += pk*math.log2(pk)*tot
    return res


class Node:
    def __init__(self):
        #叶节点标记
        self.leaf = False
        #结论(只有在叶节点时有效)
        self.result = None
        #训练数据集
        self.trainData = {}
        #测试数据集(用来正向剪枝)
        self.testData = {}
        #划分属性的下标
        self.divide = 0
        #子节点
        self.children = {}
        #属性数量
        self.attributeNum = 0

    #计算一个集合的信息熵

    def Train(self, trainData, testData, attributeNum, pregruning=False):
        self.trainData = trainData
        self.testData = testData
        self.attributeNum = attributeNum
        classNum = 0
        #寻找数量最多的类
        for key in self.trainData.keys():
            if len(self.trainData[key]) > classNum:
                self.result = key
                classNum = len(self.trainData[key])
        tot = 0
        for key in self.testData:
            tot += len(self.testData[key])
        if self.result not in self.testData.keys():
            accurcy = 0
        else:
            accurcy = len(self.testData[self.result])
        existAttribute = {}
        #线判断是否所有的属性都属于同一类别
        for key in self.trainData.keys():
            for data in self.trainData[key]:
                if tuple(data) not in existAttribute.keys():
                    existAttribute[tuple(data)] = {}
                if key not in existAttribute[tuple(data)].keys():
                    existAttribute[tuple(data)][key] = 0
                existAttribute[tuple(data)][key] += 1
        if len(existAttribute.keys()) == 1:
            self.leaf = True
            num = 0
            lis = None
            for key in existAttribute.keys():
                lis = key
            for key in existAttribute[lis].keys():
                if existAttribute[lis][key] > num:
                    num = existAttribute[lis][key]
                    self.result = key
            return accurcy
        existClass = []
        for key in self.trainData.keys():
            if len(self.trainData[key]) != 0:
                existClass.append(key)
        if len(existClass) == 1:
            self.leaf = True
            self.result = existClass[0]
            return accurcy
        #增益熵数组
        Gain = []
        #遍历所有属性，计算增益熵
        for index in range(attributeNum):
            Dv = {}
            for key in self.trainData.keys():
                for data in self.trainData[key]:
                    if data[index] not in Dv.keys():
                        Dv[data[index]] = []
                    Dv[data[index]].append(key)
            Gain.append(SigmaEnt(Dv))
        #获取Dv最大的下标,也就是划分属性
        self.divide = Gain.index(max(Gain))
        #划分
        nextTrainDatas = {}
        nextTestDatas = {}
        for key in self.trainData.keys():
            for data in self.trainData[key]:
                #假如该条件不在子节点的键中
                if data[self.divide] not in self.children.keys():
                    #新建子节点
                    self.children[data[self.divide]] = Node()
                    nextTrainDatas[data[self.divide]] = {}
                    nextTestDatas[data[self.divide]] = {}
        for key in self.testData.keys():
            for data in self.testData[key]:
                nextTestDatas[data[self.divide]] = {}
        for key in self.trainData.keys():
            for data in self.trainData[key]:
                if key not in nextTrainDatas[data[self.divide]].keys():
                    nextTrainDatas[data[self.divide]][key] = []
                nextTrainDatas[data[self.divide]][key].append(data)
        #将满足条件的训练集再次划分到所有子节点中
        for key in self.testData.keys():
            for data in self.testData[key]:
                if key not in nextTestDatas[data[self.divide]].keys():
                    nextTestDatas[data[self.divide]][key] = []
                nextTestDatas[data[self.divide]][key].append(data)
        accnext = 0
        for key in nextTrainDatas:
            nextTestData = {}
            if key in nextTestDatas.keys():
                nextTestData = nextTestDatas[key]
            accnext += self.children[key].Train(trainData=nextTrainDatas[key],
                                                testData=nextTestData,
                                                attributeNum=attributeNum,
                                                pregruning=pregruning)
        if accnext < accurcy and pregruning:
            self.children = None
            self.leaf = True
        return max(accnext, accurcy)

