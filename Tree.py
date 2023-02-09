import random

import numpy as np
from Node import Node

def dfs(node, data):
    if node.leaf:
        return node.result
    if data[node.divide] not in node.children.keys():
        return node.result
    return dfs(node.children[data[node.divide]], data)


class DecicisonTree:
    def __init__(self, data=[], label=[]):
        #类型标签
        self.classLabel = np.unique(label)
        #类型数量
        self.classNum = len(self.classLabel)
        #数据集
        self.dataSet = {}
        #属性数量
        self.attributeNum = np.shape(data)[1]
        size = np.shape(data)[0]
        #储存数据集
        for index in range(size):
            if label[index] not in self.dataSet.keys():
                self.dataSet[label[index]] = []
                self.dataSet[label[index]].append(data[index])
            else:
                self.dataSet[label[index]].append(data[index])
        #建立头节点
        self.headNode = Node()

    #切割数据集
    def __dataSplit(self, ratio):
        testData = {}
        trainData = {}
        for key in self.dataSet.keys():
            size = len(self.dataSet[key])
            offSet = int(size*ratio)
            testData[key] = []
            trainData[key] = []
            if size == 0 or offSet < 1:
                trainData[key].extend(self.dataSet[key])
                continue
            random.shuffle(self.dataSet[key])
            testData[key].extend(self.dataSet[key][:offSet])
            trainData[key].extend(self.dataSet[key][offSet:])
        return trainData, testData

    #训练
    def Train(self, test_size = 0.3, pregruning=False):
        trainData, testData = self.__dataSplit(test_size)
        self.headNode.Train(trainData=trainData,
                            testData=testData,
                            attributeNum=self.attributeNum,
                            pregruning=pregruning)

    def Predict(self, data):
        return dfs(self.headNode, data)



if __name__ == "__main__":
    testDatas = {'banna': [[3, 2, 1], [5, 4, 3], [2, 3, 3], [1, 3, 4], [1, 2, 3]],
                 'apple': [[3, 2, 1], [5, 4, 3], [4, 5, 6], [2, 3, 6], [4, 5, 6]]}
    datas = [[3, 2, 1], [5, 4, 3], [4, 5, 6], [1, 3, 4], [1, 2, 3],[3, 2, 1], [5, 4, 3], [4, 5, 6], [2, 3, 6], [4, 5, 6]]
    labels = ['banna', 'banna', 'banna', 'banna', 'banna', 'apple', 'apple' ,'apple' ,'apple', 'apple']
    dTree = DecicisonTree(datas, labels)
    dTree.Train(test_size=0.3, pregruning=True)
    print(dTree.Predict([4, 5, 6]))