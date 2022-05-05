import os
from shutil import copyfile
from numpy import source
import re

STrP = "../aclImdb/train/pos"
STrN = "../aclImdb/train/neg"
STeP = "../aclImdb/test/pos"
STeN = "../aclImdb/test/neg"
TrP = "./trainPos"
TrN = "./trainNeg"
TeP = "./testPos"
TeN = "./testNeg"

def moveStoD(source, destination):
    posTrainName = os.listdir(source)
    count = 0
    tempList = []
    tempDict = {}

    for x in posTrainName:
        rating = re.search(r"(?<=_)\d*", x).group()
        if tempDict.get(rating) == None:
            tempDict[rating] = 1
            count += 1
        elif tempDict[rating] >= 25:
            continue
        else:
            tempDict[rating] += 1
            count += 1
    
        tempList.append(x)

        if count == 100:
            break
    
    tempCount = 0
    for i in range(len(tempList)):
        sourceReview = source + "/" + posTrainName[i]
        destinationReview = destination + "/" + posTrainName[i]
        copyfile(sourceReview, destinationReview)
        tempCount += 1

    print(tempDict)
    print(tempCount)
    return

# moveStoD(STeP, DTeP)
