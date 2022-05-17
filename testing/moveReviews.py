from argparse import Action
import os
from shutil import copyfile
from numpy import source
import re

STrP = "../aclImdb/train/pos"
STrN = "../aclImdb/train/neg"
STeP = "../aclImdb/test/pos"
STeN = "../aclImdb/test/neg"
DTrP = "./trainPos"
DTrN = "./trainNeg"
DTeP = "./testPos"
DTeN = "./testNeg"

print(len(os.listdir(DTrP)))
print(len(os.listdir(DTrN)))
print(len(os.listdir(DTeP)))
print(len(os.listdir(DTeN)))

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
        elif tempDict[rating] >= 250:
            continue
        else:
            tempDict[rating] += 1
            count += 1

        tempList.append(x)

        if count >= 1000:
            break
    
    tempCount = 0
    for i in range(len(tempList)):
        sourceReview = source + "/" + tempList[i]
        destinationReview = destination + "/" + tempList[i]
        copyfile(sourceReview, destinationReview)
        tempCount += 1

    return

actualVal = {'1':0,'2':0,'3':0,'4':0,'7':0,'8':0,'9':0,'10':0}
reviewNames = os.listdir(DTrP)
for x in reviewNames:
    rating = re.search(r"(?<=_)\d*", x).group()
    actualVal[rating] += 1

print(actualVal)

# moveStoD(STrP, DTrP)
# moveStoD(STrN, DTrN)
# moveStoD(STeP, DTeP)
# moveStoD(STeN, DTeN)
