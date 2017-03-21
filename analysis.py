import math
import copy
import numpy as np
import random
import sys
from hw3 import read_data, LogisticRegSparse,SVM,Dataset,get_word_map
import nbc
import matplotlib.pyplot as plt

def shuffle(arr1,arr2):
    mixed = list(zip(arr1,arr2))
    random.shuffle(mixed)
    na1,na2 = zip(*mixed)
    return na1,na2
    
def write_data(fname,rev,lab):
    with open(fname,"w") as fp:
        for index in xrange(len(lab)):
            content = str(index)+"\t"+str(lab[index])+"\t"+str(rev[index])
            fp.write(content+"\n")
            
def kfold_split(datafile,folds):
    trn_reviews,trn_labels = read_data(datafile)
    trn_reviews,trn_labels = shuffle(trn_reviews,trn_labels)
    fname = "folds.dat"
    part = len(trn_labels)/folds
    for index in xrange(folds):
        dfname = str(index+1)+fname
        if index == folds-1:
            write_data(dfname,trn_reviews[index*part:],trn_labels[index*part:])
        else:
            write_data(dfname,trn_reviews[index*part:(index+1)*part],trn_labels[index*part:(index+1)*part])
    return trn_reviews,trn_labels

def run(data1,data2,choice,feat_len=4000):
    trn_reviews,trn_labels = data1[0],data1[1]
    tst_reviews,tst_labels = data2[0],data2[1]
    word_map= get_word_map(trn_reviews,feat_len)
    feat_len = min(feat_len,len(word_map.keys()))
    trainset = Dataset(trn_reviews,trn_labels,word_map)
    testset = Dataset(tst_reviews,tst_labels,word_map)
    if choice==0: #LR
        model = LogisticRegSparse(trainset,feat_len)
    elif choice == 1: #SVM
        model = SVM(trainset,feat_len)
    model.train()
    return model.zero_one_loss(testset)

def analyse(datafile,reportfile,folds=10):
    all_reviews,all_labels = kfold_split(datafile,folds)
    partitions = [0.01,0.03,0.05,0.08,0.1,0.15]
    part = len(all_labels)/folds
    loss_mean = []
    loss_std = []
    for partition in partitions:
        TSS = partition*len(all_reviews)
        fold_loss = []
        rem_rev = []
        rem_lab = []
        for index in xrange(folds):
            if (index == folds - 1):
                tst_rev = all_reviews[index*part:]
                tst_lab = all_labels[index*part:]
            else:
                tst_rev = all_reviews[index*part:(index+1)*part]
                tst_lab = all_labels[index*part:(index+1)*part]
            
            rem_rev += all_reviews[:index*part]
            rem_rev += all_reviews[(index+1)*part:]
            
            rem_lab += all_labels[:index*part]
            rem_lab += all_labels[(index+1)*part:]
            rem_rev,rem_lab = shuffle(rem_rev,rem_lab)
            fname = "P_"+str(partition)
            
            trainset = rem_rev[:int(TSS)],rem_lab[:int(TSS)] 
            fold_loss.append(run(trainset,(tst_rev,tst_lab),1))
        loss_mean.append(np.mean(fold_loss))
        loss_std.append(np.std(fold_loss)/math.sqrt(10))
    with open(reportfile,"w") as fp:
        fp.write("Train_percent,Zloss_mean,Zloss_std\n")
        for index in xrange(len(partitions)):
            content = str(partitions[index])+","+str(loss_mean[index])+","+str(loss_std[index])+"\n"
            fp.write(content)
    
def testrun():
    data = read_data("yelp_data.csv")
    run(data,data,1)
def main():
    if len(sys.argv) < 3:
        print ("Usage: Datafile Reportfile <folds optional>")
    else:
        analyse(sys.argv[1],sys.argv[2])
main()
