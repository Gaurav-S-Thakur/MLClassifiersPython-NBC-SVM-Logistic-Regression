from collections import Counter
import string
import math
import numpy as np
import sys
import operator
import random
import copy

def trim(word):
    word = word.translate(None, string.punctuation)
    word = word.lower()
    return word

def sigmoid(x):
    try:
        return float(1)/(1+math.exp(-x))
    except OverflowError, e:
        return 1

def printvec(vec):
    string = ""
    for item in vec:
        string+= str(item)+" "
    print string

def toInt(vector):
    newvec = []
    for item in vector:
        newvec.append(int(item))
    return newvec

def read_data(filename):
    reviews=[]
    labels=[]
    with open(filename,"r") as fp:
        for line in fp:
            line = line.rstrip()
            values = line.split("\t")
            labels.append(values[1])
            reviews.append(values[2])
    return (reviews,labels)   

def get_word_map(dataset,feat_len): #dataset = list of reviews
    word_map = {}
    word_list = get_most_commons(dataset,feat_len)
    for index in xrange(len(word_list)):
        word_map[word_list[index]]=index
    return word_map

def get_most_commons(dataset, total=4000, skip=100):
    my_list = []
    for item in dataset:
        vals = trim(item).split()
        my_list += list(set(vals))
    
    mylist = sorted(my_list)
    listCount = set(Counter(mylist).most_common(total+skip)) - set(Counter(mylist).most_common(skip))
    temp =  sorted(listCount, key=operator.itemgetter(1,0), reverse=True)
    words = [item[0] for item in temp]
    return words

class Dataset(object):
    def __init__(self,reviews,labels,word_map,style=1):
        self.records = self.vecload(reviews,word_map)
        self.labels = toInt(labels)
    
    def vecload(self,reviews,word_map):
        records = []
        n_features = len(word_map.keys())
        for review in reviews:
            review = review.strip()
            words = trim(review).split(" ")
            review_map = np.zeros(n_features)
            for word in words:
                if word in word_map:
                    """
                    review_map[word_map[word]] = 1
                    """
                    if review_map[word_map[word]] > 0:
                        review_map[word_map[word]] = 2 
                    else:
                        review_map[word_map[word]] = 1
                    
            records.append(review_map)
        return records
                
    def shuffle(self):
        mixed = list(zip(self.records,self.labels))
        random.shuffle(mixed)
        self.records, self.labels = zip(*mixed)


class LogisticRegSparse(object):
    def __init__(self,trainset,features):
        self.trainset = trainset
        self.weights = self.weight_init(features)

    def weight_init(self,features):
        # For n features, the weight vector has n+1 dimensions, last index for bias term
        for index in xrange(len(self.trainset.records)):
            self.trainset.records[index] = np.append(self.trainset.records[index],1)
        return np.zeros(features+1)
    
    def train(self,**kwargs):
        prev_weights = np.zeros(len(self.weights))
        alpha = 0.01 if not "alpha" in kwargs else float(kwargs["alpha"])
        epochs = 100 if not "epochs" in kwargs else int(kwargs["epochs"])
        reg = 0.01 if not "reg_lamda" in kwargs else float(kwargs["reg_lambda"])
        verbose = False if not "verbose" in kwargs else True
        num_samples = len(self.trainset.records)
        self.trainset.shuffle()
        
        for epoch in xrange(epochs):
            batcherror = 0
            diffvec = []
            for index in xrange(num_samples):
                yhat = sigmoid(np.dot(self.trainset.records[index],self.weights))
                diffvec.append(self.trainset.labels[index]-yhat)
            batcherror = sum(np.square(diffvec))
            gradient = np.zeros(len(self.weights))
            for i in xrange(num_samples):
                gradient += diffvec[i]*self.trainset.records[i]
            gradient -= reg*self.weights
            self.weights += alpha*gradient
            #if verbose:
            #    #print ("Iteration:"+str(epoch)+"\t-----------> Error:"+str(batcherror))
            if verbose:
                print "Current Weights: "+str(self.weights)
                print "Old Weights: "+str(prev_weights)
            if epoch and np.linalg.norm(self.weights-prev_weights) < float(1)/10**6:
                break
            prev_weights = copy.copy(self.weights)
            

    def predict(self,input_vector):
        yhat = sigmoid(np.dot(input_vector,self.weights))
        if yhat > 0.5:
            return 1
        else:
            return 0
        
    def zero_one_loss(self,testset):
        error = 0
        for index in xrange(len(testset.records)):
            testset.records[index] = np.append(testset.records[index],1)
        for index in xrange(len(testset.records)):
            prediction = self.predict(testset.records[index])
            if not prediction == testset.labels[index]:
                error+=1
        zloss = float(error)/len(testset.records)
        print ("Zero-One-Loss-LR: "+str(zloss))
        return zloss
    
    def load(self, filename):
        weights = []
        with open(filename,"r") as fp:
            for line in fp:
                weights.append(float(line))
        self.weights = weights
    
    def save(self, filename):
        with open(filename,"w") as fp:
            fp.write(str(self.weights[0]))
            for wt in self.weights[1:]:
                fp.write("\n"+str(wt))

class SVM(object):
    def __init__(self,trainset,features):
        self.trainset = trainset
        self.weights = self.weight_init(features)

    def weight_init(self,features):
        # For n features, the weight vector has n+1 dimensions, last index for bias term
        for index in xrange(len(self.trainset.records)):
            self.trainset.records[index] = np.append(self.trainset.records[index],1)
        return np.zeros(features+1)
    
    def train(self,**kwargs):
        prev_weights = np.zeros(len(self.weights))
        alpha = 0.5 if not "alpha" in kwargs else float(kwargs["alpha"])
        epochs = 100 if not "epochs" in kwargs else int(kwargs["epochs"])
        reg = 0.01 if not "reg_lamda" in kwargs else float(kwargs["reg_lambda"])
        verbose = False if not "verbose" in kwargs else True
        num_samples = len(self.trainset.records)
        self.trainset.shuffle()
        for epoch in xrange(epochs):
            if verbose:
                print ("Iteration: "+str(epoch))
            grad_vec = np.zeros(len(self.weights))
            for index in xrange(num_samples):
                y = 1 if self.trainset.labels[index] > 0 else -1
                if y*np.dot(self.trainset.records[index],self.weights) < 1:
                    grad_vec += reg*self.weights - y*self.trainset.records[index]
                else:
                    grad_vec += reg*self.weights
            self.weights -= (alpha/len(self.trainset.labels))*grad_vec
            
            if epoch and np.linalg.norm(self.weights-prev_weights) < float(1)/10**6:
                break
            prev_weights = copy.copy(self.weights)
    
    def predict(self,input_vector):
        yhat = np.dot(input_vector,self.weights)
        if yhat > 0:
            #print "One"
            return 1
        else:
            #print "Zero"
            return 0
        
    def zero_one_loss(self,testset):
        error = 0
        for index in xrange(len(testset.records)):
            testset.records[index] = np.append(testset.records[index],1)
        for index in xrange(len(testset.records)):
            prediction = self.predict(testset.records[index])
            if not prediction == testset.labels[index]:
                error+=1
        zloss = float(error)/len(testset.records)
        print ("Zero-One-Loss-SVM: "+str(zloss))
        return zloss
    
    def load(self, filename):
        weights = []
        with open(filename,"r") as fp:
            for line in fp:
                weights.append(float(line))
        self.weights = weights
    
    def save(self, filename):
        with open(filename,"w") as fp:
            fp.write(str(self.weights[0]))
            for wt in self.weights[1:]:
                fp.write("\n"+str(wt))
    
def run1(data1,data2,feat_len=4000):
    trn_reviews,trn_labels = read_data(data1)
    tst_reviews,tst_labels = read_data(data2)
    word_map= get_word_map(trn_reviews,feat_len)
    feat_len = min(feat_len,len(word_map.keys()))
    trainset = Dataset(trn_reviews,trn_labels,word_map)
    testset = Dataset(tst_reviews,tst_labels,word_map)
    model = LogisticRegSparse(trainset,feat_len)
    model.train(verbose=1)
    #model.train()
    model.save("LRmodel")
    return model.zero_one_loss(testset)

def run2(data1,data2,feat_len=4000):
    trn_reviews,trn_labels = read_data(data1)
    tst_reviews,tst_labels = read_data(data2)
    word_map= get_word_map(trn_reviews,feat_len)
    feat_len = min(feat_len,len(word_map.keys()))
    trainset = Dataset(trn_reviews,trn_labels,word_map)
    testset = Dataset(tst_reviews,tst_labels,word_map)
    model = SVM(trainset,feat_len)
    model.train(verbose=1)
    #model.train()
    model.save("SVMmodel")
    return model.zero_one_loss(testset)

def run():
    if not len(sys.argv)==4 and not len(sys.argv)==5:
        print ("Usage: python hw3.py trainset testset model_num <feature length optional>")
        exit()
    if sys.argv[3]=="1" and len(sys.argv)==4:
        run1(sys.argv[1],sys.argv[2])
    elif sys.argv[3]=="1" and len(sys.argv)==5:
        run1(sys.argv[1],sys.argv[2],int(sys.argv[4]))
    elif sys.argv[3]=="2" and len(sys.argv)==4:
        run2(sys.argv[1],sys.argv[2])
    elif sys.argv[3]=="2" and len(sys.argv)==5:
        run2(sys.argv[1],sys.argv[2],int(sys.argv[4]))
    else:
        print ("Usage-\n1: Logistic Regression\n2: SVM")
#run()
