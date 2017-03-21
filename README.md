# MLClassifiersPython-NBC-SVM-Logistic-Regression

### Description
The given project contains an implementation for a Logistic Regression Classifier and an SVM classifier with hinge loss cost function. The models allow L2 regularization techinque and the parsing allows for feature contruction of text reviews using a bag-of-words (unigram) model. The yelp-dataset can also be found within the project.

### Models
The Logistic Regression and SVM classifier implemented in this project are learned using a gradient descent technique. In the case of SVM, we use a Sub-gradient descent method to accomodate the hinge-loss function. The learning rate, regularization parameters can be altered inside the 'train' function provided for each of the model classes. The 'verbose' argument when set during training allows you to print out the training details during the learning phase which may help you debug the code in case there are changes made to it.

### Requirements
* Python 2.7
* Anaconda distribution preferably
* Numpy
