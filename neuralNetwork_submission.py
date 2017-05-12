# -*- coding: utf-8 -*-

import classificationMethod
import numpy as np
import util


def softmax(X):
    e = np.exp(X - np.max(X))
    det = np.sum(e, axis=1)
    return (e.T / det).T


def sigmoid(X):
    return 1. / (1. + np.exp(-X))


def ReLU(X):
    return X * (X > 0.)


def binary_crossentropy(true, pred):
    pred = pred.flatten()
    return -np.sum(true * np.log(pred) + (1. - true) * np.log(1. - pred))


def categorical_crossentropy(true, pred):
    return -np.sum(np.log(pred[np.arange(len(true)), true]))


class NeuralNetworkClassifier(classificationMethod.ClassificationMethod):
    def __init__(self, legalLabels, type, seed):
        self.legalLabels = legalLabels
        self.type = type
        self.hiddenUnits = [100, 100]
        self.numpRng = np.random.RandomState(seed)
        self.initialWeightBound = None
        self.epoch = 300

    def train(self, trainingData, trainingLabels):

        """
        Outside shell to call your method.
        Iterates several learning rates and regularization parameter to select the best parameters.

        Do not modify this method except uncommenting some lines.
        """
        if len(self.legalLabels) > 2:
            zeroFilledLabel = np.zeros((trainingData.shape[0], len(self.legalLabels)))
            zeroFilledLabel[np.arange(trainingData.shape[0]), trainingLabels] = 1.
        else:
            zeroFilledLabel = np.asarray(trainingLabels).reshape((len(trainingLabels), 1))

        trainingLabels = np.asarray(trainingLabels)

        self.initializeWeight(trainingData.shape[1], len(self.legalLabels))

        for i in range(self.epoch):
            netOut = self.forwardPropagation(trainingData)

            # If you want to print the loss, please uncomment it
            # print("Step: ", (i+1), " - ", self.loss(trainingLabels, netOut))

            self.backwardPropagation(netOut, zeroFilledLabel, 0.05 / float(len(trainingLabels)))

            # If you want to print the accuracy for the training data, please uncomment it
            # guesses = self.classify(trainingData)
            # acc = [guesses[i] == trainingLabels[i] for i in range(trainingLabels.shape[0])].count(True)
            # print ("Training accuracy:", acc / float(trainingLabels.shape[0]) * 100., "%")

    def initializeWeight(self, featureCount, labelCount):
        """
        Initialize weights and bias with randomness.

        Do not modify this method.
        """
        self.W = []
        self.b = []
        curNodeCount = featureCount
        self.layerStructure = self.hiddenUnits[:]

        if labelCount == 2:
            self.outAct = sigmoid
            self.loss = binary_crossentropy
            labelCount = 1  # sigmoid function makes the scalar output (one output node)
        else:
            self.outAct = softmax
            self.loss = categorical_crossentropy

        self.layerStructure.append(labelCount)
        self.nLayer = len(self.layerStructure)

        for i in range(len(self.layerStructure)):
            fan_in = curNodeCount
            fan_out = self.layerStructure[i]
            if self.initialWeightBound is None:
                initBound = np.sqrt(6. / (fan_in + fan_out))
            else:
                initBound = self.initialWeightBound
            W = self.numpRng.uniform(-initBound, initBound, (fan_in, fan_out))
            b = self.numpRng.uniform(-initBound, initBound, (fan_out,))
            self.W.append(W)
            self.b.append(b)
            curNodeCount = self.layerStructure[i]

    def forwardPropagation(self, trainingData):
        """

        Fill in this function!

        Arguments : trainingData : (N x D)-sized numpy array
            - N : the number of training instances
            - D : the number of features
        Return : forward propagation result of the neural network

        Propagate forward the neural network, using weight and biases saved in self.W and self.b.
        You may use self.outAct and ReLU for the activation function.

        !! Note the type of weight matrix and bias vector and number o layers:

        ** self.W : list of each layer's weights, while each weights are saved as NumPy array
        ** self.b : list of each layer's biases, while each biases are saved as NumPy array
        ** self.nLayer : number of layers of the network

        Also, for activation functions:

        ** self.outAct: (automatically selected) network output activation function
        ** ReLU: rectified linear unit used for the activation function of hidden layers

        """

        "*** YOUR CODE HERE ***"

        netOut = 0
        return netOut

    def backwardPropagation(self, netOut, trainingLabels, learningRate):
        """

        Fill in this function!

        Arguments:    netOut : forward propagation result of the neural network
                        trainingLabels: (D x C) 0-1 NumPy array
                            - N : the number of training instances
                            - C : the number of legel labels
                        learningRate: python float, learning rate parameter for the gradient descent
        Return: None
                This function back-propagated result of the network weights. 
                That is, update weights and biases in self.W and self.b based on the error.

        ** hint for trainingLabels
        Here, 'trainingLabels' is not a list of labels' index.
        It is converted into a matrix (as a NumPy array) which is filled to 0, but has 1 on its true label.
        Therefore, let's assume i-th data have a true label c, then trainingLabels[i, c] == 1.

        Also note that if this is a binary classification problem, the number of classes 
        which neural network makes is reduced to 1.
        So to match the number of classes, for the binary classification problem, trainingLabels is flatten
        to 1-D array. (Here, let's assume i-th data have a true label c, then trainingLabels[i] == c)

        It looks complicated, but it is simple to use.
        In conclusion, you may use trainingLabels to calcualte the error of the neural network output:

        !! delta = netOut - trainingLabels

        and do backpropagation.

        """

        "*** YOUR CODE HERE ***"

        delta = netOut - trainingLabels

    def classify(self, testData):
        """
        Classify the data based on the trained model.

        Do not modify this method.
        """
        finalNetOut = self.forwardPropagation(testData)

        if self.outAct == softmax:
            return np.argmax(finalNetOut, axis=1)
        elif self.outAct == sigmoid:
            return finalNetOut > 0.5

