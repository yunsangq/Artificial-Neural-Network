import neuralNetwork_submission

import samples
import sys
import util
import numpy as np
import _pickle as cPickle

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28
FACE_DATUM_WIDTH = 60
FACE_DATUM_HEIGHT = 70


def basicFeatureExtractorDigit(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)
    """
    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0
    return features


def basicFeatureExtractorFace(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is an edge (1) or no edge (0)
    """
    features = util.Counter()
    for x in range(FACE_DATUM_WIDTH):
        for y in range(FACE_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0
    return features


def basicFeatureDataToNumpyArray(basicFeatureData):
    """
    Convert basic feature data(Counter) to N x d numpy array
    """
    basicFeatureData = list(basicFeatureData)

    N = len(basicFeatureData)  # n of samples
    D = len(basicFeatureData[0])  # n of pixels

    keys = list(basicFeatureData[0].keys())

    data = np.zeros((N, D))
    for i in range(N):
        for j in range(D):
            data[i][j] = basicFeatureData[i][keys[j]]

    return data


def runClassifier(dataset, numTraining):
    if dataset == 'faces':
        legalLabels = range(2)
        featureFunction = basicFeatureExtractorFace

        rawTrainingData = samples.loadDataFile("facedata/facedatatrain", numTraining, FACE_DATUM_WIDTH,
                                               FACE_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTraining)
        rawTestData = samples.loadDataFile("facedata/facedatatest", TEST_SET_SIZE, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", TEST_SET_SIZE)

    elif dataset == 'digits':
        legalLabels = range(10)
        featureFunction = basicFeatureExtractorDigit

        rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining, DIGIT_DATUM_WIDTH,
                                               DIGIT_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
        rawTestData = samples.loadDataFile("digitdata/testimages", TEST_SET_SIZE, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", TEST_SET_SIZE)

    else:
        raise Exception

    # load NN classifier
    classifier = neuralNetwork_submission.NeuralNetworkClassifier(legalLabels, "NeuralNetwork", 123)

    # converting data to np.array
    trainingData = basicFeatureDataToNumpyArray(map(featureFunction, rawTrainingData)).astype(np.float32)
    testData = basicFeatureDataToNumpyArray(map(featureFunction, rawTestData)).astype(np.float32)

    print("Training...")
    classifier.train(trainingData, trainingLabels)

    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    print("Performance on the test set:", str(correct),
          ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels)),
          "[Dataset: " + dataset + ", Number of training samples: " + str(numTraining) + "]")


if __name__ == '__main__':
    runClassifier('digits', 500)


