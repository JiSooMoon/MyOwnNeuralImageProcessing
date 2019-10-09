
import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
tf.random.set_seed(1618)

# Disable some troublesome logging.
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
# ALGORITHM = "custom_net"
ALGORITHM = "tf_net"





class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        return 1/(1+np.exp(-x))
    #    pass   #TODO: implement

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        f = 1/(1+np.exp(-x))
        df = f * (1-f)
        return df
        #pass   #TODO: implement

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100):
        
        epoch = 0
        # print(xVals.shape[0])
        while epoch < epochs:
            # make batch generate
            xbatchGen = self.__batchGenerator(l = xVals, n = mbs)
            ybatchGen = self.__batchGenerator(l = yVals, n = mbs)
            for num in range ((int) (xVals.shape[0] / mbs)):
                print("%d\t%d" % (epoch, num), end="\r" )
                xbatch = next(xbatchGen)
                ybatch = next(ybatchGen)
                L1out, L2out = self.__forward(input = xbatch)

                le2 = ybatch - L2out
                l2d = le2 * self.__sigmoidDerivative(L2out)
                l1e = np.dot(l2d ,np.transpose(self.W2))
                l1d = l1e * self.__sigmoidDerivative(L1out)
                l1a = np.dot(np.transpose(xbatch),  l1d) * self.lr
                l2a = np.dot(np.transpose(L1out) ,l2d) * self.lr

                self.W1 = self.W1 + l1a
                self.W2 = self.W2 + l2a
            epoch = epoch + 1
            
        print('===========================================================')

        # pass                                   #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.
        # return xbatchGen

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        print(xVals)
        _, layer2 = self.__forward(xVals)
        for i in range(layer2.shape[0]):
            ind = np.argmax(layer2[i])
            layer2[i] = 0
            layer2[i][ind] = 1
        return layer2



# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)



#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw            #TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    # print(np.ceil(xTrain/255))
    xTrain = np.ceil(xTrain/255)
    xTrain = np.reshape(xTrain,(xTrain.shape[0], xTrain.shape[1] * xTrain.shape[2]))
    xTest = np.ceil(xTest/255)
    xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1] * xTest.shape[2]))
    
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data):
    
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        # self.train(xTrain,yTrain)
        print("Building and training Custom_NN.")
        # print(xTrain.shape)
        neuralModel = NeuralNetwork_2Layer(inputSize= IMAGE_SIZE, outputSize= NUM_CLASSES, neuronsPerLayer = 512)
        neuralModel.train(xVals = xTrain,yVals = yTrain, epochs = 3)
        
        # print("Not yet implemented.")                   #TODO: Write code to build and train your custon neural net.
        return neuralModel
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), tf.keras.layers.Dense(IMAGE_SIZE, activation=tf.nn.relu)
        ,tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax)])
        model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        # model = keras.Sequential()
        # lossType = keras.losses.categorical_crossentropy
        # opt = tf.train.AdamOptimizer()
        # inshape = (IMAGE_SIZE,)
        # model.add(keras.layers.Dense(NUM_CLASSES, input_shape = inshape, activation=tf.nn.softmax))
        # model.compile(optimizer = opt, loss = lossType) #building
        print("===============================")
        model.fit(x = xTrain, y = yTrain, batch_size=100,  epochs = 3)
        
        print("***********************************")
        # print("Not yet implemented.")                   #TODO: Write code to build and train your keras neural net.
        return model
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        # print("Not yet implemented.")                   #TODO: Write code to run your custon neural net.
        print(model.predict(data))
        return model.predict(data)
        # return None
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        # print("Not yet implemented.")                   #TODO: Write code to run your keras neural net.
        # return model.evaluate()
        
        # for i in range(data.shape[0]):
        #     ind = np.argmax(data[i])
        #     data[i] = 0
        #     data[i][ind] = 1
        print(data[0])
        return data
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()


#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    # print("here")
    main()
