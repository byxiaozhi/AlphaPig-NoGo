from keras.backend import set_session
from keras.regularizers import L2
from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, ReLU, add, Reshape, Softmax
from keras.models import load_model, Model
import random
import numpy as np
import tensorflow as tf
import os
import struct

modelPathName = 'model'
trainPathName = 'dataset'


def ConvolutionalLayer(x, filters=64):
    x = Conv2D(filters=filters, kernel_size=3, data_format='channels_first',
               padding='same', use_bias=False, activation='linear', kernel_regularizer=L2(0.0001))(x)
    x = BatchNormalization(axis=1)(x)
    x = ReLU()(x)
    return x


def ResidualLayer(in_x, filters=64):
    x = ConvolutionalLayer(in_x, filters)
    x = Conv2D(filters=filters, kernel_size=3, data_format='channels_first',
               padding='same', use_bias=False, activation='linear', kernel_regularizer=L2(0.0001))(x)
    x = BatchNormalization(axis=1)(x)
    x = add([in_x, x])
    x = ReLU()(x)
    return x


def ValueHead(x, units=64):
    x = Conv2D(filters=1, kernel_size=1, data_format='channels_first', padding='same',
               use_bias=False, activation='linear', kernel_regularizer=L2(0.0001))(x)
    x = BatchNormalization(axis=1)(x)
    x = ReLU()(x)
    x = Flatten()(x)
    x = Dense(units, use_bias=False, activation='linear',
              kernel_regularizer=L2(0.0001))(x)
    x = ReLU()(x)
    x = Dense(1, name='valueHead', use_bias=False, activation='tanh',
              kernel_regularizer=L2(0.0001))(x)
    return x


def PolicyHead(x, w, h):
    x = Conv2D(
        filters=1, kernel_size=1, data_format='channels_first', padding='same', use_bias=False, activation='linear', kernel_regularizer=L2(0.0001))(x)
    x = Flatten()(x)
    x = Softmax(name='policyHead')(x)
    return x


def BuildModel(channel, size) -> Model:
    input = Input(shape=(channel, size, size))
    x = ConvolutionalLayer(input)
    for i in range(5):
        x = ResidualLayer(x)
    x = ConvolutionalLayer(x)
    valueHead = ValueHead(x)
    policyHead = PolicyHead(x, size, size)
    model = Model(inputs=[input], outputs=[valueHead, policyHead])
    model.compile(
        loss={'valueHead': 'mean_squared_error',
              'policyHead': 'categorical_crossentropy'},
        loss_weights={'valueHead': 0.5, 'policyHead': 0.5},
        optimizer=SGD(learning_rate=0.01, momentum=0.9))
    return model


def writeConv(outfile, layer):
    for x in np.array(layer.weights).transpose(0, 4, 3, 1, 2).flatten():
        outfile.write(struct.pack('f', x))


def writeNormalize(outfile, layer):
    weights = np.array(layer.weights)
    for i in [1, 0, 2, 3]:
        for x in weights[i].flatten():
            outfile.write(struct.pack('f', x))


def writeDense(outfile, layer):
    for x in np.array(layer.weights).transpose(0, 2, 1).flatten():
        outfile.write(struct.pack('f', x))


def saveModel(filePath):
    with open(filePath, 'wb') as outfile:
        outfile.write(struct.pack('iiiii', 0, 0, 0, 0, 0))
        writeNormalize(outfile, model.layers[2])
        writeConv(outfile, model.layers[1])
        for i in range(5):
            writeNormalize(outfile, model.layers[i * 7 + 5])
            writeConv(outfile, model.layers[i * 7 + 4])
            writeNormalize(outfile, model.layers[i * 7 + 8])
            writeConv(outfile, model.layers[i * 7 + 7])
        writeNormalize(outfile, model.layers[40])
        writeConv(outfile, model.layers[39])
        writeNormalize(outfile, model.layers[43])
        writeConv(outfile, model.layers[42])
        for i in range(np.array(model.layers[46].weights).shape[2]):
            outfile.write(struct.pack('f', 0))
        writeDense(outfile, model.layers[46])
        for i in range(np.array(model.layers[50].weights).shape[2]):
            outfile.write(struct.pack('f', 0))
        writeDense(outfile, model.layers[50])
        outfile.write(struct.pack('f', 0))
        writeConv(outfile, model.layers[47])


print('Loading model...')
if os.path.exists(modelPathName):
    model = load_model(modelPathName)
else:
    model = BuildModel(6, 9)
    model.save(modelPathName)
    saveModel('network.weights')

print('Loading data...')
trainDatas = []
for file in os.listdir(trainPathName):
    if os.path.exists(os.path.join(trainPathName, file)):
        trainDatas.append(np.loadtxt(os.path.join(trainPathName, file)))
trainDatas = np.concatenate(trainDatas, axis=0)

print('Start training...')
for i in range(4):
    trainData = trainDatas[np.random.choice(
        trainDatas.shape[0], 256 * 1000, replace=False)]
    trainInput = trainData[:, :6*9*9].reshape(-1, 6, 9, 9)
    tarinValueOutput = trainData[:, 7*9*9:9*7*9+1].reshape(-1, 1)
    tarinPolicyOutput = trainData[:, 6*9*9:7*9*9].reshape(-1, 9, 9)
    for i in range(len(trainData)):
        times = random.randint(1, 4)
        trainInput[i] = np.rot90(trainInput[i], times, [1, 2])
        tarinPolicyOutput[i] = np.rot90(tarinPolicyOutput[i], times, [0, 1])
        if random.choice([True, False]):
            trainInput[i] = np.flip(trainInput[i], [1])
            tarinPolicyOutput[i] = np.flip(tarinPolicyOutput[i], [0])
    tarinPolicyOutput = tarinPolicyOutput.reshape(-1, 81)
    model.fit(trainInput, [tarinValueOutput,
                           tarinPolicyOutput], epochs=1, batch_size=256)

print('Save model...')
model.save(modelPathName)
saveModel('network.weights')
