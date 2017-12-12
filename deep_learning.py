#!/home/go96bix/my/programs/Python-3.6.1/bin/python3.6
import pandas
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
import matplotlib
matplotlib.rcParams['backend'] = 'TkAgg'
import matplotlib.pyplot as plt
import os
import numpy as np

# history = []
def data():
    # Loading the MNIST dataset in Keras
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # Preparing the image data
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    # Preparing the labels
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    x_train=    train_images
    y_train = train_labels
    x_test = test_images
    y_test = test_labels
    # return train_images, train_labels, test_images, test_labels
    return x_train, y_train, x_test, y_test

def plotting(history):
    # history_dict = history.history
    # print(history_dict.keys())
    model_hist = history
    acc_values = model_hist[:, 1]
    loss_values = model_hist[:, 2]
    val_acc_values = model_hist[:, 3]
    val_loss_values = model_hist[:, 4]

    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, label='train loss')
    plt.plot(epochs, val_loss_values, label='validation loss')

    plt.plot(epochs, acc_values, label='train accuracy')
    plt.plot(epochs, val_acc_values, label='validation accuracy')


def model(x_train, y_train, x_test, y_test, tensorboard = True, path = "/home/go96bix/projects/Masterarbeit/ML"):
    # The network architecture
    model = models.Sequential()
    # using the Sequential class (only for linear  stacks of layers, which is the most common network architecture by far)
    # "functional API" (for directed acyclic graphs of layers, allowing to build completely arbitrary architectures).

    model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dense(10, activation='softmax')) # 10 because of 10 possible outputs 0,1,2,3...
    # The second layer did not receive an input shape argument --instead it automatically
    # inferred its input shape as being the output shape of the layer that came before.

    # The compilation step
    model.compile(optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])


    # for observing live changes to network
    if tensorboard:
        if not os.path.isdir(path + '/my_log_dir'):
            os.makedirs(path + '/my_log_dir')
        tensorboard = TensorBoard(
            # Log files will be written at this location
            log_dir=path+'/my_log_dir',
            # We will record activation histograms every 1 epoch
            histogram_freq=1,
            # We will record embedding data every 1 epoch
            embeddings_freq=1,
        )

    # Training the network
    hist = model.fit(x_train, y_train, epochs=20, batch_size=256, validation_data=(x_test, y_test),callbacks=[tensorboard])
    # history.append(hist)
    print(hist.history)
    if not os.path.isfile("history.csv"):
        pandas.DataFrame(hist.history).to_csv("history.csv")
    else:
        pandas.DataFrame(hist.history).to_csv("history.csv",mode = 'a',header=False)
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'model': model}


# Evaluating the network
# test_loss, test_acc = network.evaluate(test_images, test_labels)
# print('teTruest_acc:', test_acc)

if __name__ == '__main__':
    # load data
    x_train, y_train, x_test, y_test = data()
    # make and train model
    model(x_train, y_train, x_test, y_test)

    # show results
    history= np.asarray(pandas.read_csv("history.csv"))
    os.remove("history.csv")
    # print(history)
    plotting(history)
    plt.legend(fontsize= 'large')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()