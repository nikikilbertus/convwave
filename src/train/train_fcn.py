# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import numpy as np
import h5py
import datetime

from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Reshape, BatchNormalization, \
    Activation, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, Callback, TensorBoard

# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    data_path = '../data/'
    print('Starting main routine...')

    # -------------------------------------------------------------------------
    # LOAD DATA AND SPLIT TRAINING AND TEST SAMPLE
    # -------------------------------------------------------------------------

    filename = os.path.join(data_path, 'training_samples.h5')

    with h5py.File(filename, 'r') as file:

        x = np.array(file['samples'])
        y = np.array(file['labels'])

    def whiten(image):
        mittelwert = 18.4455575155
        standardabweichung = 3.66447104786
        return (image - mittelwert) / standardabweichung

    x = np.array([whiten(_) for _ in x])

    # Reshape to make it work with keras
    x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    y = y.reshape((y.shape[0], y.shape[1], 1)).astype('int')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,
                                                        random_state=42)

    # -------------------------------------------------------------------------
    # DEFINE THE MODEL
    # -------------------------------------------------------------------------

    print('Defining the model...')

    print(x_train.shape)
    print()

    model = Sequential()
    # -------------------------------------------------------------------------
    model.add(Conv2D(128, (3, 7),
                     input_shape=x_train[0].shape,
                     data_format='channels_last',
                     padding='same',
                     kernel_initializer='random_uniform',
                     name='Start'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    # -------------------------------------------------------------------------
    for i in range(6):
        model.add(Conv2D(128, (3, 7),
                         padding='same',
                         kernel_initializer='random_uniform',
                         name=str(i)))
        model.add(BatchNormalization())
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Dropout(rate=0.3))
    # -------------------------------------------------------------------------
    model.add(Conv2D(1, (1, 1),
                     activation='sigmoid',
                     kernel_initializer='random_uniform',
                     name="Ende"))
    # -------------------------------------------------------------------------
    model.add(Reshape((513, 1),
                      name="Reshape"))


    def false_ratio(y_true, y_pred):
        return K.mean(np.abs(y_pred - y_true))

    optimizer = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=[false_ratio])

    # -------------------------------------------------------------------------
    # CUSTOM CALLBACK FOR THE LEARNING RATE
    # -------------------------------------------------------------------------

    class PrintLearningRate(Callback):
        def on_epoch_begin(self, epoch, logs=None):
            lr = K.eval(self.model.optimizer.lr)
            print('\nLearning Rate:', lr, end='\n', flush=True)
    print_learning_rate = PrintLearningRate()

    # -------------------------------------------------------------------------
    # CALLBACK TO REDUCE LEARNING RATE
    # -------------------------------------------------------------------------

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.707, patience=8,
                                  epsilon=1e-03, min_lr=1e-08)

    # -------------------------------------------------------------------------
    # CALLBACK FOR TENSORBOARD
    # -------------------------------------------------------------------------

    tensorboard = TensorBoard(log_dir='./logs/{:%Y-%m-%d_%H:%M:%S}'.
                              format(datetime.datetime.now()))

    # -------------------------------------------------------------------------
    # FIT THE MODEL AND SAVE THE WEIGHTS
    # -------------------------------------------------------------------------

    model.fit(x_train, y_train,
              batch_size=16,
              epochs=50,
              validation_split=0.1,
              callbacks=[reduce_lr, print_learning_rate, tensorboard])
    model.save_weights('model_weights.h5')

    # -------------------------------------------------------------------------
    # MAKE PREDICTIONS AND EVALUATE THE ACCURACY
    # -------------------------------------------------------------------------

    np.set_printoptions(threshold=np.inf)
    y_pred = np.round(model.predict(x_test))
    # print(y_pred.squeeze())

    correct = 0
    ratio = []
    for i in range(len(y_test)):
        if all(y_pred[i] == y_test[i]):
            correct += 1
        ratio.append(np.sum(np.abs(y_pred[i] - y_test[i])) / len(y_test[i]))

    print()
    print('Accuracy: {:.3f}'.format(correct / len(y_test)))
    print('Average Ratio: {:.3f}'.format(np.mean(ratio)))

    filename = os.path.join(data_path, 'test_predictions.h5')
    with h5py.File(filename, 'w') as file:

        file['x'] = np.array(x_test)
        file['y'] = np.array(y_pred.squeeze())