
#model = Sequential()
#model.add(Dense(12,input_dim=8, kernel_initializer='random_uniform'))
# Erzeugt ein Layer mit 12 Knoten und 8 Input Knoten
#Dense, bedeutet, dass alle Knoten in der Layer zur Input Layer verbunden sind (Standart NN)
#random_uniform = alles Werte zwischen 0 und 0.05 haben gleiche Chance
#zero = alles Werte gleich Null
#random_normal = Normalverteilung bei 0 bis 0.05


'''Handwrite Tutorial
Deep Neuronal Networks
'''

from tensorflow import keras

import numpy as np
from keras.datasets import mnist

from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import TensorBoard
import time
np.random.seed(1671)

#network and training
LN_R = 0.001
NB_Epoch = 10
BATCH_SIZE = 128
VERBOSE =1
NB_CLASSES = 10 # number of outputs = number of digits
OPTIMIZER = keras.optimizers.SGD(lr=LN_R) # SGD optimizer
OPTIMIZER2 = keras.optimizers.Adam(lr=LN_R) # SGD optimizer
NB_HIDDEN = 128
VALIDATION_SPLIT = 0.2 # Trainsize
DROPOUT=0.1 # Look how Networks performance, if some of Nodes disapears when calculated
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
RESHAPED = 784

X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_train = X_train.astype('float32')

#normalize
X_train = X_train/255
X_test = X_test/255

y_train = np_utils.to_categorical(y_train,NB_CLASSES)
y_test = np_utils.to_categorical(y_test,NB_CLASSES)
print(X_train.shape[0], ' train samples')
print(X_test.shape[0], ' test samples')

# 10 outputs
# final stage is softmax
model = keras.models.Sequential()
model.add(keras.layers.Dense(NB_CLASSES, input_shape = (RESHAPED,)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(NB_HIDDEN))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(DROPOUT))

model.add(keras.layers.Dense(NB_CLASSES))
model.add(keras.layers.Activation('softmax'))
model.summary()

tensorboard = TensorBoard(log_dir="log\{}".format(round(time.time()))) # tensorboards needs a callback in fit function

# Mögliche Loss Funktion
#MSE - Mean Squared Error, Avareges all Mistakes
#Binary corr-entropy: binary log. loss. For binary labels
#Categorical cross-etnropy: multiclass logarithmic loss. Default for softmax

#Keras metrics:
# accuracy
# Precision
# Recall

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER2, metrics=['accuracy'])

# fit- functions
# specifies epochs: number of times model ist exposed to the training set
# batch_size : number of training instances observed beofre optimizer performs a weight update

model.fit(X_train,y_train, batch_size = BATCH_SIZE, epochs = NB_Epoch,
                    verbose= VERBOSE, validation_split= VALIDATION_SPLIT, callbacks=[tensorboard])

score = model.evaluate(X_test,y_test,verbose=VERBOSE)
print('Test score:',score[0])
print('Test accuracy:', score[1])

# Call tensorboard over Anacondaprompt, go to Project directory and call:
# tensorboard --logdir=log/
#open in browser: http://localhost:6006/














