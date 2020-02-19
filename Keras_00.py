from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import Flatten
#model = Sequential()
#model.add(Dense(12,input_dim=8, kernel_initializer='random_uniform'))
# Erzeugt ein Layer mit 12 Knoten und 8 Input Knoten
#Dense, bedeutet, dass alle Knoten in der Layer zur Input Layer verbunden sind (Standart NN)
#random_uniform = alles Werte zwischen 0 und 0.05 haben gleiche Chance
#zero = alles Werte gleich Null
#random_normal = Normalverteilung bei 0 bis 0.05


'''Handwrite Tutorial'''

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
np.random.seed(1671)

#network and training
NB_Epoch = 20
BATCH_SIZE = 128
VERBOSE =1
NB_CLASSES = 10 # number of outputs = number of digits
OPTIMIZER = SGD() # SGD optimizer
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2 # Trainsize

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
model = Sequential()
model.add(Dense(NB_CLASSES, input_shape = (RESHAPED,)))
model.add(Activation('softmax'))
model.summary()

# Mögliche Loss Funktion
#MSE - Mean Squared Error, Avareges all Mistakes
#Binary corr-entropy: binary log. loss. For binary labels
#Categorical cross-etnropy: multiclass logarithmic loss. Default for softmax

#Keras metrics:
# accuracy
# Precision
# Recall

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

# fit- functions
# specifies epochs: number of times model ist exposed to the training set
# batch_size : number of training instances observed beofre optimizer performs a weight update

history = model.fit(X_train,y_train, batch_size = BATCH_SIZE, epochs = NB_Epoch, verbose= VERBOSE, validation_split= VALIDATION_SPLIT)

score = model.evaluate(X_test,y_test,verbose=VERBOSE)
print('Test score:',score[0])
print('Test accuracy:', score[1])