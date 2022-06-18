import keras
from keras.layers import Input, Dense
from keras import regularizers
import tensorflow as tf
from keras.models import Model
import numpy as np
from sklearn import metrics
import numpy as np
import pandas as pd
import pickle
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
import glob
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
df = pd.read_csv(r"C:\Users\hp\Desktop\phish-main\Phishing\phishing.csv")

# Splitting the dataset into dependant and independant fetature

y = df['class'].values
X = df.drop('class',axis=1).values 

print(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)
input_dim = X_train.shape[1]                                                      
encoding_dim = input_dim

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh",
                activity_regularizer=regularizers.l1(10e-4))(input_layer)
encoder = Dense(int(encoding_dim), activation="tanh")(encoder)

encoder = Dense(int(encoding_dim-2), activation="tanh")(encoder)
code = Dense(int(encoding_dim-4), activation="tanh")(encoder)        #bottleneck layer
decoder = Dense(int(encoding_dim-2), activation='tanh')(code)

decoder = Dense(int(encoding_dim), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='tanh')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.summary()

autoencoder.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

#Training the model
autoencoder.fit(X_train, X_train, epochs=10, batch_size=64, shuffle=True, validation_split=0.2) 

acc_train_auto = autoencoder.evaluate(X_train, X_train)[1]
acc_test_auto = autoencoder.evaluate(X_test, X_test)[1]

print('\nAutoencoder: Accuracy on training Data: {:.3f}' .format(acc_train_auto*100))
print('Autoencoder: Accuracy on test Data: {:.3f}' .format(acc_test_auto*100))

# saving and loading model architecture in json format

# saving in json format
json_model = autoencoder.to_json()
json_file = open('autoencoder_json.json', 'w')
json_file.write(json_model)

# loading model architecture from json file
from keras.models import model_from_json
json_file = open('autoencoder_json.json', 'r')
json_model = model_from_json(json_file.read())

# saving and loading model architecture in yaml format

# saving in yaml format
yaml_model = autoencoder.to_yaml()
yaml_file = open('autoencoder_yaml.yaml', 'w')
yaml_file.write(yaml_model)

# loading model architecture from yaml file
from keras.models import model_from_yaml
yaml_file = open('autoencoder_yaml.yaml', 'r')
yaml_model = model_from_yaml(yaml_file.read())

# saving model weights
autoencoder.save_weights('autoencoder_weights.h5')

# loading weights of a keras model
json_model.load_weights('autoencoder_weights.h5')

# saving whole model
autoencoder.save('autoencoder_model.h5')

# loading whole model
from keras.models import load_model
model1 = load_model('autoencoder_model.h5')