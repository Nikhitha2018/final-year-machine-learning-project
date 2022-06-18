
import numpy as np
import pandas as pd
from sklearn import metrics 
import pickle
import warnings
warnings.filterwarnings('ignore')
from numpy import loadtxt
import xgboost
import pickle
from sklearn import model_selection
from sklearn.metrics import accuracy_score
# Gradient Boosting Classifier Model
from sklearn.ensemble import GradientBoostingClassifier
df = pd.read_csv("E:\phish\Phishing\phish_book.csv")
data = pd.read_csv("C:/Users/hp/desktop/project/urldata.csv")
#droping index column
data = data.drop(['Domain'],axis = 1)
# Splitting the dataset into dependant and independant fetature

y = df['class'].values
X = df.drop('class',axis=1).values 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from keras.activations import tanh
# Defining Generator Model
latent_dim =15
D = 15
def build_generator(latent_dim):
  i = Input(shape=(latent_dim,))
  x = Dense(256, activation=LeakyReLU(alpha=0.2))(i)
  x = BatchNormalization(momentum=0.7)(x)
  x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
  x = BatchNormalization(momentum=0.7)(x)
  x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
  x = BatchNormalization(momentum=0.7)(x)
  x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
  x = BatchNormalization(momentum=0.7)(x)
  x = Dense(D, activation='LeakyReLU')(x)  #because Image pixel is between -1 to 1.
  model = Model(i, x)  #i is input x is output layer
  return model
i = Input(shape=(latent_dim,))
x = Dense(256, activation=LeakyReLU(alpha=0.2))(i)
x = BatchNormalization(momentum=0.7)(x)
x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
x = BatchNormalization(momentum=0.7)(x)
x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
x = BatchNormalization(momentum=0.7)(x)
x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
x = BatchNormalization(momentum=0.7)(x)
x = Dense(D, activation='LeakyReLU')(x)  #because Image pixel is between -1 to 1.
model = Model(i, x)  #i is input x is output layer
model.summary()

def build_discriminator(img_size):
  i = Input(shape=(img_size,))
  x = Dense(512, activation=LeakyReLU(alpha=0.2))(i)
  x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
  x = Dense(1, activation='sigmoid')(x)
  model = Model(i, x)
  
  return model
img_size=15
i = Input(shape=(img_size,))
x = Dense(512, activation=LeakyReLU(alpha=0.2))(i)
x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(i, x)
model.summary()

# Build and compile the discriminator
D=15   # insert the dataset shape i.e., (None,16) in the Model
discriminator = build_discriminator(D)
discriminator.compile ( loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5), metrics=['accuracy'])
# Build a
# ator(img)
combined_model_gen = Model(z, fake_pred)  #first is noise and 2nd is fake prediction
# Compile the combined model
combined_model_gen.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5))
batch_size = 64
epochs = 10
sample_period =200
ones = np.ones(batch_size)
zeros = np.zeros(batch_size)
#store generator and discriminator loss in each step or each epoch
d_losses = []
g_losses = []
daccuracy=[]
#FIRST we will train Discriminator(with real urls and fake urls)
# Main training loop
for epoch in range(0,epochs):
  ###########################
  ### Train discriminator ###
  ###########################
  # Select a random batch of images
  idx = np.random.randint(0,X_train.shape[0], batch_size)
  real_imgs = X_train[idx] 
  # Generate fake images
  noise = np.random.randn(batch_size, latent_dim)  #generator to generate fake imgs
  fake_imgs = generator.predict(noise)
  # Train the discriminator
  # both loss and accuracy are returned
  d_loss_real, d_acc_real = discriminator.train_on_batch(real_imgs, ones)  #belong to positive class(real imgs)
  d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_imgs, zeros)  #fake imgs
  d_loss = 0.5 * (d_loss_real + d_loss_fake)
  d_acc  = 0.5 * (d_acc_real + d_acc_fake)
  daccuracy.append(d_acc)
  #######################
  ### Train generator ###
  #######################
  noise = np.random.randn(batch_size, latent_dim)
  g_loss = combined_model_gen.train_on_batch(noise, ones)  
  #Now we are trying to fool the discriminator that generate imgs are real that's why we are providing label as 1
  # do it again!
  noise = np.random.randn(batch_size, latent_dim)
  g_loss = combined_model_gen.train_on_batch(noise, ones)
  # Save the losses
  d_losses.append(d_loss)  #save the loss at each epoch
  g_losses.append(g_loss)
  if epoch:
    print(f"epoch: {epoch+1}/{epochs}, d_loss: {d_loss:.2f}, d_acc: {d_acc:.2f}, g_loss: {g_loss:.2f}")

combined_model_gen.fit(X_train,y_train)
import joblib
filename = "Completed_model.joblib"
joblib.dump(combined_model_gen, filename)