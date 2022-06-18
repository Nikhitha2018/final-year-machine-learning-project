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
df = pd.read_csv("E:\phish\Phishing\phishing.csv")
data = pd.read_csv("C:/Users/hp/desktop/project/urldata.csv")
#droping index column
data = data.drop(['Domain'],axis = 1)
# Splitting the dataset into dependant and independant fetature

y = df['class'].values
X = df.drop('class',axis=1).values 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)

# instantiate the model
gbc = GradientBoostingClassifier(max_depth=4,learning_rate=0.7)

# fit the model 
gbc.fit(X_train,y_train)

# load data
# split data into X and y

# save model to file
pickle.dump(gbc, open("gboost.pickle.dat", "wb"))
 
# some time later...
 
# load model from file
loaded_model = pickle.load(open("gboost.pickle.dat", "rb"))
# make predictions for test data
y_pred = loaded_model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


