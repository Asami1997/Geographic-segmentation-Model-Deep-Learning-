# ANN
# install theano (Done)
# install tensorflow(Done)
# install keras(Done)


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# Indeoendent variables
X = dataset.iloc[:, 3:13].values
# Dependent variables
y = dataset.iloc[:, 13].values

#check if there are any null values in the dataset 
pd.isnull(dataset).values.any()

#Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Encoding the country column 
label_encoder_country = LabelEncoder()
X[:, 1] = label_encoder_country.fit_transform(X[:, 1])

#Encoding the gender column
label_encoder_gender = LabelEncoder()
X[: , 2] = label_encoder_gender.fit_transform(X[:, 2])

# Encoding country column into dummy variables
hot_encoder_country =  OneHotEncoder(categorical_features = [1])
X = hot_encoder_country.fit_transform(X).toarray()


X = X[:, 1: ]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
# THis function only builds the archeticture of the classifier
def build_classifier_arch(optimizer):
   # Initialize ANN as a sequential model
   classifier = Sequential()
   # Add input layer and the first hidden layer
   classifier.add(Dense (units = 6, kernel_initializer ='uniform', activation = 'relu' , input_dim =11))
   # Add the second hidden layer
   classifier.add(Dense (units = 6, kernel_initializer ='uniform', activation = 'relu'))
   # Add the output layer
   classifier.add(Dense (units = 1, kernel_initializer ='uniform', activation = 'sigmoid'))
   # Compiling the ANN by applying the stochastic gradient descent
   # the accurcy is in brackets because the metrics argument expects a list of metrics but here we only want the accurecy metric
   classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
   return classifier
 
# Train classifier using K-FOLD cross validation 
# special kind of classifier that can warp the cross validation method
classifier_k = KerasClassifier(build_fn = build_classifier_arch)

# A dictionary that will contain all the keys and values of the hyperparameter when want to try
parameters = {'batch_size': [25,32],
              'epochs':[100, 500],
              'optimizer':['adam','rmsprop']}

# Implement Grid Search

gridsearch = GridSearchCV(estimator = classifier_k,
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv = 10)
# will fit the ann into the training set while running a grid search to find the optimal hyper params
gridsearch = gridsearch.fit(X_train, y_train)
best_parametrs = gridsearch.best_params_
# best accuracy resulting from best selection
best_accuracy = gridsearch.best_score_

y_pred = gridsearch.predict(X_test)

from keras.models import load_model

gridsearch.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'


