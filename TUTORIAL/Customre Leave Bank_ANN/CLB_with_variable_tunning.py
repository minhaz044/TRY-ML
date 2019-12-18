import numpy as np
import pandas as pd
import io
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from google.colab import files



uploaded = files.upload()
dataset = pd.read_csv(io.BytesIO(uploaded['Churn_Modelling.csv']))
#Preduploaded = files.upload()
#predDataSet = pd.read_csv(io.BytesIO(Preduploaded['RealDataPred.csv']))
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

#Data Preprocess
levelEnc_x1=LabelEncoder()
x[:,1]=levelEnc_x1.fit_transform(x[:,1])
levelEnc_x2=LabelEncoder()
x[:,2]=levelEnc_x2.fit_transform(x[:,2])

oneHotEnc=OneHotEncoder(categorical_features=[1])
x=oneHotEnc.fit_transform(x).toarray()
x=x[:,1:]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
standardScaler=StandardScaler()
x_train=standardScaler.fit_transform(x_train)
x_test=standardScaler.fit_transform(x_test)

classifier=Sequential()
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(x_train,y_train,batch_size=10,epochs=100)

#Model
def buildModel(optimizer):
  classifier=Sequential()
  classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
  classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
  classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
  classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
  return classifier
'''
classifier=KerasClassifier(build_fn=buildModel,batch_size=10,epochs=10)
accuraces=cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=10,n_jobs=-1)
print('Mean :' )
print(accuraces.mean())
print('Varience :' )
print(accuraces.std())
'''
classifier=KerasClassifier(build_fn=buildModel)

parameter={
    'batch_size':[25,32],
    'nb_epoch':[100,500],
    'optimizer':['adam','rmsprop']
}

gridSearch=GridSearchCV(estimator=classifier,param_grid=parameter,scoring='accuracy',cv=10)
gridSearch=gridSearch.fit(x_train,y_train)
best_param=gridSearch.best_params
best_accuracy=gridSearch.best_score_

print(best_param)
print(best_accuracy)
#Predict
#y_pred=classifier.predict(x_test)
#y_pred=(y_pred>0.5)


#cm=confusion_matrix(y_test,y_pred)
#cm.view()



