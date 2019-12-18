#Run On CoLab
import numpy as np
import pandas as pd
import io
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix


from google.colab import files
uploaded = files.upload()
dataset = pd.read_csv(io.BytesIO(uploaded['Churn_Modelling.csv']))
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values
print(x[0])
print(y[0])

levelEnc_x1=LabelEncoder()
x[:,1]=levelEnc_x1.fit_transform(x[:,1])
print(x[0])
levelEnc_x2=LabelEncoder()
x[:,2]=levelEnc_x2.fit_transform(x[:,2])
print(x[0])

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

y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.5)

cm=confusion_matrix(y_test,y_pred)
cm.view()
