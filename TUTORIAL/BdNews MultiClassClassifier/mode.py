import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from google.colab import files
import io
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from keras.utils import to_categorical

uploaded = files.upload()
file=io.BytesIO(uploaded['test.csv'])
print('File Upload ')
dataset = pd.read_csv(file).replace(np.nan,0)
x=dataset.iloc[:(1149),1:].values
y=dataset.iloc[:(1149),:1].values
forTest=dataset.iloc[1:10,1:].values



#y= to_categorical(y)
print(y.shape)
print(x.shape)



x=keras.preprocessing.sequence.pad_sequences(x,value=0,padding='post',maxlen=200)

train_data, test_data, train_labels, test_labels = train_test_split(x, y, test_size=0.25)

#imdb_data=keras.datasets.imdb
#(train_data,train_labels ),(test_data,test_labels)=imdb_data.load_data(num_words=10000)

#Make them 256 char long ,BCZ variable size  prob for  our Model
#train_data=keras.preprocessing.sequence.pad_sequences(train_data,value=0,padding='post',maxlen=200)
#test_data=test_data.

model=keras.Sequential()
model.add(keras.layers.Embedding(100000,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dense(5,activation='softmax'))

model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_data,train_labels,batch_size=700,shuffle=True,epochs=20,verbose=1)
#model.fit(train_data,train_labels,batch_size=10,epochs=10,validation_data=)
#print('Mean :' )
#print(accuraces.mean())
#print('Varience :' )
#print(accuraces.std())
print('===================Predict============')

inputIndex=0
one,two,three,four=0,0,0,0
result=0;

prediction=model.predict_classes(forTest,verbose=1)
for i in prediction:
  print(i)
# print('STOP')
# while 0 < 10000000000:
#   inputIndex = input('Enter Data Index')
#   prediction=model.predict_classes(x)
#   for i in prediction:
#     if(i==1):
#       one=+1
#     elif (i==2):
#       two+=1
#     elif (i==3):
#       three+=1
#     elif (i==4):
#       four+=1
#   print('One=>' + str(one) + 'two=>' + str(two) +'three=>' + str(three) +'four=>' + str(four) + 'Actual Result:' + str(y[int(inputIndex)]))
#   one,two,three,four=0,0,0,0
#   print("Data Reset")



print('..............STOP')