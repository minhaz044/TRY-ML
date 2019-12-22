import tensorflow as tf
from tensorflow import keras


imdb_data=keras.datasets.imdb
(train_data,train_labels ),(test_data,test_labels)=imdb_data.load_data(num_words=10000)

#Make them 256 char long ,BCZ variable size  prob for  our Model
train_data=keras.preprocessing.sequence.pad_sequences(train_data,value=0,padding='post',maxlen=256)
test_data=test_data.

model=keras.Sequential()
model.add(keras.layers.Embedding(10000,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16,activation='relu'))
model.add(keras.layers.Dense(1,activation='sigmoid'))

model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train_data,train_labels,batch_size=512,epochs=10,verbose=1)
#model.fit(train_data,train_labels,batch_size=10,epochs=10,validation_data=)
print('Mean :' )
print(accuraces.mean())
print('Varience :' )
print(accuraces.std())






print('..............STOP')