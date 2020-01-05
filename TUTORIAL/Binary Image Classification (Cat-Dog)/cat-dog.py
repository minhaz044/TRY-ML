from keras.models import Sequential
from keras.layers import  Conv2D,Convolution2D,MaxPool2D,Flatten,Dense
import warnings
warnings.filterwarnings("ignore")

#************Model**********
#Define Layer
classifier=Sequential()
#Convolution
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))     #Convolution2D(numberOfFeaturedetector,Row,Column)
#Max Polling
classifier.add(MaxPool2D(pool_size=(2,2)))
#Flattening
classifier.add(Flatten())
#FullConnection 
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


#*****************Data Preprocessing***************#
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
                            rescale=1./255,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                          'training_set/',
                          target_size=(64, 64),
                          batch_size=32,
                          class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
                        'test_set/',
                        target_size=(64, 64),
                        batch_size=32,
                        class_mode='binary')

classifier.fit_generator(
        train_generator,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=validation_generator,
        validation_steps=800)












