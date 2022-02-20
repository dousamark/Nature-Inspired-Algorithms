import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import tensorflow as tf
import cv2, numpy as np


#modely
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception

from tensorflow.python.framework.ops import disable_eager_execution

#pro ResNet50
""" disable_eager_execution() """

#VGG and Res
trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="train",target_size=(224,224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="test", target_size=(224,224))

#Xcep
""" trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="train",target_size=(299,299))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="test", target_size=(299,299))
 """
#VGG
"""
model = VGG16(weights='imagenet', include_top=True)

for layer in model.layers[:19]:
    layer.trainable = False

X= model.layers[-2].output
predictions = Dense(5, activation="softmax")(X)
model_final = Model(model.input, predictions)

model_final.compile(loss = "categorical_crossentropy", optimizer = 'SGD', metrics=["accuracy"])

from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("vgg16.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=40, verbose=1, mode='auto')
model_final.fit_generator(generator= traindata, steps_per_epoch= 2, epochs= 30, validation_data= testdata, validation_steps=1, callbacks=[checkpoint,early])
model_final.save_weights("vgg16.h5") """

#ResNet
""" model = ResNet50(include_top=True,weights='imagenet',input_shape=(224, 224, 3))

for layer in model.layers[:19]:
    layer.trainable = False

X= model.layers[-2].output
predictions = Dense(5, activation="softmax")(X)
model_final = Model(model.input, predictions)

model_final.compile(loss = "categorical_crossentropy", optimizer = 'SGD', metrics=["accuracy"])

from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("res.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=40, verbose=1, mode='auto')
model_final.fit_generator(generator= traindata, steps_per_epoch= 2, epochs= 30, validation_data= testdata, validation_steps=1, callbacks=[checkpoint,early])
model_final.save_weights("res.h5") """

#ResNet 2 
""" model = ResNet50(weights='imagenet', include_top=False)

# Add Custom layers
x = model.output
x = GlobalAveragePooling2D()(x)
# ADD a fully-connected layer
x = Dense(1024, activation='relu')(x)
# Softmax Layer
predictions = Dense(5, activation='softmax')(x)
model = Model(inputs=model.input, outputs=predictions)

# Compile Model
model.compile(optimizer='SGD', loss='categorical_crossentropy',metrics=['accuracy'])

# Train

model.fit_generator(generator= traindata, steps_per_epoch= 2, epochs= 30, validation_data= testdata, validation_steps=1)
model.save_weights("resnet.h5") """

#Xcepiton
""" model = Xception(include_top=True,weights='imagenet',input_shape=(299, 299, 3))

for layer in model.layers[:19]:
    layer.trainable = False

X= model.layers[-2].output
predictions = Dense(5, activation="softmax")(X)
model_final = Model(model.input, predictions)

model_final.compile(loss = "categorical_crossentropy", optimizer = 'SGD', metrics=["accuracy"])

from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("xcep.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=40, verbose=1, mode='auto')
model_final.fit_generator(generator= traindata, steps_per_epoch= 2, epochs= 30, validation_data= testdata, validation_steps=1, callbacks=[checkpoint,early])
model_final.save_weights("xcep.h5") """