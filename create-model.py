import pandas as pd
import cv2 as cv
import numpy as np
import random
from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, MaxPooling2D
from keras.optimizers import Adam
     
#%%

model=Sequential()
model.add(Conv2D(16,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy',
  optimizer='rmsprop',metrics=['accuracy'])

#%%

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator,load_img
from sklearn.model_selection import train_test_split


earlystop = EarlyStopping(patience = 10)
learning_rate_decrease = ReduceLROnPlateau(monitor = 'val_acc',patience = 2,verbose = 1,factor = 0.5,min_lr = 0.00001)
callbacks = [earlystop,learning_rate_decrease]

train_datagen = ImageDataGenerator(rotation_range=15,
                                rescale=1./255,
                                shear_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1
                                )

train_label_generator = pd.read_csv("processed_train\\eliminated\\datasetFinal.csv")

train_label_generator['label'] = train_label_generator['label'].replace([-1],0)

train_label_generator["label"] = train_label_generator["label"].replace({0:'notbanana',1:'BANANA'})
train_df,validate_df = train_test_split(train_label_generator,test_size=0.20,
  random_state=42)

train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
total_train=train_df.shape[0]
total_validate=validate_df.shape[0]
batch_size=15

train_generator = train_datagen.flow_from_dataframe(train_df,
                                                 "processed_train\\eliminated\\all",x_col='name',y_col='label',
                                                 target_size=(32, 32),
                                                 class_mode='categorical',
                                                 batch_size=batch_size)


validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "processed_train\\eliminated\\all", 
    x_col='name',
    y_col='label',
    target_size=(32, 32),
    class_mode='categorical',
    batch_size=batch_size
)

epochs=200
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)

model.save("model.h5")

#%%
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

train_accuracy = model.history.history['accuracy']
val_accuracy = model.history.history['val_accuracy']
epochs = np.arange(1, len(val_accuracy)+1)

# plotting accuracy
plt.title("Accuracy Plot")
plt.xlabel("Epochs")
plt.ylabel("")
plt.plot(epochs, train_accuracy, color ="#FD9A02", label='Training Accuracy')
plt.plot(epochs, val_accuracy, color ="blue", label='Validation Accuracy')
plt.legend()
plt.show()



