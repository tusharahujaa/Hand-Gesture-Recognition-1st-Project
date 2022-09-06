from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

#=================================================================================================================================================================================================================
# Step 1 - PREPROCESSING

#Seprating labels and images
imagepaths1 = []
imagepaths2 = []
# Go through all the files and subdirectories inside a folder and save path to images inside list
for root, dirs, files in os.walk(".", topdown=False):
  for name in files:
    path = os.path.join(root, name)
    path1 = path.split("\\")
    if path.endswith("jpg"): # We want only the images
      path1 = path.split("\\")[2]
      if(path1=="train"):
        imagepaths1.append(path)
      else:
        imagepaths2.append(path)

#---------------------------------------------------------------------------------------------------
#For train data
      
X_train = [] # Image data
y_train= [] # Labels

# Loops through imagepaths to load images and labels into arrays
for path in imagepaths1:
  img = cv2.imread(path) # Reads image and returns np.array
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts into the corret colorspace (GRAY)
  img = cv2.resize(img, (64, 64))
  X_train.append(img)

  # Processing label in image path
  label = path.split("\\")[3]
  y_train.append(label)

      
X_train = np.array(X_train, dtype="uint8")
X_train = X_train.reshape(len(imagepaths1), 64, 64, 1)
y_train = np.array(y_train)

#---------------------------------------------------------------------------------------------------
#For test data

X_test = [] # Image data
y_test= [] # Labels

# Loops through imagepaths to load images and labels into arrays
for path in imagepaths2:
  img = cv2.imread(path) # Reads image and returns np.array
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts into the corret colorspace (GRAY)
  img = cv2.resize(img, (64, 64))
  X_test.append(img)
    
  # Processing label in image path
  label = path.split("\\")[3]
  y_test.append(label)

X_test = np.array(X_test, dtype="uint8")
X_test = X_test.reshape(len(imagepaths2), 64, 64, 1)
y_test = np.array(y_test)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=4)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=4)

#=================================================================================================================================================================================================================
# Step 2 - Building the CNN

# Initializing the CNN
classifier = Sequential()

# First convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding connections the btw conv and dense layers 
classifier.add(Flatten())

# Adding a fully connected layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=4, activation='softmax')) # softmax for more than 2

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2

#=================================================================================================================================================================================================================
# Step 3 - Preparing the train/test data and training the model
History=classifier.fit(X_train,y_train, epochs=20, batch_size=64, verbose=2,validation_data=(X_test,y_test))

#=================================================================================================================================================================================================================
#Loss Graph
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Accuracy Graph
plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Confusion Matrix
predictions = classifier.predict(X_test) # Make predictions towards the test set
y_pred = np.argmax(predictions, axis=1) # Transform predictions into 1-D array with label number
y_test = np.argmax(y_test, axis=1)

cm=confusion_matrix(y_test,y_pred)
cmd_obj = ConfusionMatrixDisplay(cm, display_labels=['Victory', 'Thumbs', 'Fist','Palm'])
cmd_obj.plot()
cmd_obj.ax_.set(
                title='Sklearn Confusion Matrix with labels!!', 
                xlabel='Predicted Values', 
                ylabel='Actual values')
plt.show()

#=================================================================================================================================================================================================================
# Saving the model
model_json = classifier.to_json()
with open("CNNMODEL.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights('CNNMODEL.h5')
