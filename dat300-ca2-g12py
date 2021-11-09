#!/usr/bin/env python
# coding: utf-8

# In[1]:


### This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math
import h5py

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import h5py
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


# Load dataset from the train file 
trn_raw = h5py.File('/kaggle/input/ca2-data/tree_train.h5', 'r')


# In[3]:


trn_raw_X = trn_raw.get('X')
trn_raw_y = trn_raw.get('y')


# In[4]:


X_trn = trn_raw_X[0:3500] 
y_trn = trn_raw_y[0:3500]
X_val = trn_raw_X[3500:4000]
y_val = trn_raw_y[3500:4000]
X_all = trn_raw_X
y_all = trn_raw_y


# In[5]:


import cv2
plt.figure(figsize = (15,15))
for i in range(9):
    plt.subplot(330 + 1 + i) # Shorthand for size 3x3, position i
    image = X_trn[i]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Reverse RGB order
    plt.imshow(image)
plt.show()


# In[6]:


# The visualization shows that the images have variable intensity - This could create difficulties for the learning.
# Therefore, visualization was made

mean = np.mean(X_trn, axis=(1,2), keepdims=True)
std = np.std(X_trn, axis=(1,2), keepdims=True)
X_trn = (X_trn - mean) / std

mean = np.mean(X_val, axis=(1,2), keepdims=True)
std = np.std(X_val, axis=(1,2), keepdims=True)
X_val = (X_val - mean) / std


# In[7]:


# Model configuration
batch_size = 128
img_width, img_height, img_num_channels = 128, 128, 3
loss_function = binary_crossentropy
no_classes = 1
no_epochs = 25
optimizer = RMSprop()
validation_split = 0.2
verbosity = 0


# In[8]:


input_train = X_trn.reshape((len(X_trn), img_width, img_height, img_num_channels))
input_test  = X_val.reshape((len(X_val), img_width, img_height, img_num_channels))


# In[9]:


input_shape = (img_width, img_height, img_num_channels)


# In[10]:


"""
Version of U-Net with dropout and size preservation (padding= 'same')
These functions are copied from Kristian's lecture and modified with regularization and learning rate
""" 
def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),              kernel_initializer = 'he_normal', padding = 'same', kernel_regularizer='L2')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),              kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x


def get_unet(input_img, n_filters = 16, dropout = 0.1, batchnorm = True, n_classes = 1, lr = 0.1):
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    model.compile(optimizer = RMSprop(learning_rate = lr), loss = binary_crossentropy, metrics = ['accuracy'])
    return model


# In[11]:


from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
input_img = Input(shape=(128,128,3))
model = get_unet(input_img, n_filters = 32, dropout = 0, batchnorm = True, n_classes = 1, lr = 0.001)


# In[ ]:


# Fit data to model

history = model.fit(input_train, y_trn,
            batch_size=64,
            epochs=11,
            shuffle=True,
            verbose=1,
            validation_split=validation_split)


# In[ ]:


# Generate generalization metrics
score = model.evaluate(input_test, y_val, verbose=1)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

y_pred = model.predict(input_test)


# In[ ]:


i=1
tst = np.round(y_pred,0)
for num in range(4):   
    plt.figure(figsize = (30,30))
    plt.subplot(420+i) # Shorthand for size 3x3, position i
    i+=1
    image = tst[num]*255
    plt.title('Predicted class')
    plt.imshow(image, cmap = 'gray')
    plt.subplot(420+i)
    image = y_val[num]*255
    plt.title('Ground truth')
    plt.imshow(image, cmap = 'gray')
    i+=1
plt.show()


# In[14]:


from tensorflow.keras.applications.vgg16 import VGG16
conv_base = VGG16(weights='imagenet', # Pre-trained on ImageNet data
                  include_top=False,        # Remove classification layer
                  input_shape=(128, 128, 3))  
for layer in conv_base.layers:
    layer.trainable = False


# In[15]:


"""
Version of U-Net with dropout and size preservation (padding= 'same')
These functions are copied from Kristian's lecture and modified with regularization and learning rate
""" 
def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),              kernel_initializer = 'he_normal', padding = 'same', kernel_regularizer='L2')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),              kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x


def get_unet_pretrained_2(input_img, n_filters = 16, dropout = 0.1, batchnorm = True, n_classes = 1, lr = 0.1):
    # Contracting Path
    
    conv_base = VGG16(weights='imagenet', # Pre-trained on ImageNet data
                  include_top=False,        # Remove classification layer
                  input_tensor=input_img)  
    for layer in conv_base.layers:
        layer.trainable = False
    s1 = conv_base.get_layer("block1_conv1").output
    p1 = MaxPooling2D((2, 2))(s1)
    p1 = Dropout(dropout)(p1)
    s1 = conv_base.get_layer("block1_conv2").output
    p2 = MaxPooling2D((2, 2))(s1)
    p2 = Dropout(dropout)(p2)                             ## (512 x 512)
    s2 = conv_base.get_layer("block2_conv2").output
    p3 = MaxPooling2D((2, 2))(s2)
    p3 = Dropout(dropout)(p3)                              ## (512 x 512)## (256 x 256)
    s3 = conv_base.get_layer("block3_conv3").output         ## (128 x 128)
    p4 = MaxPooling2D((2, 2))(s3)
    p4 = Dropout(dropout)(p4) 
    s4 = conv_base.get_layer("block4_conv3").output 
    
    b1 = conv_base.get_layer("block5_conv3").output         ## (32 x 32)
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(b1)
    u6 = concatenate([u6, s4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, s3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, s2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, s1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid', kernel_regularizer='L2')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    model.compile(optimizer = RMSprop(learning_rate = lr), loss = binary_crossentropy, metrics = ['accuracy'])
    return model


# In[ ]:


from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
input_img = Input(shape=(128,128,3))
model_pretrained = get_unet_pretrained_2(input_img, n_filters = 32, dropout = 0, batchnorm = True, n_classes = 1, lr = 0.0002)


# In[ ]:


history = model_pretrained.fit(input_train, y_trn,
            batch_size=64,
            epochs=10,
            shuffle=True,
            verbose=1,
            validation_split=validation_split)


# In[ ]:


# Generate generalization metrics
score_pretrained = model_pretrained.evaluate(input_test, y_val, verbose=1)
print(f'Test loss: {score_pretrained[0]} / Test accuracy: {score_pretrained[1]}')

y_pred_pretrained = model_pretrained.predict(input_test)
i=1
tst = np.round(y_pred_pretrained,0)
for num in range(4):   
    plt.figure(figsize = (30,30))
    plt.subplot(420+i) # Shorthand for size 3x3, position i
    i+=1
    image = tst[num]*255
    plt.title('Predicted class')
    plt.imshow(image, cmap = 'gray')
    plt.subplot(420+i)
    image = y_val[num]*255
    plt.title('Ground truth')
    plt.imshow(image, cmap = 'gray')
    i+=1
plt.show()


# In[ ]:


y_pred_pretrained = model_pretrained.predict(input_test)
i=1
tst = np.round(y_pred_pretrained,0)
for num in range(4):   
    plt.figure(figsize = (30,30))
    plt.subplot(420+i) # Shorthand for size 3x3, position i
    i+=1
    image = tst[num]*255
    plt.title('Predicted class')
    plt.imshow(image, cmap = 'gray')
    plt.subplot(420+i)
    image = y_val[num]*255
    plt.title('Ground truth')
    plt.imshow(image, cmap = 'gray')
    i+=1
plt.show()


# In[16]:


from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
input_img = Input(shape=(128,128,3))
model_pretrained_2 = get_unet_pretrained_2(input_img, n_filters = 64, dropout = 0.01, batchnorm = True, n_classes = 1, lr = 0.0002)


# In[17]:



history = model_pretrained_2.fit(input_train, y_trn,
            batch_size=32,
            epochs=5,
            shuffle=True,
            verbose=1,
            validation_split=validation_split)


# In[ ]:


# Generate generalization metrics
score_pretrained = model_pretrained.evaluate(input_test, y_val, verbose=1)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

y_pred_pretrained_2 = model.predict(input_test)
i=1
tst = np.round(y_pred_pretrained_2,0)
for num in range(4):   
    plt.figure(figsize = (30,30))
    plt.subplot(420+i) # Shorthand for size 3x3, position i
    i+=1
    image = tst[num]*255
    plt.title('Predicted class')
    plt.imshow(image, cmap = 'gray')
    plt.subplot(420+i)
    image = y_val[num]*255
    plt.title('Ground truth')
    plt.imshow(image, cmap = 'gray')
    i+=1
plt.show()


# In[ ]:


model_pretrained = get_unet_pretrained_2(input_img, n_filters = 32, dropout = 0.01, batchnorm = True, n_classes = 1, lr = 0.0002)
history = model_pretrained.fit(input_train, y_trn,
            batch_size=64,
            epochs=5,
            shuffle=True,
            verbose=1,
            validation_split=validation_split)


# In[ ]:


tst_raw = h5py.File('/kaggle/input/ca2-data/tree_train.h5', 'r')
tst_raw = tst_raw.get('X')
mean = np.mean(tst_raw, axis=(1,2), keepdims=True)
std = np.std(tst_raw, axis=(1,2), keepdims=True)
X_tst = (tst_raw - mean) / std
input_test = X_tst.reshape((len(X_tst), img_width, img_height, img_num_channels))
#history = model_pretrained.fit(input_train, y_trn,
#            batch_size=64,
#            epochs=10,
#            shuffle=True,
#            verbose=1,
#            validation_split=validation_split)
y_pred_prd = model_pretrained.predict(input_test)
i=1
tst = np.round(y_pred_prd,0)


# In[ ]:


tst = tst.flatten()
targets = (tst > 0.5).astype("int64")


# In[ ]:


csv_data = {'Predicted':targets} 
df = pd.DataFrame(csv_data) 
df.index.name = 'Id'
print(df.head())
df.to_csv('submission4.csv')


# In[ ]:


tst = tst.flatten()
reslist = []
indlist = []
for pos, val in enumerate(tst):
    reslist.append(val)
    indlist.append(pos)
res = pd.DataFrame()
res['Id'] = indlist
res['Predicted'] = reslist
res.to_csv('resultat.csv', index = False, compression = None) 


# In[ ]:


a = res.to_csv('resultat.csv', index=False, compression = None) 


# In[ ]:


result = pd.read_csv('/kaggle/working/resultat.csv')


# In[ ]:




