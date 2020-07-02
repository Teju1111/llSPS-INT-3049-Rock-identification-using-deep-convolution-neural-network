#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
keras.__version__


# # importing model libraries

# In[2]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator


# # initialising the model 

# In[3]:


model=Sequential()


# # add conv2d layer

# In[4]:


model.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))


# # add maxpool layer

# In[5]:


model.add(MaxPooling2D(pool_size=(2,2)))


# # add flatten layer

# In[6]:


model.add(Flatten())


# # add hidden n output layers

# In[7]:


model.add(Dense(output_dim=500,activation='relu',init='random_uniform'))


# In[8]:


model.add(Dense(output_dim=300,activation='relu',init='random_uniform'))


# In[9]:


model.add(Dense(output_dim=150,activation='relu',init='random_uniform'))


# In[10]:


model.add(Dense(output_dim=64,activation='relu',init='random_uniform'))


# In[11]:


model.add(Dense(output_dim=3,activation='softmax',init='uniform'))


# In[12]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# # training model

# In[13]:


train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
x_train=train_datagen.flow_from_directory(r'C:\Users\Asus\Desktop\dataset\train_set',target_size=(64,64),batch_size=32,class_mode='categorical')
x_test=train_datagen.flow_from_directory(r'C:\Users\Asus\Desktop\dataset\test_set',target_size=(64,64),batch_size=32,class_mode='categorical')


# In[14]:


print(x_train.class_indices)


# In[16]:


model.fit_generator(x_train,steps_per_epoch=4,epochs=100, validation_data=x_test,validation_steps=2)


# In[17]:


model.save("cnn.h5")


# In[19]:


from keras.models import load_model
from keras.preprocessing import image


# In[20]:


model=load_model("cnn.h5")


# In[30]:


img=image.load_img(r'C:\Users\Asus\Desktop\dataset\train_set\Metamorphic rocks\quartzite.png',target_size=(64,64))


# In[31]:


import numpy as np


# In[32]:


x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)


# In[33]:


pred=model.predict_classes(x)
pred


# In[ ]:




