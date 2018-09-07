
# coding: utf-8

# In[48]:


import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test)= mnist.load_data()

#normailize the data -> scale the value from [0 1]
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#2 types of models one is sequencial
model = tf.keras.models.Sequential()
#add layer
#flatten layer
model.add(tf.keras.layers.Flatten())
#Dense layer 128 unint(neuro) in the layer, + sigmoid function 
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
#output layer
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

#parameters for training model
model.compile(optimizer ='adam', loss = 'sparse_categorical_crossentropy' ,
             metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 3)

#calculate the validation loss
val_loss, val_accuracy = model.evaluate(x_test, y_test)
print(val_loss, val_accuracy) #after 3 iteration, loss =0.086 accuaracy = 0.97



# In[49]:


#save model
model.save('number_epic.model')
#read model
new_model = tf.keras.models.load_model('number_epic.model')


# In[50]:


#prediction
predictions = new_model.predict([x_test])
print(predictions)


# In[56]:


#looks not friedly --> lets use numpy
import numpy as np
print(np.argmax(predictions[1]))


# In[55]:


#To show the predict image
plt.imshow(x_test[1])
plt.show()


# In[51]:


import matplotlib.pyplot as plt

#print(x_train[0])

#if its image
plt.imshow(x_train[0])
plt.show()

#if its binary
plt.imshow(x_train[0], cmap = plt.cm.binary)


# In[52]:


print(x_train[0])

