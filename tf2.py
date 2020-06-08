# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:10:18 2020

@author: Gaurav
"""

import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix

#%% gradient tape

x = tf.Variable(1.0)

with tf.GradientTape(persistent=True) as tape2:
    with tf.GradientTape(persistent=True) as tape:
        y = 3*x + 2*tf.pow(x,3) + 4
    dy_dx = tape.gradient(y,x)
print(tape2.gradient(dy_dx, x)) #second order defrentiation
del tape, tape2


#%% simple linear model, process:
# 1) Create model
# 2) create loss function
# 3) create training loop structure
# 4) train
class Model():
    def __init__(self):
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)
    
    def __call__(self,x):
        x = tf.convert_to_tensor(x, dtype='float32')
        y = self.w * x + self.b
        return y

model = Model()

x = tf.constant(3.0)
print(model(x))

def loss_function(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred))

x = np.random.rand(1000, 1)
y = x * 15.6 + 2.3 + np.random.randn(1000, 1)

plt.scatter(x, y, s=0.1, c='k')
plt.scatter(x, model(x), s=0.1, c='b')

def train(model, x, y_true, loss_function, learning_rate):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = loss_function(y_true, y_pred)
    dw, db = tape.gradient(loss, [model.w, model.b])
    model.w.assign_sub(learning_rate*dw)
    model.b.assign_sub(learning_rate*db)

history = {'loss': [], 'epoch_num': [], 'w': [], 'b':[]}
for i in range(1000):
    history['epoch_num'].append(i)
    history['w'].append(model.w.numpy())
    history['b'].append(model.b.numpy())
    history['loss'].append(loss_function(y, model(x)).numpy())
    train(model, x, y, loss_function, 0.1)
    print(f"epoch = {i} loss = {history['loss'][-1]}")

plt.figure()
plt.plot(history['epoch_num'], history['w'])
plt.plot(history['epoch_num'], history['b'])
plt.twinx()
plt.plot(history['epoch_num'], history['loss'], c='k')

#%% hidden layers custom keras
# 1) Create model
# 2) create loss function
# 3)define grad
# 4) create training loop structure
# 5) train

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
tf.keras.backend.set_floatx('float64')
df = pd.read_csv('iris.csv', header=None)
x = df.iloc[:,:-1].values
oh = OneHotEncoder()
y = oh.fit_transform(df.iloc[:,-1].values.reshape(-1,1)).toarray()

model = Sequential([Dense(10, input_shape=(4,), activation='relu', dtype='float64', name='input_'),
                    Dense(5, activation='relu', name='dense1_'),
                    Dense(3, activation='softmax', name='softmax_')])

pred = pd.DataFrame(oh.inverse_transform(tf.one_hot(tf.argmax(model(x), axis=-1),
                                       depth=3, dtype='float64').numpy()))

def loss_func(model, y, x, training):
    y_ = model(x, training=training)
    loss = CategoricalCrossentropy()(y, y_)
    return loss

print(f"{loss_func(model, y, x, False)}")

def grad(model, y, x):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        loss = loss_func(model, y, x, True)
    gradient = tape.gradient(loss, model.trainable_variables)
    return gradient

optimiser = Adam()
print('iteration ', optimiser.iterations.numpy(), ' loss',loss_func(model, y, x, True).numpy())

for _ in range(1000):
    grads = grad(model, y, x)
    optimiser.apply_gradients(zip(grads, model.trainable_variables))
    print('iteration ', optimiser.iterations.numpy(), ' loss',loss_func(model, y, x, True).numpy())

y_ = pd.DataFrame(oh.inverse_transform(tf.one_hot(tf.argmax(model(x), axis=-1),
                                       depth=3, dtype='float64').numpy()), columns=['y_pred'])
y_['y_true'] = df.iloc[:,-1]

cm = confusion_matrix(y_['y_true'], y_['y_pred'], labels=oh.categories_[0])
