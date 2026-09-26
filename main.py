from sklearn.datasets import make_circles
import tensorflow as tf
import numpy as np
#  make 1000 examples
n_sample = 1000

X,y = make_circles(n_sample,noise=0.03,random_state=42)

print(X)

import pandas as pd

circles=pd.DataFrame({"x0":X[:,0],"X1":X[:,1],"label":y})

print(circles)

#  visualvise with plot
import matplotlib.pyplot as plt

plt.scatter(circles["x0"], circles["X1"], c=circles["label"])
plt.xlabel("x0")
plt.ylabel("X1")
plt.show()

print(X)

tf.random.set_seed(42)

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(1, input_shape=(2,))
# ])

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(1, input_shape=(2,)),
#     tf.keras.layers.Dense(1)
# ])
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu", input_shape=(2,)),
    tf.keras.layers.Dense(10),    
    tf.keras.layers.Dense(1, activation="sigmoid")
])
# accuracy: 1.0000 - loss: 0.1369 

model.compile(loss=tf.keras.losses.BinaryCrossentropy,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['accuracy'])
# try 1st
# model.fit(X,y,epochs=5)
# try 2nd
model.fit(X,y,epochs=200)

def plot_decision_boundry(model,X,y):
    x_min,x_max = X[:,0].min() -0.1 , X[:,0].max() + 0.1
    y_min,y_max = y[:,0].min() - 0.1 , y[:,0].max() + 0.1
    xx,yy = np.meshgrid(np.linspace(x_min,x_max),np.linspace(y_min,y_max))
    
    # create X value
    x_in = np.c()