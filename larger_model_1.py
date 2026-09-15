import numpy as np
import tensorflow as tf   

# Create features
X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# Create labels
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

X = tf.constant(X, shape=(8, 1))
Y=tf.constant(y)

# steps in modelling with tensor

tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(1)
]
)


model.compile(loss=tf.keras.losses.MeanAbsoluteError,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])

model.fit(X,Y,epochs=100)

print(model.predict(np.array([[17.0]])))