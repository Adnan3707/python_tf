import numpy as np
import tensorflow as tf
numpy_A = np.arange(1,25,dtype=np.int32)
A=tf.constant(numpy_A)
numpy_B = tf.constant(numpy_A,shape=(2,3,4))
print(numpy_B)

A = tf.cast(A, tf.float32)

variance = tf.math.reduce_variance(A)
std = tf.math.reduce_std(A)

print('variance',variance)
print('std',std)

b = tf.constant([[3, 1, 4, 1, 5, 9, 2, 6],
                 [3, 4, 5, 1, 2, 3, 4, 0]])

print(tf.argmax(b))           # 5  (flattens to 1-D first, then argmax)
print(tf.argmax(b, axis=1))  # 5  (same thing, explicit)   
