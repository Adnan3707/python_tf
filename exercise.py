import tensorflow as tf
random_1 = tf.random.Generator.from_seed(42)

mtrx = tf.constant(random_1.normal(shape=(224,224,3)))
# print('arg max',tf.argmax(mtrx, axis=1))

# Create a tensor with shape [10] using your own choice of values, then find the index which has the maximum value.
# One-hot encode the tensor you created in 9.
nine = tf.constant([3,4,2,1,3,4,5,6,7])
print('args max q9',tf.argmax(nine))
print(tf.one_hot(nine,depth=9))