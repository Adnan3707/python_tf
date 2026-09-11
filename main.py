import tensorflow as tf
print(tf.__version__)

# creating tensor with tf.constant()
scaler = tf.constant(7)

# check number of dimensions of a tensor
print(scaler.ndim)

# create a vector
vector = tf.constant([10,10])

print(vector)

# create another matrix
another_matrix = tf.constant([[0. , 7.],
                              [9. , 2.],
                              [8.,9.]],dtype=tf.float16)

# print(another_matrix.ndim)

# lets create a tensor

# tensor =tf.constant([[[]]])

# creating random Tensors

random_1 = tf.random.Generator.from_seed(42)

random_1 = random_1.normal(shape=(3,2))
print('random tensor',random_1)