import tensorflow as tf

labels=[0,1,2,3,4]
indices = [0, 1, 2]
depth = 3
# print(tf.one_hot(indices, depth))
print(tf.one_hot(labels,depth=4))