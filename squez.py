import tensorflow as tf

tf.random.set_seed(42)
rndm= G = tf.random.uniform(
    shape=[1, 1, 1, 1, 50]
)
print(rndm.shape)



print(tf.squeeze(rndm).shape)

print(tf.squeeze(rndm,axis=1).shape)