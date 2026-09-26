import tensorflow as tf
from tensorflow.keras.datasets import mnist

(X_train,y_train),(X_test,y_test) = mnist.load_data()

import matplotlib.pyplot as plt

example_dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
print(len(example_dataset))
# print(next(iter(example_dataset)))

def normalize_img(image,label):
    return (tf.cast(image,tf.float32)/255.0 , label)

example_dataset = example_dataset.map(normalize_img,num_parallel_calls=tf.data.AUTOTUNE)

# for x,y in example_dataset:
#     print(x)
#     print(y)
#     break

example_dataset = example_dataset.shuffle(len(example_dataset))

example_dataset = example_dataset.batch(64)

for (img,label) in example_dataset:
    print(img.numpy().shape)
    break