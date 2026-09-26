import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

print(tf.__version__)

# Load the Fashion MNIST dataset (compatible with modern TensorFlow versions)
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
data = {
    'train': (train_images, train_labels),
    'test': (test_images, test_labels)
}

print(train_images.shape)
print(train_labels[0])
# input shape = 28 * 28
#  output classes = 10 i.e  len(class_name)
# Show the first training example
import matplotlib.pyplot as plt
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Visualize the first 100 training examples and their class names.
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for image, label, axis in zip(train_images[:100], train_labels[:100], axes.flat):
    axis.imshow(image, cmap='gray')
    axis.set_title(class_names[label])
    axis.axis('off')
fig.tight_layout()
plt.show()

# Set random seed
tf.random.set_seed(42)

# 1. Create the model (this time 3 layers)
# non linear ------------
model_3 = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(5,activation='relu'), # add 100 dense neurons
  tf.keras.layers.Dense(4,activation='relu'), # add another layer with 10 neurons
  tf.keras.layers.Dense(10,activation='sigmoid')
])
# non linear ------------
# 2. Compile the model
model_3.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                optimizer=tf.keras.optimizers.Adam(), # use Adam instead of SGD
                metrics=['accuracy'])

# 3. Fit the model
history = model_3.fit(train_images, train_labels, epochs=40 , validation_data =(test_images,test_labels)) # fit for 100 passes of the data

     