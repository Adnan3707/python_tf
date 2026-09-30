# inspect the data
# %%
import os
import random
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# %%
#  list number of files
for dirpath, dirnames, filenames in os.walk("pizza_steak"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def get_random_image(directory, class_name):
  class_directory = Path(directory) / class_name
  if not class_directory.is_dir():
    raise FileNotFoundError(f"Class directory not found: {class_directory}")

  image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
  image_paths = [
    path for path in class_directory.iterdir()
    if path.is_file() and path.suffix.lower() in image_extensions
  ]
  if not image_paths:
    raise FileNotFoundError(f"No image files found in: {class_directory}")

  with Image.open(random.choice(image_paths)) as image:
    image = image.copy()

  plt.imshow(image)
  plt.title(class_name)
  plt.axis('off')
  plt.show()
  return image
# %%
img = get_random_image('C:/Interview Process/python_tf/pizza_steak/test', 'steak')
image = np.array(img, dtype=np.float32) / 255.0
# %%
import tensorflow as tf

tf.constant(image)

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.random.set_seed(42)
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale = 1./255)

#  Import data from directories and turn into batches
train_dir = 'C:/Interview Process/python_tf/pizza_steak/train'
test_dir = 'C:/Interview Process/python_tf/pizza_steak/test'

train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               batch_size=32,
                                               target_size=(224,224),
                                               class_mode='binary',
                                               seed=42)
valid_data = valid_datagen.flow_from_directory(directory=test_dir,
                                               batch_size=32,
                                               target_size=(224,224),
                                               class_mode='binary',
                                               seed=42)

model_1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=10,
                          kernel_size=3,
                          activation='relu',
                          input_shape=(224,224,3)),
    tf.keras.layers.Conv2D(10,3,activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2,
                              padding='valid'),
    tf.keras.layers.Conv2D(10,3,activation='relu'),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

#  compile our CNN
model_1.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])
# 04ms/step - accuracy: 1.0000 - loss: 1.2797e-08 - val_accuracy: 0.5000 - val_loss: 20.4517

model_1.fit(train_data,epochs=5,steps_per_epoch=len(train_data),validation_data=valid_data,validation_steps=len(valid_data))
# %%
