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
# %%
model_1.fit(train_data,epochs=5,steps_per_epoch=len(train_data),validation_data=valid_data,validation_steps=len(valid_data))
# %%
# Check out the layers in our model
model_1.summary()

#  pre process a data

#  creating a batch
# %%
# Create train and test data generators and rescale the data 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

train_data = train_datagen.flow_from_directory(directory='C:/Interview Process/python_tf/pizza_steak/train',target_size=(224,224),class_mode='binary',
                                               batch_size=32)
test_data = test_datagen.flow_from_directory(directory='C:/Interview Process/python_tf/pizza_steak/test',target_size=(224,224),class_mode='binary',
                                               batch_size=32)

# %%
images,labels = next(train_data)
len(images),len(labels)
# %%

images[8]
# %%
# Make the creating of our model a little easier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential
# %%
# Create the model (this can be our baseline, a 3 layer Convolutional Neural Network)
model_4 = Sequential([
  Conv2D(filters=10, 
         kernel_size=(3,3) ,
         strides=1,
         padding='valid',
         activation='relu', 
         input_shape=(224, 224, 3)), # input layer (specify input shape)
  Conv2D(10, 3, activation='relu'),
  Conv2D(10, 3, activation='relu'),
  Flatten(),
  Dense(1, activation='sigmoid') # output layer (specify output shape)
])
# Compile the model
model_4.compile(loss='binary_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

# %%
# Fit the model
history_4 = model_4.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data))

# Results:- accuracy: 0.9653 - loss: 0.1286 - val_accuracy: 0.8360 - val_loss: 0.3737
# %%
import pandas as pd
pd.DataFrame(history_4.history).plot(figsize=(10, 7));
# %%

# Create the model (this can be our baseline, a 3 layer Convolutional Neural Network)
model_5 = Sequential([
  Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
  MaxPool2D(pool_size=2), # reduce number of features by half
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Flatten(),
  Dense(1, activation='sigmoid')
])
# Compile model (same as model_4)
model_5.compile(loss='binary_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])
# Fit the model
history_5 = model_5.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data))
# accuracy: 0.8333 - loss: 0.3934 - val_accuracy: 0.7580 - val_loss: 0.4792
# %%
# Plot the validation and training data separately
def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();
  
# %%
plot_loss_curves(history_5)

# %%
# Create ImageDataGenerator training instance with data augmentation
train_datagen_augmented = ImageDataGenerator(rescale=1/255.,
                                             rotation_range=20, # rotate the image slightly between 0 and 20 degrees (note: this is an int not a float)
                                             shear_range=0.2, # shear the image
                                             zoom_range=0.2, # zoom into the image
                                             width_shift_range=0.2, # shift the image width ways
                                             height_shift_range=0.2, # shift the image height ways
                                             horizontal_flip=True) # flip the image on the horizontal axis

# Create ImageDataGenerator training instance without data augmentation
train_datagen = ImageDataGenerator(rescale=1/255.) 

# Create ImageDataGenerator test instance without data augmentation
test_datagen = ImageDataGenerator(rescale=1/255.)

# Import data and augment it from training directory
print("Augmented training images:")
train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                   target_size=(224, 224),
                                                                   batch_size=32,
                                                                   class_mode='binary',
                                                                   shuffle=False) # Don't shuffle for demonstration purposes, usually a good thing to shuffle

# Create non-augmented data batches
print("Non-augmented training images:")
train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(224, 224),
                                               batch_size=32,
                                               class_mode='binary',
                                               shuffle=False) # Don't shuffle for demonstration purposes

print("Unchanged test images:")
test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=(224, 224),
                                             batch_size=32,
                                             class_mode='binary')
# %%
# Display the same randomly selected training image with and without augmentation.
random_image_path = random.choice(train_data.filepaths)
with Image.open(random_image_path) as source_image:
  source_image = source_image.convert('RGB').resize((224, 224))

# Apply the transform before rescaling; standardize() applies rescale=1/255.
original_array = np.asarray(source_image, dtype=np.float32)
augmented_array = train_datagen_augmented.random_transform(original_array.copy())
augmented_array = train_datagen_augmented.standardize(augmented_array)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(original_array / 255.0)
axes[0].set_title('Non-augmented')
axes[0].axis('off')
axes[1].imshow(np.clip(augmented_array, 0, 1))
axes[1].set_title('Augmented')
axes[1].axis('off')
plt.tight_layout()
plt.show()

# %%
# Create the model (same as model_5)
model_6 = Sequential([
  Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
  MaxPool2D(pool_size=2), # reduce number of features by half
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Flatten(),
  Dense(1, activation='sigmoid')
])

# Compile the model
model_6.compile(loss='binary_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

# %%
# Fit the model
history_6 = model_6.fit(train_data_augmented, # changed to augmented training data
                        epochs=5,
                        steps_per_epoch=len(train_data_augmented),
                        validation_data=test_data,
                        validation_steps=len(test_data))
# accuracy: 0.6480 - loss: 0.6434 - val_accuracy: 0.8000 - val_loss: 0.5162
# %%
plot_loss_curves(history_6)
# %%
train_data_shuffled = train_datagen.flow_from_directory(train_dir,
                                               target_size=(224, 224),
                                               batch_size=32,
                                               class_mode='binary',
                                               shuffle=True)
# %%
model_7 = Sequential([
  Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
  MaxPool2D(pool_size=2), # reduce number of features by half
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Flatten(),
  Dense(1, activation='sigmoid')
])

# %%
# Compile the model
model_7.compile(loss='binary_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

history_7 = model_7.fit(train_data_shuffled, # changed to augmented training data
                        epochs=5,
                        steps_per_epoch=len(train_data_shuffled),
                        validation_data=test_data,
                        validation_steps=len(test_data))
# accuracy: 0.8387 - loss: 0.3740 - val_accuracy: 0.8680 - val_loss: 0.3309
# %%
class_names = ['pizza', 'steak']

# Load an image and prepare it for the model.
def get_image(path):
  image_path = Path(path)

  if image_path.is_file():
    with Image.open(image_path) as image:
      return image.convert('RGB')

  if image_path.is_dir():
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    image_paths = [
      p for p in image_path.iterdir()
      if p.is_file() and p.suffix.lower() in image_extensions
    ]
    if not image_paths:
      raise FileNotFoundError(f"No image files found in: {image_path}")

    with Image.open(random.choice(image_paths)) as image:
      return image.convert('RGB')

  raise FileNotFoundError(f"Image file or directory not found: {image_path}")


img_for_pred = get_image(r'C:/Interview Process/python_tf/pizza.jpg')

np.asarray(img_for_pred).shape

# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(img, img_shape=224):
  """
  Reads an image from filename, turns it into a tensor
  and reshapes it to (img_shape, img_shape, colour_channel).
  """


  # Decode the read file into a tensor & ensure 3 colour channels 
  # (our model is trained on images with 3 colour channels and sometimes images have 4 colour channels)
  if isinstance(img, Image.Image):
    img = np.asarray(img)

  if isinstance(img, np.ndarray):
    # get_image() already returns a decoded PIL image; accept its NumPy array too.
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    if img.shape.rank == 2:
      img = tf.expand_dims(img, axis=-1)
    if img.shape[-1] == 1:
      img = tf.image.grayscale_to_rgb(img)
    elif img.shape[-1] == 4:
      img = img[..., :3]
  else:
    img = tf.image.decode_image(img, channels=3)

  # Resize the image (to the same size our model was trained on)
  img = tf.image.resize(img, size = [img_shape, img_shape])

  # Rescale the image (get all values between 0 and 1)
  img = img/255.
  return img

# %%
img_for_pred = load_and_prep_image(img_for_pred)

# %%
img_for_pred.shape
# %%
pred = model_7.predict(tf.expand_dims(img_for_pred, axis=0))
# %%
pred_class = class_names[int(pred[0][0] >= 0.5)]
pred_class
# %%
