# %%
train_dir = "10_food_classes_all_data/train/"
test_dir = "10_food_classes_all_data/test/"

import os

class_names = sorted(
	entry for entry in os.listdir(train_dir)
	if os.path.isdir(os.path.join(train_dir, entry))
)


# %%
import random
from pathlib import Path

import matplotlib.pyplot as plt

image_paths = [
	image_path
	for class_name in class_names
	for image_path in (Path(train_dir) / class_name).iterdir()
	if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
]

random_image_path = random.choice(image_paths)
random_image = plt.imread(random_image_path)
print(f"Image shape: {random_image.shape}")
plt.imshow(random_image)
plt.axis("off");


        # Preprocess Data
# %%

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Rescale the data and create data generator instances
train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

# Load data in from directories and turn it into batches
train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(224, 224),
                                               batch_size=32,
                                               class_mode='categorical',
                                               shuffle=True)

test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=(224, 224),
                                             batch_size=32,
                                             class_mode='categorical',
                                             shuffle=False)

# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

# Create our model (a clone of model_8, except to be multi-class)
model_9 = Sequential([
  Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Conv2D(10, 3, activation='relu'),
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Flatten(),
  Dense(10, activation='softmax') # changed to have 10 neurons (same as number of classes) and 'softmax' activation
])

# Compile the model
model_9.compile(loss="categorical_crossentropy", # changed to categorical_crossentropy
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# %%
history_9 = model_9.fit(train_data, # now 10 different classes 
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data))

# %%
def plot_history(history):
  """Plot training and validation accuracy and loss from a Keras History."""
  history_data = history.history
  epochs = range(1, len(history_data["loss"]) + 1)
  fig, axes = plt.subplots(1, 2, figsize=(12, 5))

  for axis, metric, title in zip(axes, ("accuracy", "loss"), ("Accuracy", "Loss")):
    axis.plot(epochs, history_data[metric], label=f"Training {title.lower()}")
    validation_metric = f"val_{metric}"
    if validation_metric in history_data:
      axis.plot(epochs, history_data[validation_metric], label=f"Validation {title.lower()}")
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(title)
    axis.legend()

  fig.tight_layout()
  plt.show()

# %%
plot_history(history_9)
# %%
# Create augmented data generator instance
train_datagen_augmented = ImageDataGenerator(rescale=1/255.,
                                             rotation_range=20, # note: this is an int not a float
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             zoom_range=0.2,
                                             horizontal_flip=True)

train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                  target_size=(224, 224),
                                                                  batch_size=32,
                                                                  class_mode='categorical')
# %%
# Clone the model (use the same architecture)
model_11 = tf.keras.models.clone_model(model_9)

# Compile the cloned model (same setup as used for model_10)
model_11.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

# %%
# Fit the model
history_11 = model_11.fit(train_data_augmented, # use augmented data
                          epochs=5,
                          steps_per_epoch=len(train_data_augmented),
                          validation_data=test_data,
                          validation_steps=len(test_data))
# %%
plot_history(history_11)

# %%
class_names = {index: name for name, index in train_data.class_indices.items()}
images_dir = Path("images")
image_paths = [
    image_path for image_path in images_dir.iterdir()
    if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
] if images_dir.exists() else []

if not image_paths:
    print(f"No image files found in '{images_dir}'. Create an 'images' folder with test images first.")
else:
    for image_path in image_paths:
        image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        image = tf.keras.utils.img_to_array(image) / 255.0
        prediction = model_11.predict(tf.expand_dims(image, axis=0), verbose=0)[0]
        pred_index = int(tf.argmax(prediction).numpy())
        pred_class = class_names[pred_index]
        pred_confidence = float(prediction[pred_index])
        print(f"{image_path.name}: {pred_class} ({pred_confidence:.2%})")



# %%
