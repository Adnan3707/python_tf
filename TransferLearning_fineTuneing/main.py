from helper import create_tensorboard_callback, plot_loss_curves, unzip_data, walk_through_dir

# %%
# How many images in each folder?
import os

# %%
# Walk through 10 percent data directory and list number of files
for dirpath, dirnames, filenames in os.walk("C:\\Interview Process\\python_tf\\TransferLearning\\10_food_classes_10_percent"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
  
# %%
# Create training and test directories
train_dir = "C:\\Interview Process\\python_tf\\TransferLearning\\10_food_classes_10_percent\\train\\"
test_dir = "C:\\Interview Process\\python_tf\\TransferLearning\\10_food_classes_10_percent\\test\\"

# %%
import tensorflow as tf
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    batch_size=32,
    image_size=(256, 256),
    label_mode='categorical'
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    batch_size=32,
    image_size=(256, 256),
    label_mode='categorical'
)
# %%
train_dataset.class_names
# %%
test_dataset.take(1)
# %%
# Building a transfer learning Modle using 
base_model = tf.keras.applications.EfficientNetB0(include_top=False)

base_model.trainable = False

inputs = tf.keras.layers.Input(shape=(256, 256, 3), name="input_layer")

x= base_model(inputs, training=False)
print(f"Shape after base model: {x.shape}")

x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)

outputs = tf.keras.layers.Dense(len(train_dataset.class_names), activation="softmax", name="output_layer")(x)

# 8. Combine the inputs with the outputs into a model
model_0 = tf.keras.Model(inputs, outputs)

# 9. Compile the model
model_0.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# 10. Fit the model and save its history
history_10_percent = model_0.fit(train_dataset,
                                 epochs=5,
                                 steps_per_epoch=len(train_dataset),
                                 validation_data=test_dataset,
                                 validation_steps=int(0.25 * len(test_dataset)),
                                 )
# %%
# Getting Feature Vectors from a trained model
input_shape = (1,4,4,3)
# create random tensor
tf.random.set_seed(42)
random_tensor = tf.random.normal(input_shape)
random_tensor
# %%
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(random_tensor)
# %%
print(f"Shape after global average pooling: {global_average_layer}")
# %%
global_max_pooling_layer = tf.keras.layers.GlobalMaxPooling2D()(random_tensor)
print(f"Shape after global max pooling: {global_max_pooling_layer}")
# %%
# Create training and test dirs
train_dir_1_percent = "C:\\Interview Process\\python_tf\\TransferLearning_fineTuneing\\10_food_classes_1_percent\\train"
test_dir = "C:\\Interview Process\\python_tf\\TransferLearning_fineTuneing\\10_food_classes_1_percent\\test"
    
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
train_data_1_percent = tf.keras.preprocessing.image_dataset_from_directory(train_dir_1_percent,
                                                                           label_mode="categorical",
                                                                           image_size=IMG_SIZE,
                                                                           batch_size=BATCH_SIZE) # default = 32

test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                label_mode="categorical",
                                                                image_size=IMG_SIZE,
                                                                batch_size=BATCH_SIZE)
# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data_augmentation = keras.Sequential([
    keras.Input(shape=(224, 224, 3)),
    layers.RandomFlip("horizontal"), 
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomHeight(0.2),
    layers.RandomWidth(0.2),
    layers.Rescaling(1./255)
], name="data_augmentation")
# %%
# Visualize our data augmentation layer (and see what happens to our data)
# View a random image and compare it to its augmented version
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
target_class = random.choice(train_data_1_percent.class_names)
target_dir = os.path.join(train_dir_1_percent, target_class)
random_image = random.choice(os.listdir(target_dir))
random_image_path = os.path.join(target_dir, random_image)

# Read and plot in the random image 
img = mpimg.imread(random_image_path)
plt.imshow(img)
plt.title(f"Original random image from class: {target_class}")
plt.axis(False);

# Now let's plot our augmented random image
augmented_img = data_augmentation(tf.expand_dims(img, axis=0))
plt.figure()
plt.imshow(tf.squeeze(augmented_img))
plt.title(f"Augmented random image from class: {target_class}")
plt.axis(False);
# print(augmented_img)
# %%
# Model 1: Feature extraction transfer learning on 1% of the data with data augmentation

input_shape = (224, 224, 3)
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    input_shape=input_shape,
)
base_model.trainable = False
inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")

x= data_augmentation(inputs)

x= base_model(x, training=False)

x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)

outputs = layers.Dense(len(train_data_1_percent.class_names), activation="softmax", name="output_layer")(x)

model_1 = tf.keras.Model(inputs, outputs)

model_1.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

history_1_percent = model_1.fit(train_data_1_percent,
                                
                                    epochs=5,   
                                    steps_per_epoch=len(train_data_1_percent),
                                    validation_data=test_data,
                                    validation_steps=int(0.25 * len(test_data)),
                                    )
# %%
train_dir_10_percent = "C:\\Interview Process\\python_tf\\TransferLearning\\10_food_classes_10_percent\\train\\"
test_dir_10_percent = "C:\\Interview Process\\python_tf\\TransferLearning\\10_food_classes_10_percent\\test\\"
# %%
import tensorflow as tf
train_dataset_10_percent = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir_10_percent,
    batch_size=32,
    image_size=(256, 256),
    label_mode='categorical'
)

test_dataset_10_percent = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir_10_percent,
    batch_size=32,
    image_size=(256, 256),
    label_mode='categorical'
)

# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data_augmentation = keras.Sequential([
    keras.Input(shape=(256, 256, 3)),
    layers.RandomFlip("horizontal"), 
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomHeight(0.2),
    layers.RandomWidth(0.2),
    # layers.Rescaling(1./255)
], name="data_augmentation")

# %%
input_shape = (256, 256, 3)
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    input_shape=input_shape,
)
base_model.trainable = False
inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")

x= data_augmentation(inputs)

x= base_model(x, training=False)

x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)

outputs = layers.Dense(len(train_dataset_10_percent.class_names), activation="softmax", name="output_layer")(x)

model_2 = tf.keras.Model(inputs, outputs)

model_2.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])
# %%
model_2.summary()
# %%
checkpoint_path="TransferLearning_fineTuneing/checkpoints/cp.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,save_best_only=False,save_freq='epoch',verbose=1)
# %%
# Fit the model saving checkpoints every epoch
initial_epochs = 5
history_10_percent_data_aug = model_2.fit(train_dataset_10_percent,
                                          epochs=initial_epochs,
                                          validation_data=test_dataset_10_percent,
                                          validation_steps=int(0.25 * len(test_dataset_10_percent)),
                                          callbacks=[checkpoint_callback])
# %%
from helper import  plot_loss_curves
plot_loss_curves(history_10_percent_data_aug)
# %%
# Load the saved model weights
checkpoint_path = "TransferLearning_fineTuneing/checkpoints/cp.weights.h5"

# Load the checkpoint into the model architecture
model_2.load_weights(checkpoint_path)
model_3 = model_2

# %%
model_3.summary()

# %%
model_3.layers
# %%
for layer in model_3.layers:
    print(layer.name, layer.trainable)
# %%

# Check which layers are tuneable (trainable)
for layer_number, layer in enumerate(model_3.layers):
  print(layer_number, layer.name, layer.trainable)
# %%
for layer_number, layer in enumerate(base_model.layers):
    print(layer_number, layer.name, layer.trainable)

# %%
base_model.trainable = True

for layer in base_model.layers[:-10]:
    layer.trainable = False

model_3.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),metrics=["accuracy"])
# %%
# Check which layers are tuneable (trainable)
for layer_number, layer in enumerate(base_model.layers):
  print(layer_number, layer.name, layer.trainable)
# %%
print(len(model_2.trainable_variables))
# %%
# Fine tune for another 5 epochs
fine_tune_epochs = 10

# Refit the model (same as model_2 except with more trainable layers)
history_fine_10_percent_data_aug = model_2.fit(train_dataset_10_percent,
                                               epochs=fine_tune_epochs,
                                               validation_data=test_data,
                                               initial_epoch=history_10_percent_data_aug.epoch[-1], # start from previous last epoch
                                               validation_steps=int(0.25 * len(test_data)) )                                               
# %%
    #   Model 4 with 100 % of Data
