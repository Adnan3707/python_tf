# %%
import tensorflow as tf
import os
train_dir = "10_food_classes_all_data/train/"
test_dir = "10_food_classes_all_data/test/"

# How many images are we working with now?
for dirpath, dirnames, filenames in os.walk("C:\\Interview Process\\python_tf\\TransferLearning_fineTuneing\\all data\\10_food_classes_all_data"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
  
# %%
IMG_SIZE = (256, 256)
train_data_10_classes_full = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                                 label_mode="categorical",
                                                                                 image_size=IMG_SIZE)

# Note: this is the same test dataset we've been using for the previous modelling experiments
test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                label_mode="categorical",
                                                                image_size=IMG_SIZE)
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

outputs = layers.Dense(len(train_data_10_classes_full.class_names), activation="softmax", name="output_layer")(x)

model_2 = tf.keras.Model(inputs, outputs)


# %%

model_2.summary()
# %%

print(len(model_2.trainable_variables))
# %%

# Check which layers are tuneable in the whole model
for layer_number, layer in enumerate(model_2.layers):
  print(layer_number, layer.name, layer.trainable)
# %%
# Unfreeze the top 10 layers in model_2's base_model
model_2_base_model = model_2.layers[2]
model_2_base_model.trainable = True

# Freeze all layers except for the last 10
for layer in model_2_base_model.layers[:-10]:
  layer.trainable = False
  
# %%

# Recompile the model (always recompile after any adjustments to a model)
model_2.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # lr is 10x lower than before for fine-tuning
                metrics=["accuracy"])
     
# %%
# Continue to train and fine-tune the model to our data
fine_tune_epochs = 10

history_fine_10_classes_full = model_2.fit(train_data_10_classes_full,
                                           epochs=fine_tune_epochs,
                                           validation_data=test_data,
                                           validation_steps=int(0.25 * len(test_data)))

# %%
import matplotlib.pyplot as plt

history = history_fine_10_classes_full.history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history["accuracy"], label="Train")
plt.plot(history["val_accuracy"], label="Validation/Test")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["loss"], label="Train")
plt.plot(history["val_loss"], label="Validation/Test")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.show()

# %%
checkpoint_path = 'C:\\Interview Process\\python_tf\\TransferLearning_fineTuneing\\TransferLearning_fineTuneing\\checkpoints\\cp.weights.h5'
fine_tuned_checkpoint_path = os.path.join(
    os.path.dirname(checkpoint_path), "fine_tuned.weights.h5"
)
model_2.save_weights(fine_tuned_checkpoint_path)
# %%
images, labels = next(iter(test_data))
predictions = model_2.predict(images, verbose=0)

plt.figure(figsize=(12, 8))
for i in range(min(9, len(images))):
    actual = train_data_10_classes_full.class_names[tf.argmax(labels[i]).numpy()]
    predicted = train_data_10_classes_full.class_names[tf.argmax(predictions[i]).numpy()]
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(f"Actual: {actual}\nPredicted: {predicted}",
              color="green" if actual == predicted else "red")
    plt.axis("off")
plt.tight_layout()
plt.show()

# %%
