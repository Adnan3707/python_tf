# %% Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import confusion_matrix

print(tf.__version__)

# %% Load data
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
# %% classNames
import matplotlib.pyplot as plt
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Visualize the first 100 training examples and their class names.

# fig, axes = plt.subplots(2, 5, figsize=(12, 5))
# for image, label, axis in zip(train_images[:100], train_labels[:100], axes.flat):
#     axis.imshow(image, cmap='gray')
#     axis.set_title(class_names[label])
#     axis.axis('off')
# fig.tight_layout()
# plt.show()

# %% Prepare normalized data
# Normalizing data

print(train_images.max() , train_images.min() )
train_images = train_images / 255.0
test_images = test_images / 255.0

print(train_images.max() , train_images.min() )

# accuracy: 0.8449 - loss: 0.4452 - val_accuracy: 0.8177 - val_loss: 0.5422


# %% Train normalized model
# Set random seed
tf.random.set_seed(42)

# 1. Create the model (this time 3 layers)
# non linear ------------
model_3 = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(5,activation='relu'), # add 100 dense neurons
  tf.keras.layers.Dense(4,activation='relu'), # add another layer with 10 neurons
  tf.keras.layers.Dense(10,activation='softmax')
])
# non linear ------------
# 2. Compile the model
model_3.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                optimizer=tf.keras.optimizers.Adam(), # use Adam instead of SGD
                metrics=['accuracy'])

# 3. Fit the model
history = model_3.fit(train_images, train_labels, epochs=40 , validation_data =(test_images,test_labels)) # fit for 100 passes of the data

print(history.history)


# %% Train non-normalized model
# Train the same model with the original 0-255 pixel values.
raw_train_images, raw_train_labels = data['train']
raw_test_images, raw_test_labels = data['test']
raw_model = tf.keras.models.clone_model(model_3)
raw_model.compile(
  loss=model_3.loss,
  optimizer=model_3.optimizer.__class__.from_config(model_3.optimizer.get_config()),
  metrics=['accuracy']
)
raw_history = raw_model.fit(
  raw_train_images,
  raw_train_labels,
  epochs=40,
  validation_data=(raw_test_images, raw_test_labels)
)

# %% Plot both training histories
# Plot the same metrics for normalized and non-normalized training.
epochs = range(1, len(history.history['loss']) + 1)
figure, axes = plt.subplots(1, 2, figsize=(12, 5))

for axis, title, run_history in zip(
  axes,
  ['Normalized data', 'Non-normalized data'],
  [history.history, raw_history.history]
):
  for metric in ['loss', 'accuracy', 'val_loss', 'val_accuracy']:
    axis.plot(epochs, run_history[metric], label=metric)
  axis.set_title(title)
  axis.set_xlabel('Epoch')
  axis.set_ylabel('Metric value')
  axis.legend()
  axis.grid(True, alpha=0.3)

figure.tight_layout()
figure.savefig('normalized_vs_non_normalized.png', dpi=150)
plt.show()

# %% Learning-rate experiment
#  Finding Ideal learning Rate 
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch/20))

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
lr_history = model_3.fit(train_images, train_labels, epochs=40 , validation_data =(test_images,test_labels),callbacks=[lr_scheduler]) # fit

# Plot training loss against the learning rate used for each epoch.
learning_rates = [
  1e-3 * 10**(epoch / 20)
  for epoch in range(len(lr_history.history['loss']))
]
plt.figure(figsize=(8, 5))
plt.semilogx(learning_rates, lr_history.history['loss'])
plt.title('Finding the ideal learning rate')
plt.xlabel('Learning rate')
plt.ylabel('Loss')
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('loss_vs_learning_rate.png', dpi=150)
plt.show()

# %% plot confusion matrics
import numpy as np
from sklearn.metrics import confusion_matrix

# %% cunvert probs in class names
y_predict = model_3.predict(test_images)
predictions = y_predict.argmax(axis=1)
predictions[:10]

# %%
def plot_confusion_matrix(true, predict, classes):
  true = np.asarray(true)
  predict = np.asarray(predict)

  if true.ndim > 1:
    true = np.argmax(true, axis=1)
  if predict.ndim > 1:
    predict = np.argmax(predict, axis=1)

  matrix = confusion_matrix(true, predict, labels=np.arange(len(classes)))
  figure, axis = plt.subplots(figsize=(8, 8))
  image = axis.imshow(matrix, cmap='Blues')
  figure.colorbar(image, ax=axis)
  axis.set(
    xticks=np.arange(len(classes)),
    yticks=np.arange(len(classes)),
    xticklabels=classes,
    yticklabels=classes,
    xlabel='Predicted label',
    ylabel='True label',
    title='Confusion matrix'
  )
  plt.setp(axis.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

  threshold = matrix.max() / 2 if matrix.size else 0
  for row in range(matrix.shape[0]):
    for column in range(matrix.shape[1]):
      axis.text(
        column,
        row,
        str(matrix[row, column]),
        ha='center',
        va='center',
        color='white' if matrix[row, column] > threshold else 'black'
      )

  figure.tight_layout()
  plt.show()
  return matrix

plot_confusion_matrix(test_labels,predictions,class_names)

#%%
def plot_random_image(model, images, true_labels, classes):
  import random

  i = random.randrange(len(images))
  target_image = images[i]
  pred_probs = model.predict(target_image[np.newaxis, ...], verbose=0)[0]
  pred_index = int(np.argmax(pred_probs))
  pred_label = classes[pred_index]
  true_label = classes[int(true_labels[i])]
  color = 'green' if pred_label == true_label else 'red'

  plt.imshow(target_image, cmap=plt.cm.binary)
  plt.title(
    f'Predicted: {pred_label} ({pred_probs[pred_index] * 100:.1f}%)\n'
    f'True: {true_label}',
    color=color
  )
  plt.axis('off')
  plt.tight_layout()
  plt.show()

# %%
plot_random_image(
  model=model_3,
  images=test_images,
  true_labels=test_labels,
  classes=class_names
)

# %%
