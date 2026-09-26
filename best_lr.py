import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from sklearn.metrics import confusion_matrix

# Make 1000 examples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

# Split into 80% training and 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

plt.figure(figsize=(7, 7))

# Set random seed
tf.random.set_seed(42)

# 1. Create the model (this time 3 layers)
# non linear ------------
model_3 = tf.keras.Sequential([
  tf.keras.layers.Dense(50,activation='relu'), # add 100 dense neurons
  tf.keras.layers.Dense(40,activation='relu'), # add another layer with 10 neurons
  tf.keras.layers.Dense(1,activation='sigmoid')
])
# non linear ------------
# 2. Compile the model
model_3.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(), # use Adam instead of SGD
                metrics=['accuracy'])
#  create a learning rate callback
lr_Scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))
# 3. Fit the model
history = model_3.fit(X_train, y_train, epochs=40, callbacks=[lr_Scheduler]) # fit for 100 passes of the data

y_pred = model_3.predict(X_test)
y_pred_labels = (y_pred >= 0.5).astype(int).ravel()

# confusion matrics
print('Confusion matricx',confusion_matrix(y_test,y_pred_labels) )


comparison = np.column_stack((y_test, y_pred_labels))
print("Actual vs Predicted:")
print(comparison)

accuracy = np.mean(y_test == y_pred_labels)
print("Test accuracy:", accuracy)

# Plot training loss and accuracy.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])

epochs = range(len(history.history['loss']))
lrs = 1e-4 * (10 ** (np.arange(len(history.history['loss'])) / 20))

ax1.plot(epochs, history.history['loss'], label='Loss', color='blue')
ax1.plot(epochs, history.history['accuracy'], label='Accuracy', color='orange')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss / Accuracy')
ax1.set_title('Training Metrics')
ax1.grid(True)
ax1.legend(loc='upper left')

# Show learning rate explicitly as a separate line.
ax2.plot(epochs, lrs, color='green', label='Learning rate')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Learning Rate')
ax2.set_title('Learning Rate over Epochs')
ax2.set_yscale('log')
ax2.grid(True)
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Optional: learning-rate vs loss plot
plt.figure(figsize=(10, 7))
plt.semilogx(lrs, history.history['loss'], color='purple')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning rate vs. loss')
plt.grid(True)
plt.show()

# Plot training loss and accuracy.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(epochs, history.history["loss"], label="Loss", color="blue")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.semilogx(lrs, history.history["loss"], color="green")
ax2.set_xlabel("Learning Rate")
ax2.set_ylabel("Loss")
ax2.set_title("Learning rate vs. loss")
ax2.grid(True)

plt.tight_layout()
plt.show()