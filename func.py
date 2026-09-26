import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import tensorflow as tf 

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
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k", s=25)
plt.title("make_circles Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.axis("equal")
plt.show()

# Set random seed
tf.random.set_seed(42)

# 1. Create the model (this time 3 layers)
# Linear vs Non Linear
# model_3 = tf.keras.Sequential([
#   tf.keras.layers.Dense(100), # add 100 dense neurons
#   tf.keras.layers.Dense(10), # add another layer with 10 neurons
#   tf.keras.layers.Dense(1)
# ])
# non linear ------------
model_3 = tf.keras.Sequential([
  tf.keras.layers.Dense(100,activation='relu'), # add 100 dense neurons
  tf.keras.layers.Dense(10,activation='relu'), # add another layer with 10 neurons
  tf.keras.layers.Dense(1,activation='sigmoid')
])
# non linear ------------
# 2. Compile the model
model_3.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(), # use Adam instead of SGD
                metrics=['accuracy'])

# 3. Fit the model
model_3.fit(X_train, y_train, epochs=25, verbose=1) # fit for 100 passes of the data

y_pred = model_3.predict(X_test)
y_pred_labels = (y_pred >= 0.5).astype(int).ravel()

# Show the circular data and the model's prediction boundary as a line
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

Z = model_3.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 8))
plt.contourf(xx, yy, Z, alpha=0.4, cmap="coolwarm")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k", s=25)
plt.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)
plt.title("make_circles Data with Prediction Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.axis("equal")
plt.show()

comparison = np.column_stack((y_test, y_pred_labels))
print("y_test vs y_pred:")
print(comparison)
print("Test accuracy:", np.mean(y_test == y_pred_labels))

print(model_3.history.history)

