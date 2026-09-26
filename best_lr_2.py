import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

import tensorflow as tf


# ==========================================
# 1. CREATE DATA
# ==========================================

X, y = make_circles(
    n_samples=1000,
    noise=0.03,
    random_state=42
)


# ==========================================
# 2. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 3. CREATE MODEL
# ==========================================

tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(40, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])


# ==========================================
# 4. COMPILE MODEL
# ==========================================

model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)


# ==========================================
# 5. CREATE LEARNING RATE LIST
# ==========================================

lrs = []


# This function changes the learning rate
# and also saves it in the lrs list

def change_learning_rate(epoch, current_lr):

    new_lr = 1e-4 * 10 ** (epoch / 20)

    lrs.append(new_lr)

    return new_lr


lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    change_learning_rate
)


# ==========================================
# 6. TRAIN MODEL
# ==========================================

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    callbacks=[lr_scheduler]
)


# ==========================================
# 7. TEST MODEL
# ==========================================

y_pred = model.predict(X_test)

y_pred_labels = (y_pred >= 0.5).astype(int).ravel()

accuracy = np.mean(y_test == y_pred_labels)

print("Test Accuracy:", accuracy)


# ==========================================
# 8. PRINT LEARNING RATES
# ==========================================

print("\nLearning Rates:")

for epoch, lr in enumerate(lrs, start=1):
    print(f"Epoch {epoch}: {lr}")


# ==========================================
# 9. GET EPOCHS
# ==========================================

epochs = range(1, len(lrs) + 1)


# ==========================================
# 10. LOSS AND ACCURACY
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    history.history["loss"],
    marker="o",
    label="Loss"
)

plt.plot(
    epochs,
    history.history["accuracy"],
    marker="o",
    label="Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Loss and Accuracy")

plt.grid(True)
plt.legend()

plt.show()


# ==========================================
# 11. LEARNING RATE
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    lrs,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate vs Epoch")

# Learning rate grows exponentially,
# so logarithmic scale makes the curve easier to see

plt.grid(True)

plt.show()


# ==========================================
# 12. LEARNING RATE VS LOSS
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    lrs,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate vs Epoch")
plt.grid(True)

plt.show()