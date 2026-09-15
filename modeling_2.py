# 2 Dense Layer, 100 epoch
import tensorflow as tf
import matplotlib.pyplot as plt

tf.random.set_seed(42)
X = tf.range(-100,100,4)
y = X +10
#  split the data into taining and test set
X_train =X[:40]  # 80% of data
X_test = X[40:]
y_train = y[:40]
y_test = y[40:]

model_2= tf.keras.Sequential([
    tf.keras.layers.Dense(10,input_shape=[1]),
    tf.keras.layers.Dense(1)
])

model_2.compile(loss=tf.keras.losses.MeanAbsoluteError(),
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])
model_2.fit(X_train, y_train, epochs=100)
y_pred_2 = model_2.predict(X_test).reshape(-1)

def plot_predictions(train_data=X_train,train_labels=y_train,
              test_data=X_test,
              test_labels=y_test,
              predictions=None
              ):
    plt.figure(figsize=(10, 7))
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", label="Testing data")
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    # Show the legend
    plt.legend();
    
plot_predictions(X_train, y_train, X_test, y_test, y_pred_2)
plt.show()