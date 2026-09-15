import numpy as np
import tensorflow as tf   
import matplotlib.pyplot as plt
from pathlib import Path

X = tf.range(-100,100,4)

y = X +10

#  split the data into taining and test set
X_train =X[:40]  # 80% of data

X_test = X[40:]

y_train = y[:40]

y_test = y[40:]

print(len(X_train),len(X_test))

# Visualize data

 
 
model_path = Path("model.keras")

if model_path.exists():
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded trained model from {model_path}")
else:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=[1], name='Dense_Layer'),
        tf.keras.layers.Dense(1, name='Output_Layer')
    ], name='model_1')

    model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=['mae'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=100)
    model.save(model_path)
    print(f"Saved trained model to {model_path}")

# from tensorflow.keras.utils import plot_model

# plot_model(model=model,show_shapes=True)

y_pred=model.predict(X_test)

def plot_pred(train_data=X_train,train_labels=y_train,
              test_data=X_test,
              test_labels=y_test,
              predictions=y_pred
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
plot_pred(train_data=X_train,
                 train_labels=y_train,
                 test_data=X_test,
                 test_labels=y_test,
                 predictions=y_pred)
plt.show()

# Model Evaluation

# Mean Absolute Error
print(y_pred.shape)
print('--------')
print(y_test.shape)

# y_pred=tf.squeeze(y_pred)

mae = tf.keras.losses.MeanAbsoluteError()
result = mae(
    y_true=y_test,
    y_pred=tf.squeeze(y_pred)
)
print(result)

# Mean Square Error
mse= tf.keras.losses.MeanSquaredError()
result = mse(y_true=y_test,y_pred=tf.squeeze(y_pred));
print('mse:',result)