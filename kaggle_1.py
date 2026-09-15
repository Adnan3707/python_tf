import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')

# print(data.head())

# One Hot Encoding
encoded = pd.get_dummies(data)
# print(encoded.head())

# creating X and Y values

y = encoded.iloc[:,3]
print(y.head())
x = encoded.drop(encoded.columns[3], axis=1)
print(x.head())



x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

tf.random.set_seed(42)

insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10,input_shape=[11]),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(1)
])
insurance_model.compile(loss=tf.keras.losses.MeanAbsoluteError(),optimizer=tf.keras.optimizers.Adam(),
                        metrics=['mae'])
history_2 = insurance_model.fit(
    x_train,
    y_train,
    epochs=250,
    validation_data=(x_test, y_test)
)
print(y_test.median(),y_test.mean())
#  8487.8803          12968.317062869404
#   loss: 3191.7837 - mae: 3191.7837 

# Plot the model trained for 200 total epochs loss curves
pd.DataFrame(history_2.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()