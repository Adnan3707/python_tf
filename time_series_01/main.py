# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data=pd.read_csv('C:/Interview Process/python_tf/time_series_01/jena_climate_2009_2016.csv')

df= data[5::6]
# print(df)

df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
temp = df['T (degC)']
# print(df[25:])
# %%
def df_to_X_y(df, window_size=5):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [[a] for a in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size]
    y.append(label)
  return np.array(X), np.array(y)

WINDOW_SIZE = 5

# %%
X,y =df_to_X_y(temp,WINDOW_SIZE)

X.shape , y.shape


X_train,y_train = X[:60000],y[:60000]
X_val,y_val = X[60000:65000],y[60000:65000]
X_test,y_test = X[65000:],y[65000:]
print(X_train[0],y[0])

model = tf.keras.Sequential([
  tf.keras.Input(shape=(WINDOW_SIZE, 1)),
  tf.keras.layers.LSTM(50, return_sequences=True),
  tf.keras.layers.LSTM(40),
  tf.keras.layers.Dense(1)
])
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')])

history = model.fit(
  X_train,
  y_train,
  validation_data=(X_val, y_val),
  epochs=15
)
# %%
y_pred = model.predict(X_val).ravel()

train_results = pd.DataFrame(data={'train predictions':y_pred,'Actual':y_val})

print(train_results)

# Multivariate Time Series

# %%
model2 = tf.keras.Sequential([
  tf.keras.Input(shape=(WINDOW_SIZE, 1)),
  tf.keras.layers.Conv1D(64, kernel_size=2),
  tf.keras.layers.LSTM(8),
  tf.keras.layers.Dense(1)
])
model2.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')])

model2.summary()
# %%
history2 = model2.fit(
  X_train,
  y_train,
  validation_data=(X_val, y_val),
  epochs=15
)
# %%
model3 = tf.keras.Sequential([
  tf.keras.Input(shape=(WINDOW_SIZE, 1)),
  tf.keras.layers.GRU(64, return_sequences=True),
  tf.keras.layers.LSTM(8),
  tf.keras.layers.Dense(8,activation='relu'),
  tf.keras.layers.Dense(1,activation='linear')
])
model3.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')])
model3.summary()

history3 = model3.fit(
  X_train,
  y_train,
  validation_data=(X_val, y_val),
  epochs=15
)
# %%
temp_df=pd.DataFrame({'tempature':temp})
temp_df['Seconds']=temp_df.index.map(pd.Timestamp.timestamp)

day = 60*60*24
year = 365.2425*day

temp_df['Day sin'] = np.sin(temp_df['Seconds'] * (2* np.pi / day))
temp_df['Day cos'] = np.cos(temp_df['Seconds'] * (2 * np.pi / day))
temp_df['Year sin'] = np.sin(temp_df['Seconds'] * (2 * np.pi / year))
temp_df['Year cos'] = np.cos(temp_df['Seconds'] * (2 * np.pi / year))
temp_df.head()
# %%
temp_df = temp_df.drop('Seconds', axis=1)
temp_df.head()
# %%
def df_to_X_y2(df,window_size=6):
    df_as_np = df.to_numpy()
    X=[]
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [r for r in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size][0]
        y.append(label)
    return np.array(X),np.array(y)
X2,y2 = df_to_X_y(temp_df)
# %%
X2.shape,y2.shape
X2[0],y2[0]

# %%
X2_train, y2_train = X2[:60000], y2[:60000]
X2_val, y2_val = X2[60000:65000], y2[60000:65000]
X2_test, y2_test = X2[65000:], y2[65000:]
X2_train.shape, y2_train.shape, X2_val.shape, y2_val.shape, X2_test.shape, y2_test.shape