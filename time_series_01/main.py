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
WINDOW_SIZE2 = 6
X2,y2 = df_to_X_y2(temp_df, WINDOW_SIZE2)
# %%
X2.shape,y2.shape
X2[0],y2[0]

# %%
X2_train, y2_train = X2[:60000], y2[:60000]
X2_val, y2_val = X2[60000:65000], y2[60000:65000]
X2_test, y2_test = X2[65000:], y2[65000:]
X2_train.shape, y2_train.shape, X2_val.shape, y2_val.shape, X2_test.shape, y2_test.shape

# %%
temp_training_mean = np.mean(X2_train[:, :,0])
temp_training_std = np.std(X2_train[:,:,0])

def preprocess(X):
    X[:,:,0]=((X[:,:,0]-temp_training_mean)/temp_training_std)
    return X
preprocess(X2_train)
preprocess(X2_val)
preprocess(X2_test)

# %%
model4 = tf.keras.Sequential([
  tf.keras.Input(shape=(WINDOW_SIZE2, X2_train.shape[2])),
  tf.keras.layers.GRU(64, return_sequences=True),
  tf.keras.layers.LSTM(8),
  tf.keras.layers.Dense(8,activation='relu'),
  tf.keras.layers.Dense(1,activation='linear')
])
model4.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')])
model4.summary()
# %%
history4 = model4.fit(
  X2_train,
  y2_train,
  validation_data=(X2_val, y2_val),
  epochs=15
)
# %%
y2_pred = model4.predict(X2_test).ravel()
test_dates = temp_df.index[65000 + WINDOW_SIZE2:]

plt.figure(figsize=(14, 6))
plt.plot(test_dates, y2_test, label='Actual temperature')
plt.plot(test_dates, y2_pred, label='Model 4 prediction')
plt.title('Model 4: actual vs predicted temperature')
plt.xlabel('Date')
plt.ylabel('Temperature (degC)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# %%

# Pressure as predictive variable

p_temp_df =pd.concat([df['p (mbar)'],temp_df],axis=1)

p_temp_df.head()
# %%
def df_to_X_y3(df,window_size=7):
    df_as_np = df.to_numpy()
    X=[]
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [r for r in df_as_np[i:i+window_size]]
        X.append(row)
        label = [df_as_np[i+window_size][0],df_as_np[i+window_size][1]]
        y.append(label)
    return np.array(X),np.array(y)
X3,y3 = df_to_X_y3(p_temp_df)
X3.shape , y3.shape
# %%
X3_train, y3_train = X3[:60000], y3[:60000]
X3_val, y3_val = X3[60000:65000], y3[60000:65000]
X3_test, y3_test = X3[65000:], y3[65000:]
X3_train.shape, y3_train.shape, X3_val.shape, y3_val.shape, X3_test.shape, y3_test.shape
# %%

p_training_mean3 = np.mean(X3_train[:, :, 0])
p_training_std3 = np.std(X3_train[:, :, 0])

temp_training_mean3 = np.mean(X3_train[:, :, 1])
temp_training_std3 = np.std(X3_train[:, :, 1])

def preprocess3(X):
  X[:, :, 0] = (X[:, :, 0] - p_training_mean3) / p_training_std3
  X[:, :, 1] = (X[:, :, 1] - temp_training_mean3) / temp_training_std3
preprocess3(X3_train)
preprocess3(X3_val)
preprocess3(X3_test)
def preprocess_output3(y):
  y[:, 0] = (y[:, 0] - p_training_mean3) / p_training_std3
  y[:, 1] = (y[:, 1] - temp_training_mean3) / temp_training_std3
  return y
preprocess_output3(y3_train)
preprocess_output3(y3_val)
preprocess_output3(y3_test)
# %%
model5 = tf.keras.Sequential([
  tf.keras.Input(shape=(X3_train.shape[1], X3_train.shape[2])),
  tf.keras.layers.GRU(64, return_sequences=True),
  tf.keras.layers.LSTM(8),
  tf.keras.layers.Dense(8,activation='relu'),
  tf.keras.layers.Dense(y3_train.shape[1],activation='linear')
])
model5.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')])
model5.summary()
# %%
history5 = model5.fit(
  X3_train,
  y3_train,
  validation_data=(X3_val, y3_val),
  epochs=15
)
# %%
y3_pred = model5.predict(X3_test)
test_dates3 = p_temp_df.index[65000 + X3_train.shape[1]:]
y3_test_original = y3_test.copy()

y3_pred[:, 0] = y3_pred[:, 0] * p_training_std3 + p_training_mean3
y3_pred[:, 1] = y3_pred[:, 1] * temp_training_std3 + temp_training_mean3
y3_test_original[:, 0] = y3_test_original[:, 0] * p_training_std3 + p_training_mean3
y3_test_original[:, 1] = y3_test_original[:, 1] * temp_training_std3 + temp_training_mean3

fig, (ax_temp, ax_pressure) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

ax_temp.plot(test_dates3, y3_test_original[:, 1], label='Actual temperature')
ax_temp.plot(test_dates3, y3_pred[:, 1], label='Model 5 prediction')
ax_temp.set_title('Model 5: actual vs predicted temperature')
ax_temp.set_ylabel('Temperature (degC)')
ax_temp.legend()
ax_temp.grid(True, alpha=0.3)

ax_pressure.plot(test_dates3, y3_test_original[:, 0], label='Actual pressure')
ax_pressure.plot(test_dates3, y3_pred[:, 0], label='Model 5 prediction')
ax_pressure.set_title('Model 5: actual vs predicted pressure')
ax_pressure.set_xlabel('Date')
ax_pressure.set_ylabel('Pressure (mbar)')
ax_pressure.legend()
ax_pressure.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()


# %%
