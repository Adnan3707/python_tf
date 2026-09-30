import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# import seaborn as sns
import xgboost as xgb

from sklearn.metrics import mean_squared_error

df = pd.read_csv('time_series_02/PJME_hourly.csv')
# print(df.head()) 
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.set_index('Datetime').sort_index()

df['PJME_MW'].plot(figsize=(12, 5), title='PJME Hourly Energy Consumption')
plt.xlabel('Datetime')
plt.ylabel('Energy (MW)')
plt.tight_layout()
plt.show()

train = df.loc[df.index<'01-01-2015']
test = df.loc[df.index >= '01-01-2015']

plt.figure(figsize=(12, 5))
plt.plot(train.index, train['PJME_MW'], label='Train')
plt.plot(test.index, test['PJME_MW'], label='Test')
plt.title('PJME Hourly Energy Consumption: Train and Test')
plt.xlabel('Datetime')
plt.ylabel('Energy (MW)')
plt.legend()
plt.tight_layout()
plt.show()

week = df.loc[(df.index > '2010-01-01') & (df.index < '2010-01-08')]
week.plot(figsize=(15, 5), title='Week Of Data')
plt.show()

# Feature creating

def create_feature(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

features = create_feature(train)
print(features.head())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, col in zip(axes, ['month', 'dayofweek', 'weekofyear']):
    grouped = features.groupby(col)['PJME_MW'].mean()
    grouped.plot(kind='bar', ax=ax, title=f'Average PJME by {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Energy (MW)')

plt.tight_layout()
plt.show()

train = create_feature(train)
test = create_feature(test)

FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
TARGET = 'PJME_MW'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)