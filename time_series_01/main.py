
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
	'C:/Interview Process/python_tf/time_series_01/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv',
	index_col='Date',
	parse_dates=['Date']
)

# print(df.head())
timesteps = df.index.to_numpy()
price =df['Closing Price (USD)'].to_numpy()

# print(timesteps[:5],price[:5])

# Plot from CSV
# import matplotlib.pyplot as plt
# import numpy as np
# plt.figure(figsize=(10, 7))
# plt.plot(date, price)
# plt.title("Price of Bitcoin from 1 Oct 2013 to 18 May 2021", fontsize=16)
# plt.xlabel("Date")
# plt.ylabel("BTC Price");
# plt.show()

#  wright way to split time series
split_size = int(.8 * (len(price)))
x_train,y_train = timesteps[:split_size],price[:split_size]
x_test,y_test = timesteps[split_size:],price[split_size:]

print(len(x_train))


def plot_time_series(timesteps, values, format='-', start=0, label=None):
	plt.plot(timesteps[start:], values[start:], format, label=label)


naive_forecast = y_test[:-1] # Naïve forecast equals every value excluding the last value
# print(naive_forecast)
naive_forecast[:10], naive_forecast[-10:] # View frist 10 and last 10 

# Plot naive forecast
offset = 300
plt.figure(figsize=(10, 7))
plot_time_series(
	timesteps=x_test,
	values=y_test,
	start=offset,
	label='Test data'
)
plot_time_series(
	timesteps=x_test[1:],
	values=naive_forecast,
	format='-',
	start=offset,
	label='Naive forecast'
)
plt.title('Bitcoin price: test data and naive forecast')
plt.xlabel('Date')
plt.ylabel('Closing price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()