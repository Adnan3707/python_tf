import tensorflow as tf

daily_sales = [50,90,-122,20,99]

tf_dataset = tf.data.Dataset.from_tensor_slices(daily_sales)

tf_dataset = tf_dataset.filter(lambda x:x>0)
tf_dataset = tf_dataset.map(lambda x : x * 95)

for sales in tf_dataset.as_numpy_iterator():
    print(sales)