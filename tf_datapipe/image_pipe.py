import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
# Find all image files in the folder, keeping their order unchanged.
images_ds = tf.data.Dataset.list_files('C:/ml data/cat/*', shuffle=False)
for image_path in images_ds:
	print(image_path.numpy().decode())

# images_ds = images_ds.concatenate(
#     tf.data.Dataset.list_files('C:/ml data/dog/*', shuffle=False)
# )
# # Convert the file paths to a list so they can be split.
# image_paths = list(images_ds.as_numpy_iterator())

# # Split the paths into 80% for training and 20% for validation.
# train_ds, val_ds = train_test_split(
#     image_paths, test_size=0.2, random_state=42
# )

# # print(val_paths)

# # Names of the image categories.
# class_names = ['cat', 'dog']

# # lab = 'C:\\cat\\Abyssinian_69.jpg'

# def get_lable(file_path):
#     return tf.strings.split(file_path, os.path.sep)[-2]

# train_ds = tf.data.Dataset.from_tensor_slices(train_ds)

# # for label in train_ds.map(get_lable):
# #     print(label)

# # for t in train_ds.take(4):
# #     print(t.numpy())
    
# def process_image_path(path):
#     label = get_lable(path)
#     img = tf.io.read_file(path)
#     img = tf.image.decode_jpeg(img)
#     img = tf.image.resize(img,[128,128])
#     return img,label

# for img,label in train_ds.map(process_image_path):
#     print('Image',img)
#     print('label',label)