import tensorflow as tf

# A 4D tensor with shape (2, 2, 2, 2).
tensor_4d = tf.constant(
	[
		[
			[[1, 2], [3, 4]],
			[[5, 6], [7, 8]],
		],
		[
			[[9, 10], [11, 12]],
			[[13, 14], [15, 16]],
		],
	]
)

print(tensor_4d)
print("Shape:", tensor_4d.shape)

# A 4D tensor with shape (2, 2, 2, 3).
tensor_4d_3 = tf.constant(
    [
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ],
        [
            [[13, 14, 15], [16, 17, 18]],
            [[19, 20, 21], [22, 23, 24]],
        ],
    ]
)

print(tensor_4d_3)
print("Shape:", tensor_4d_3.shape)
