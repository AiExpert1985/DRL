import numpy as np
#
# arr = np.array([[[1, 2, 3],
#                  [1, 2, 3],
#                  [1, 2, 3],
#                  [1, 2, 3],
#                  [1, 2, 3],
#                  ],
#                 [[1, 2, 3],
#                  [1, 2, 3],
#                  [1, 2, 3],
#                  [1, 2, 3],
#                  [1, 2, 3],
#                  ],
#                 [[1, 2, 3],
#                  [1, 2, 3],
#                  [1, 2, 3],
#                  [1, 2, 3],
#                  [1, 2, 3],
#                  ]])
# result = np.where(arr < 3)
# print([(a, b, c) for a, b, c in zip(result[0], result[1], result[2])])
# print(arr[result])

# print(arr.shape)
# mean = np.mean(arr, axis=2)
# print(mean)
# print(mean.shape)
b = np.array([1, 2, 3, 3, 2, 7, 7])
# print(np.argmax(b))

print(np.maximum(b, 2))
