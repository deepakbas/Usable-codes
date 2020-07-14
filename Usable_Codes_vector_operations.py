#Comparison of elementwise operation over naive approaches
#naive_relu
import numpy as np
def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x

x = np.array([[5, 78, -2, 34, 0],
[6, 79, 3, 35, 1],
[7, -80, 4, -36, 2]])
naive_relu(x)
#elementwise_relu
z = np.maximum(x, 0.)

#naive_add
def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x

x = np.array([[12, 3, 6, 14], [23, 2, 5, 7]])
y = np.array([[1, 31, 4, 4], [2, 20, 15, 3]])
naive_add(x, y)
#elementwise_add
z = x + y

#adding matrix and vector; naive vs broadcasting
def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x

x = np.array([[12, 3, 6, 14], [23, 2, 5, 7]])
y = np.array([1, 31, 4, 4])
naive_add_matrix_and_vector(x, y)

#elementwise addition through broadcasting
y=y[None, :]
y=np.repeat(y, repeats = [2], axis=0)
x+y

#
import numpy as np
x = np.random.random((64, 3, 32, 10))
y = np.random.random((32, 10))
z = np.maximum(x, y)

#naive vector dot product
def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z

x = np.array([12, 3, 6, 14])
y = np.array([1, 31, 4, 4])

#scalar output
naive_vector_dot(x, y)
x.dot(y)

#vector output
import numpy as np
def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z

x = np.array([[12, 3, 6, 14], [23, 2, 5, 7]])
y = np.array([1, 31, 4, 4])

z1 = naive_matrix_vector_dot(x, y)
y=y.reshape(4,1)
x.dot(y)

#elementwise addition through reshape
#matrix mul
def naive_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]
    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)
    return z

x = np.array([[12, 3, 6, 14], [23, 2, 5, 7]])
y = np.array([[1, 31, 4, 4], [2, 20, 15, 3]])

y=y.reshape(4,2)
naive_matrix_dot(x, y)