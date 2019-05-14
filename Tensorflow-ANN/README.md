# tensorflow-ann-mnist
TensorFlow implementation of an Artificial Neural Network on MNIST data.

To set up the libraries and plotting in a Jupyter notebook:

```python
import numpy as np

import tensorflow as tf


%matplotlib inline
import matplotlib.pyplot as plt
```

To read the data and visualise an image

```python
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

data, labels = mnist.train.next_batch(1)
```

```python
print len(data)
```

```python
plt.imshow(data.reshape(28,28))
```

To start a simple, direct network as presented on the [TensorFlow MNIST tutorial](https://www.tensorflow.org/versions/r0.10/tutorials/mnist/beginners/index.html).

```python
# Simple network.

tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.nn.bias_add(tf.matmul(x, W), b))
```

Alternatively to create a hidden layer in the network:

```python
tf.reset_default_graph()

num_neurons = 40

x = tf.placeholder(tf.float32, [None, 784])

W1 = tf.Variable(tf.random_normal([784, num_neurons]))
b1 = tf.Variable(tf.zeros([num_neurons]))

h1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x, W1), b1))

W2 = tf.Variable(tf.random_normal([num_neurons, 10]))
b2 = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.nn.bias_add(tf.matmul(h1, W2), b2))
```

To train the model pass the network setup over to the model class

```python
from model import Model

model = Model()
model.build(x, y)
s = model.train(mnist, 10)
```

Using the returned graph session you can now visualise one of the neurons weights. This basically shows where that neuron is paying attention.

```python
W1_ = s.run(W1)
plt.imshow(W1_[:,0].reshape(28,28), cmap=plt.get_cmap('gray'))
```