---
layout:     post
title:      "Tensorflow review"
subtitle:   "Tensorflow"
date:       2018-01-25
author:     "hadxu"
header-img: "img/in-post/cs224n/cs224n_head.png"
tags:
    - Tensorflow
    - Python
    - cs224n
---

# 自己动手实现一个word2vec分析《倚天屠龙记》
这篇文章讲解了word2vec的tensorflow的原理，前几篇文章讲解了中文的分词，中文的分词主要有

* 基于DAG图的动态规划算法
* 基于HMM的维比特算法的分词

当我们将词分好，就可以使用tensorflow来训练一个word2vec来进行分析。具体代码下所示。先看效果：

![](/img/in-post/cs224n/word2vec.jpg)

可以看见，张无忌与武当派的关系非常相近，同时与周芷若也非常相近，但是该word2vec不是特别好，原因在于分词不是特别完美，而且很多没有意义的词也进行计算。


# cs224n之Tensorflow复习课

这一节课主要是介绍了Tensorflow工具，因为在下面的一系列作业中，都用Tensorflow来作为工具来完成作业。

**Tensorflow特点**

* 基于图的运算法则
* 计算之前先定义一张图
* 基于Session
* 基于张量(tensorflow)
* 分布式计算

下面的这些代码来自于cs20的习题课。

- Tensorflow简单计算

```python
import tensorflow as tf

x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x,y)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('graphs/normal_loading',sess.graph)
	for _ in range(20):
		res = sess.run(z)
		print(res)
		print(tf.get_default_graph())
		writer.close()
```

- placeholder使用

```
import tensorflow as tf

a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([5,5,5],tf.float32)

c = a+b

writer = tf.summary.FileWriter('graphs/placeholders',tf.get_default_graph())

with tf.Session() as sess:
	print(sess.run(c, {a:[1,2,3]}))
writer.close()

a = tf.add(2,5)
b = tf.multiply(a, 3)

with tf.Session() as sess:
	print(sess.run(b))
	print(sess.run(b,feed_dict={a:15}))
```

- variable 使用

```
W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()
sess1.run(W.initializer)
sess2.run(W.initializer)

print(sess1.run(W.assign_add(10)))

print(sess2.run(W.assign_add(10)))
print(sess1.run(W.assign_add(100)))
print(sess2.run(W.assign_add(13)))

sess1.close()
sess2.close()
```

> **需要注意的是每个session都有自己的工作空间**

##  简单线性回归

在Tensorflow中，读取数据可以使用

```
dataset = tf.data.Dataset.from_tensor_slices((data[:,0],data[:,1]))
```

然后使用迭代器

```
iterator = dataset.make_initializable_iterator()

X, Y = iterator.get_next()
```

定义线性回归参数

```
w = tf.get_variable('weights', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))
```

定义计算法则

```
Y_predicted = X * w + b
loss = tf.square(Y - Y_predicted, name='loss')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
```

开始训练

```
start = time.time()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)

	for i in range(100):
		sess.run(iterator.initializer)
		total_loss = 0

		try:
			while True:
				_,l = sess.run([optimizer,loss])
				total_loss += l
		except Exception as e:
			pass
		print('Epoch {0}: {1}'.format(i,total_loss/n_samples))
	writer.close()
	w_out, b_out = sess.run([w, b])
	print('w: %f, b: %f' %(w_out, b_out))
print('Took: %f seconds' %(time.time() - start))
```

展示结果

```
plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data with squared error')
# plt.plot(data[:,0], data[:,0] * (-5.883589) + 85.124306, 'g', label='Predicted data with Huber loss')
plt.legend()
plt.show()
```

### MNIST最简单的实例

```
import numpy as np
import tensorflow as tf
import time

import utils

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 30
n_train = 60000
n_test = 10000

# Step 1: Read in data
mnist_folder = 'data/mnist'

train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# Step 2: Create datasets and iterator
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000) # if you want to shuffle your data
train_data = train_data.batch(batch_size)

test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(batch_size)

iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                           train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)	# initializer for train_data
test_init = iterator.make_initializer(test_data)	# initializer for train_data

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
w = tf.get_variable(name='weights', shape=(784, 10), initializer=tf.random_normal_initializer(0, 0.01))
b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
logits = tf.matmul(img, w) + b 

# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label, name='entropy')
loss = tf.reduce_mean(entropy, name='loss') # computes the mean over all the examples in the batch

# Step 6: define training op
# using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())
with tf.Session() as sess:
   
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # train the model n_epochs times
    for i in range(n_epochs): 	
        sess.run(train_init)	# drawing samples from train_data
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # test the model
    sess.run(test_init)			# drawing samples from test_data
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Accuracy {0}'.format(total_correct_preds/n_test))
writer.close()
```

### Eager模式

为什么Google推出Eager模式呢？

* 搭建模型更简单了
* 调试不会报错在sess中了

先来看一个简单的例子

**求导**

假设有这样的函数
```
def g(x,y):
	return x**2+y**2
```

会一点高等数学的同学都知道怎么求解该梯度,假设使用eager

```
g = tfe.gradients_function(f)
```

或者使用Python的装饰器

```
@tfe.gradients_function
def g(x,y):
	return x**2+y**2
```

这两个效果是一样的。

#### huber loss

> 该损失函数是非常有用的，均方损失函数对异常点信息非常敏感，因为误差非常大，而huber_loss不一样，对异常点信息给予很小的权重。

```
def huber_loss(y, y_predicted, m=1.0):
	"""Huber loss."""
	t = y - y_predicted
	return t ** 2 if tf.abs(t) <= m else m * (2 * tf.abs(t) - m)
```

### word2vec

```
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf

import utils
import word2vec_utils

VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128            # dimension of the word embedding vectors
SKIP_WINDOW = 1             # the context window
NUM_SAMPLED = 64            # number of negative examples to sample
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 100000
VISUAL_FLD = 'visualization'
SKIP_STEP = 5000

DOWNLOAD_URL = 'http://mattmahoney.net/dc/text8.zip'
EXPECTED_BYTES = 31344016
NUM_VISUALIZE = 3000        # number of tokens to visualize



def word2vec(dataset):
    """ Build the graph for word2vec model and train it """
    # Step 1: get input, output from the dataset
    with tf.name_scope('data'):
        iterator = dataset.make_initializable_iterator()
        center_words, target_words = iterator.get_next()

    """ Step 2 + 3: define weights and embedding lookup.
    In word2vec, it's actually the weights that we care about 
    """
    with tf.name_scope('embed'):
        embed_matrix = tf.get_variable('embed_matrix', 
                                        shape=[VOCAB_SIZE, EMBED_SIZE],
                                        initializer=tf.random_uniform_initializer())
        embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embedding')

    # Step 4: construct variables for NCE loss and define loss function
    with tf.name_scope('loss'):
        nce_weight = tf.get_variable('nce_weight', shape=[VOCAB_SIZE, EMBED_SIZE],
                        initializer=tf.truncated_normal_initializer(stddev=1.0 / (EMBED_SIZE ** 0.5)))
        nce_bias = tf.get_variable('nce_bias', initializer=tf.zeros([VOCAB_SIZE]))

        # define loss function to be NCE loss function
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, 
                                            biases=nce_bias, 
                                            labels=target_words, 
                                            inputs=embed, 
                                            num_sampled=NUM_SAMPLED, 
                                            num_classes=VOCAB_SIZE), name='loss')

    # Step 5: define optimizer
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    
    utils.safe_mkdir('checkpoints')

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        sess.run(tf.global_variables_initializer())

        total_loss = 0.0 # we use this to calculate late average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter('graphs/word2vec_simple', sess.graph)

        for index in range(NUM_TRAIN_STEPS):
            try:
                loss_batch, _ = sess.run([loss, optimizer])
                total_loss += loss_batch
                if (index + 1) % SKIP_STEP == 0:
                    print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                    total_loss = 0.0
            except tf.errors.OutOfRangeError:
                sess.run(iterator.initializer)
        writer.close()




def gen():
    yield from word2vec_utils.batch_gen(DOWNLOAD_URL, EXPECTED_BYTES, VOCAB_SIZE, 
                                        BATCH_SIZE, SKIP_WINDOW, VISUAL_FLD)

def main():
	dataset = tf.data.Dataset.from_generator(gen, 
										(tf.int32, tf.int32), 
										(tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE, 1])))
	word2vec(dataset)
if __name__ == '__main__':
	main()
```

本文章写得仓促，所有代码来自[cs20](https://github.com/chiphuyen/stanford-tensorflow-tutorials/tree/master/examples)
目的是重温一下tensorflow的操作。有时间自己实现一个词向量。








