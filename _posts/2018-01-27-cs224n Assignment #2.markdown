---
layout:     post
title:      "CS224N Assignment #2"
subtitle:   "cs224n"
date:       2018-01-27
author:     "hadxu"
header-img: "img/in-post/cs224n/cs224n_head.png"
tags:
    - Tensorflow
    - Python
    - cs224n
---

# CS224N第二次作业

> 这一次的作业是采用```Tensorflow```来完成各种作业

### Q1: Tensorflow Softmax (25 points, coding)

### Q2: Neural Transition-Based Dependency Parsing (50 points, mostly coding with a bit of theory)

### Q3: Recurrent Neural Networks: Language Modeling (25 points, theory)

##首先来看第一次作业(难度：简单)

> 要求我们用tf来实现```softmax```以及```cross-entropy```

```python
def softmax(x):
    """
    Compute the softmax function in tensorflow.

    You might find the tensorflow functions tf.exp, tf.reduce_max,
    tf.reduce_sum, tf.expand_dims useful. (Many solutions are possible, so you may
    not need to use all of these functions). Recall also that many common
    tensorflow operations are sugared (e.g. x + y does elementwise addition
    if x and y are both tensors). Make sure to implement the numerical stability
    fixes as in the previous homework!

    Args:
        x:   tf.Tensor with shape (n_samples, n_features). Note feature vectors are
                  represented by row-vectors. (For simplicity, no need to handle 1-d
                  input as in the previous homework)
    Returns:
        out: tf.Tensor with shape (n_sample, n_features). You need to construct this
                  tensor in this problem.
    """

    ### YOUR CODE HERE

    out = tf.exp(x)/tf.reduce_sum(tf.exp(x),axis=1,keep_dims=True)


    ### END YOUR CODE

    return out
```

注意要将```int```转换为```float类型```,使用```tf.to_float```.

```python
def cross_entropy_loss(y, yhat):
    """
    Compute the cross entropy loss in tensorflow.
    The loss should be summed over the current minibatch.

    y is a one-hot tensor of shape (n_samples, n_classes) and yhat is a tensor
    of shape (n_samples, n_classes). y should be of dtype tf.int32, and yhat should
    be of dtype tf.float32.

    The functions tf.to_float, tf.reduce_sum, and tf.log might prove useful. (Many
    solutions are possible, so you may not need to use all of these functions).

    Note: You are NOT allowed to use the tensorflow built-in cross-entropy
                functions.

    Args:
        y:    tf.Tensor with shape (n_samples, n_classes). One-hot encoded.
        yhat: tf.Tensorwith shape (n_sample, n_classes). Each row encodes a
                    probability distribution and should sum to 1.
    Returns:
        out:  tf.Tensor with shape (1,) (Scalar output). You need to construct this
                    tensor in the problem.
    """

    ### YOUR CODE HERE
    out = -tf.reduce_sum(tf.to_float(y)*tf.log(yhat))
    ### END YOUR CODE

    return out
```

> tf实现简单神经网络

```python
class SoftmaxModel(Model):
    """Implements a Softmax classifier with cross-entropy loss."""

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors.

        These placeholders are used as inputs by the rest of the model building
        and will be fed data during training.

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of shape
                                              (batch_size, n_features), type tf.float32
        labels_placeholder: Labels placeholder tensor of shape
                                              (batch_size, n_classes), type tf.int32

        Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
        """
        ### YOUR CODE HERE
        self.input_placeholder = tf.placeholder(tf.float32,(self.config.batch_size,self.config.n_features))
        self.labels_placeholder = tf.placeholder(tf.int32,(self.config.batch_size,self.config.n_classes))
        ### END YOUR CODE

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """Creates the feed_dict for training the given step.

        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        If label_batch is None, then no labels are added to feed_dict.

        Hint: The keys for the feed_dict should be the placeholder
                tensors created in add_placeholders.

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### YOUR CODE HERE
        feed_dict = {self.input_placeholder:inputs_batch,self.labels_placeholder:labels_batch}
        ### END YOUR CODE
        return feed_dict

    def add_prediction_op(self):
        """Adds the core transformation for this model which transforms a batch of input
        data into a batch of predictions. In this case, the transformation is a linear layer plus a
        softmax transformation:

        yhat = softmax(xW + b)

        Hint: Each ROW of self.inputs is a single example. This is generally best-practice for
              tensorflow code.
        Hint: Make sure to create tf.Variables as needed.
        Hint: For this simple use-case, it's sufficient to initialize both weights W
                    and biases b with zeros.

        Args:
            input_data: A tensor of shape (batch_size, n_features).
        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        ### YOUR CODE HERE
        with tf.variable_scope('transformation'):
            bias = tf.Variable(tf.zeros([self.config.n_classes]))
            W = tf.Variable(tf.zeros([self.config.n_features,self.config.n_classes]))
            z = tf.matmul(self.input_placeholder,W)+bias
        pred = softmax(z)
        ### END YOUR CODE
        return pred

    def add_loss_op(self, pred):
        """Adds cross_entropy_loss ops to the computational graph.

        Hint: Use the cross_entropy_loss function we defined. This should be a very
                    short function.
        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar)
        """
        ### YOUR CODE HERE
        loss = cross_entropy_loss(self.labels_placeholder, pred)
        ### END YOUR CODE
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/api_docs/python/tf/train/Optimizer

        for more information. Use the learning rate from self.config.

        Hint: Use tf.train.GradientDescentOptimizer to get an optimizer object.
                    Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        ### YOUR CODE HERE
        train_op = tf.train.GradientDescentOptimizer(self.config.lr).minimize(loss)
        ### END YOUR CODE
        return train_op
```

## 任务2 依存句法分析

首先手动完成```transition-based```解析器，开始不知道依赖怎么来的，问了知乎一个大神，知道了依赖是读取进来的，怪我想多了。。。

接下来完成tf版本的句法分析

```python
def parse_step(self, transition):
    """Performs a single parse step by applying the given transition to this partial parse

    Args:
        transition: A string that equals "S", "LA", or "RA" representing the shift, left-arc,
                    and right-arc transitions. You can assume the provided transition is a legal
                    transition.
    """
    ### YOUR CODE HERE
    if transition == 'S':
        self.stack.append(self.buffer.pop(0))
    elif transition == 'LA':
        self.dependencies.append((self.stack[-1], self.stack[-2]))
        self.stack.pop(-2)
    elif transition == 'RA':
        self.dependencies.append((self.stack[-2], self.stack[-1]))
        self.stack.pop(-1)
```

* Minibatch版本的parser

```

def minibatch_parse(sentences, model, batch_size):
    """Parses a list of sentences in minibatches using a model.

    Args:
        sentences: A list of sentences to be parsed (each sentence is a list of words)
        model: The model that makes parsing decisions. It is assumed to have a function
               model.predict(partial_parses) that takes in a list of PartialParses as input and
               returns a list of transitions predicted for each parse. That is, after calling
                   transitions = model.predict(partial_parses)
               transitions[i] will be the next transition to apply to partial_parses[i].
        batch_size: The number of PartialParses to include in each minibatch
    Returns:
        dependencies: A list where each element is the dependencies list for a parsed sentence.
                      Ordering should be the same as in sentences (i.e., dependencies[i] should
                      contain the parse for sentences[i]).
    """

    ### YOUR CODE HERE
    partial_parses = [PartialParse(s) for s in sentences]
    unfinished_parse = partial_parses
    while len(unfinished_parse) > 0:
        minibatch = unfinished_parse[0:batch_size]
        while len(minibatch) > 0:
            transitions = model.predict(minibatch)
            for index, action in enumerate(transitions):
                minibatch[index].parse_step(action)
            minibatch = [parse for parse in minibatch if len(parse.stack) > 1 or len(parse.buffer) > 0]
        unfinished_parse = unfinished_parse[batch_size:]
    dependencies = []
    for n in range(len(sentences)):
        dependencies.append(partial_parses[n].dependencies)
    ### END YOUR CODE

    return dependencies
```

* xavier初始化

```python
def _xavier_initializer(shape, **kwargs):
    """Defines an initializer for the Xavier distribution.
    Specifically, the output should be sampled uniformly from [-epsilon, epsilon] where
        epsilon = sqrt(6) / <sum of the sizes of shape's dimensions>
    e.g., if shape = (2, 3), epsilon = sqrt(6 / (2 + 3))

    This function will be used as a variable initializer.

    Args:
        shape: Tuple or 1-d array that species the dimensions of the requested tensor.
    Returns:
        out: tf.Tensor of specified shape sampled from the Xavier distribution.
    """
    ### YOUR CODE HERE
    epsilon = np.sqrt(6 / np.sum(shape))
    out = tf.random_uniform(shape=shape,minval=-epsilon,maxval=epsilon)

    ### END YOUR CODE
    return out
```

* 神经网络训练

**给定```stack,buffer```以及```dependencies```，来预测下一个转换状态是什么？**

* 词嵌入层

```python
def add_embedding(self):
    """Adds an embedding layer that maps from input tokens (integers) to vectors and then
    concatenates those vectors:
        - Creates a tf.Variable and initializes it with self.pretrained_embeddings.
        - Uses the input_placeholder to index into the embeddings tensor, resulting in a
            tensor of shape (None, n_features, embedding_size).
        - Concatenates the embeddings by reshaping the embeddings tensor to shape
            (None, n_features * embedding_size).

    Hint: You might find tf.nn.embedding_lookup useful.
    Hint: You can use tf.reshape to concatenate the vectors. See following link to understand
        what -1 in a shape means.
        https://www.tensorflow.org/api_docs/python/tf/reshape

    Returns:
        embeddings: tf.Tensor of shape (None, n_features*embed_size)
    """
    ### YOUR CODE HERE
    emb = tf.Variable(self.pretrained_embeddings)
    looked_up = tf.nn.embedding_lookup(emb, self.input_placeholder)
    embeddings = tf.reshape(looked_up, [-1, self.config.n_features*self.config.embed_size])
    ### END YOUR CODE
    return embeddings
```

* 预测

```python
def add_prediction_op(self):
    """Adds the 1-hidden-layer NN:
        h = Relu(xW + b1)
        h_drop = Dropout(h, dropout_rate)
        pred = h_dropU + b2

    Note that we are not applying a softmax to pred. The softmax will instead be done in
    the add_loss_op function, which improves efficiency because we can use
    tf.nn.softmax_cross_entropy_with_logits

    Use the initializer from q2_initialization.py to initialize W and U (you can initialize b1
    and b2 with zeros)

    Hint: Note that tf.nn.dropout takes the keep probability (1 - p_drop) as an argument.
            Therefore the keep probability should be set to the value of
            (1 - self.dropout_placeholder)

    Returns:
        pred: tf.Tensor of shape (batch_size, n_classes)
    """

    x = self.add_embedding()
    ### YOUR CODE HERE
    W = tf.Variable(tf.truncated_normal([self.config.n_features*self.config.embed_size, self.config.hidden_size]),
                        dtype = tf.float32)
    b1 = tf.Variable(tf.constant(0, tf.float32, [self.config.hidden_size]))
    U = tf.Variable(tf.truncated_normal([self.config.hidden_size, self.config.n_classes]),
                        dtype = tf.float32)
    b2 = tf.Variable(tf.constant(0, tf.float32, [self.config.n_classes]))

    h = tf.nn.relu(tf.matmul(x, W) + b1)
    h_drop = tf.nn.dropout(h, self.config.dropout)
    pred = tf.matmul(h_drop,U) + b2
    ### END YOUR CODE
    return pred
```

* loss

```python
def add_loss_op(self, pred):
    """Adds Ops for the loss function to the computational graph.
    In this case we are using cross entropy loss.
    The loss should be averaged over all examples in the current minibatch.

    Hint: You can use tf.nn.softmax_cross_entropy_with_logits to simplify your
                implementation. You might find tf.reduce_mean useful.
    Args:
        pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                network before the softmax layer.
    Returns:
        loss: A 0-d tensor (scalar)
    """
    ### YOUR CODE HERE
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels = self.labels_placeholder))

    ### END YOUR CODE
    return loss
```
* train_op

```python
def add_training_op(self, loss):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train. See

    https://www.tensorflow.org/api_docs/python/tf/train/Optimizer

    for more information.

    Use tf.train.AdamOptimizer for this model.
    Use the learning rate from self.config.
    Calling optimizer.minimize() will return a train_op object.

    Args:
        loss: Loss tensor, from cross_entropy_loss.
    Returns:
        train_op: The Op for training.
    """
    ### YOUR CODE HERE
    train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
    ### END YOUR CODE
    return train_op
```

## 3.Recurrent Neural Networks:Language Modeling 
