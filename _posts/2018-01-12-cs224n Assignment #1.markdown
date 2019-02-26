---
layout:     post
title:      "CS224N Assignment #1"
subtitle:   "cs224n"
date:       2018-01-12
author:     "hadxu"
header-img: "img/in-post/cs224n/cs224n_head.png"
tags:
    - Tensorflow
    - Python
    - cs224n
---

# CS224N Assignment #1

1. 
推导Softmax

```
softmax(x) = softmax(x+c)
```

**这样推导有什么用呢？**

在于计算exp的期间，有可能出现溢出，因此需要减去最大值。
计算softmax代码如下：
```
orig_shape = x.shape

if len(x.shape) > 1:
    # Matrix
    ### YOUR CODE HERE
    x -= np.max(x,axis=1).reshape((-1,1))
    x = np.exp(x)
    x /= np.sum(x,axis=1).reshape((-1,1))

    ### END YOUR CODE
else:
    # Vector
    ### YOUR CODE HERE
    x -= np.max(x)
    x = np.exp(x)
    x /= np.sum(x)

    ### END YOUR CODE

assert x.shape == orig_shape
return x
```

2. 习题e,计算sigmoid及其导数

```
def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Arguments:
    x -- A scalar or numpy array.

    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE
    
    s = 1./(1+np.exp(-x))

    ### END YOUR CODE

    return s

def sigmoid_grad(s):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input s should be the sigmoid
    function value of your original input x.

    Arguments:
    s -- A scalar or numpy array.

    Return:
    ds -- Your computed gradient.
    """

    ### YOUR CODE HERE
    ds = s*(1-s)
    ### END YOUR CODE

    return ds
```

2. (f)实现梯度检查，使用传统的梯度计算检查

```
def gradcheck_naive(f, x):
    """ Gradient check for a function f.

    Arguments:
    f -- a function that takes a single argument and outputs the
         cost and its gradients
    x -- the point (numpy array) to check the gradient at
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4        # Do not change this!

    # Iterate over all indexes ix in x to check the gradient.
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        # Try modifying x[ix] with h defined above to compute numerical
        # gradients (numgrad).

        # Use the centered difference of the gradient.
        # It has smaller asymptotic error than forward / backward difference
        # methods. If you are curious, check out here:
        # https://math.stackexchange.com/questions/2326181/when-to-use-forward-or-central-difference-approximations

        # Make sure you call random.setstate(rndstate)
        # before calling f(x) each time. This will make it possible
        # to test cost functions with built in randomness later.

        ### YOUR CODE HERE:
        
        x[ix] += h
        random.setstate(rndstate)
        f_1 = f(x)[0]

        x[ix] -= 2*h
        random.setstate(rndstate)
        f_2 = f(x)[0]

        numgrad = (f_1 - f_2)/(2*h)


        ### END YOUR CODE

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (
                grad[ix], numgrad)
            return

        it.iternext() # Step to next dimension

    print "Gradient check passed!"
```

2. (g) 实现neural
> 正向反向推导，很简单。

```
def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.
    ### YOUR CODE HERE: forward propagation
    
    z1 = np.dot(X,W1) + b1
    a1 = sigmoid(z1)
    scores = np.dot(a1,W2) + b2

    y_hat = softmax(scores)

    cost = -np.sum(labels*np.log(y_hat))


    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    d_score = y_hat - labels
    gradW2 = np.dot(a1.T,d_score)
    gradb2 = np.sum(d_score,axis=0)

    d_a1 = np.dot(d_score,W2.T)*sigmoid_grad(a1)

    gradW1 = np.dot(X.T,d_a1)
    gradb1 = np.sum(d_a1,axis=0)


    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad
```

3. 实现word2vec

这部分是课程的核心，首先要我们实现一个方法softmaxCostAndGradient。

```
def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    # 计算每一个词的的预测与真实值之间的误差，并返回梯度。
    
    ### YOUR CODE HERE
    
    vhat = predicted
    z = np.dot(outputVectors,vhat)
    preds = softmax(z)
    cost = -np.log(preds[target])

    z = preds.copy()
    z[target] -= 1
    grad = np.outer(z, vhat)
    gradPred = np.dot(outputVectors.T, z)


    ### END YOUR CODE

    return cost, gradPred, grad
```

实现skipgram。

```
def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    j = tokens[currentWord]  #获得当前词的下标
    predicted = inputVectors[j] #获得当前词的输入向量
    for word in contextWords:
        target = tokens[word] #获得上下文的每一个词的下标
        cost_, gradPred, grad = word2vecCostAndGradient(predicted, target, outputVectors, dataset)
        cost += cost_
        gradIn[j] += gradPred
        gradOut += grad

    ### END YOUR CODE

    return cost, gradIn, gradOut
```

还需要我们实现CBOW，其实与skip-gram差不多。

```
def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    

    predicted_indices = [tokens[word] for word in contextWords]


    predicted_vectors = inputVectors[predicted_indices]


    predicted = np.sum(predicted_vectors, axis=0)


    target = tokens[currentWord]
    cost, gradIn_predicted, gradOut = word2vecCostAndGradient(predicted, target, outputVectors, dataset)
    # 将所有的预测的概率加起来
    for i in predicted_indices:
        gradIn[i] += gradIn_predicted

    ### END YOUR CODE

    return cost, gradIn, gradOut
```

4. 在q3_sgd.py中需要我们实现sgd算法。简单！

```
### YOUR CODE HERE
cost,grad = f(x)
x -= step * grad
### END YOUR CODE
```
