---
layout:     post
title:      "记录一次keras与pytorch的源码比较"
subtitle:   "keras、pytorch"
date:       2018-08-08
author:     "hadxu"
header-img: "img/hadxu.jpg"
tags:
    - Python
    - keras
    - pytorch
---


# 记录一次Keras与Pytorch的源码比较

Keras，Pytorch两个都是我最喜欢的深度学习框架。keras的优点

1. 简洁
2. 快速上手
3. 工程使用广泛，得益于后端是```Tensorflow```
4. 数据挖掘竞赛常用，几分钟通过keras代码可以搭建baseline。

而Pytorch的有点则是

1. 神经网络透明
2. 速度快，得益于底层是C
3. 适用于研究，发论文。

可以发现，keras与Pytorch的侧重点不同，keras侧重工程，Pytroch侧重研究。不管怎么说，两个都是非常棒的框架，今天，我和大家分享下如何同时编写Keras与Pytorch的代码，并使得两个框架的结果是相通的。

##### 不适合小白，带有大量的源码阅读过程。需要读者同时熟练keras以及pytorch。使用的Pycharm代码编写环境，方便函数跳转。

### 题目:输入x(随机数)，然后对x+2，最后通过sigmoid函数输出。很简单是不是？如果想让两者的输出值都一样呢？

## keras版本

```Python
from keras.layers import *
from keras.models import Model

x_in = Input(shape=(10,))
x = Lambda(lambda x: x + 2)(x_in)
out = Dense(1, activation='sigmoid')(x)
model = Model(inputs=x_in, outputs=out)

value = np.random.rand(5, 10)
res = model.predict(value)
```

## pytorch版本

```python
import torch as t
from torch import nn as nn
from torch.nn import functional as F
value = np.random.rand(5, 10)
value = t.Tensor(value)
x = value + 2
linear = nn.Linear(10, 1)
out = linear(x)
out = F.sigmoid(out)
```

很快，我们将两个不同版本的源码写好了，很明显，输出的结果是不一样的。下面我们需要对```Dense、Linear```进行分析。

## keras的Dense分析

在神经网络中，Dense是非常常见的层，使得输入和输出进行全连接，但是，我们现在不知道Dense里面的权重初始化，于是我们点击进去看看

```python
def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
```

可以看见，Dense采用了```glorot_uniform```初始化方法，那么该方法计算公式是什么？我们查到


```python
def glorot_normal(seed=None):
    """Glorot normal initializer, also called Xavier normal initializer.

    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.

    # Arguments
        seed: A Python integer. Used to seed the random generator.

    # Returns
        An initializer.

    # References
        Glorot & Bengio, AISTATS 2010
        http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    """
    return VarianceScaling(scale=1.,
                           mode='fan_avg',
                           distribution='normal',
                           seed=seed)
```

可以发现,该初始化方式是随机初始化，那么如何判断两者的结果是否一致呢？

#### 采用常数初始化

查看Keras文档，常数初始化为```Constant```

```python
...
init = Constant(0.1)
out = Dense(1, activation='sigmoid',kernel_initializer=init)(x)
...

```

这个时候我们将权重全部设置为```0.1```。下面该设置pytorch的layer层的权重。

## pytorch的Linear分析

```python
def __init__(self, in_features, out_features, bias=True):
    super(Linear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = Parameter(torch.Tensor(out_features, in_features))
    if bias:
        self.bias = Parameter(torch.Tensor(out_features))
    else:
        self.register_parameter('bias', None)
    self.reset_parameters()

def reset_parameters(self):
    stdv = 1. / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdv, stdv)
    if self.bias is not None:
        self.bias.data.uniform_(-stdv, stdv)
```

看源代码可以发现，pytorch的权重初始化也是```glorot_uniform```初始化，我们需要进行常量初始化，并将bias设置为False。

```python
...
linear = nn.Linear(10, 1, bias=False)
nn.init.constant_(linear.weight, 0.1)
out = linear(x)
...
```

通过上面的代码，最终我们的结果是一致的。整个的代码如下

```python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     example
   Description :
   Author :       haxu
   date：          2018/7/31
-------------------------------------------------
   Change Activity:
                   2018/7/31:
-------------------------------------------------
"""
__author__ = 'haxu'

import numpy as np
from keras.layers import *
from keras import backend as K
from keras.models import Model
from keras.initializers import Constant
import torch as t
from torch.nn import functional as F
from torch import nn as nn

t.set_printoptions(precision=8)


def keras_model(value):
    x_in = Input(shape=(10,))
    x = Lambda(lambda x: x + 2)(x_in)
    init_w = Constant(0.1)
    out = Dense(1, activation='sigmoid', kernel_initializer=init_w)(x)
    model = Model(inputs=x_in, outputs=out)
    res = model.predict(value)
    print(res)


def pytorch(value):
    value = t.Tensor(value)
    x = value + 2
    linear = nn.Linear(10, 1, bias=False)
    nn.init.constant_(linear.weight, 0.1)
    out = linear(x)
    out = F.sigmoid(out.data.numpy())
    print(out)


if __name__ == '__main__':
    x = np.random.rand(5, 10)
    keras_model(x)
    print('*' * 50)
    pytorch(x)
```

输出

```shell
[[0.92670244]
 [0.9355231 ]
 [0.9238077 ]
 [0.92526585]
 [0.9187089 ]]
**************************************************
[[0.92670244]
 [0.9355231 ]
 [0.9238077 ]
 [0.9252657 ]
 [0.9187089 ]]
```

基本是一致的。


## Conclusion
通过这个小实验，我们学会了很多，其中包括keras的初始化方式，pytorch的初始化方式。


keras：

掌握

```python
from keras.initializers import Constant
init_w = Constant(0.1)
```

pytorch掌握

```python
nn.init.constant_(linear.weight, 0.1)
```

其实在任何深度学习框架中，都有一点就是自动求导的机制，笔者也写过深度学习框架方面的知识，[《从零实现深度学习框架》](https://hadxu.github.io/2018/03/12/%E4%BB%8E%E9%9B%B6%E5%AE%9E%E7%8E%B0%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6(%E7%AC%AC%E4%B8%80%E5%A4%A9)/)。












