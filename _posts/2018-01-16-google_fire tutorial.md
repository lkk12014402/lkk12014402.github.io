---
layout:     post
title:      "Google Fire使用"
subtitle:   "Python"
date:       2018-01-14
author:     "hadxu"
header-img: "img/in-post/hadxu.jpg"
tags:
    - Python
---


# Google Fire使用

前几天看了Pytorch的教程，发现很多同学在使用Google的框架fire。开始不知道是啥，后来知道，原来是命令行解析工具。突然发现，搞机器学习的话，需要大量的参数配置，这样可以通过命令行来使用。今天简单介绍一下fire的使用。

### 安装

```
pip install fire
```

### 使用

首先假设有一个类有两个方法

```
class Calc(object):
	def double(self,number):
		return number ** 2

	def sub(self,a,b):
		return a - b
# 将方法注册进去
if __name__ == '__main__':
	fire.Fire(Calc)
```
终端使用

```
python fire_test.py sub --a 13 --b 14
```

#### 传入数组

```
def order_by_length(*items):
  sorted_items = sorted(items, key=lambda item: (len(str(item)), str(item)))
  return ' '.join(sorted_items)

if __name__ == '__main__':
  fire.Fire(order_by_length)
```

#### 终端传入类型

```
$ python example.py 10
int
$ python example.py 10.0
float
$ python example.py hello
str
$ python example.py '(1,2)'
tuple
$ python example.py [1,2]
list
$ python example.py True
bool
$ python example.py {name: David}
dict
```

## 总结 
使用该工具是非常方便便于我们调试的，但是其实还有一个办法就是新建一个Config.py文件，在Config中将所需要的参数配置好，在模型配置的时候读入即可。