---
layout:     post
title:      "拍拍贷-第三届魔镜杯rank6解决方案"
subtitle:   "魔镜杯"
date:       2018-07-28
author:     "hadxu"
header-img: "img/in-post/paipai/background.png"
tags:
    - Python
    - NLP
---

# 拍拍贷第三届魔镜杯rank 6 SuperGUTs解决方案

### 该问题是关于自然语言处理的一个非常常见的问题即判断两个句子的相似性，很容易想到该问题在智能客服或者垂直问答系统中使用，那么如何判断两个句子的相似性呢？

* 使用树模型(非结构化特征难以构建)
* 使用传统LR(弊端同上)
* 使用nn(对于非结构化特征非常适用)

#### 同时，该大赛已经将训练好的词向量以及句子向量已经给我们参赛者，我们的目标就是在该向量上进行构建模型，判断两个句子的相似概率？当时打这个比赛就是因为问题简单，且向量训练好，于是我们团队就搞一波。下面，就一步一步介绍我们的方法。


## 1. 构建baseline(麻婆豆腐AI)

很明显，baseline为一个简单的nn，结构如下：

```python
   INPUT1             IINPUT2
    |                    |
embedding_q1        embedding_q2
    |                    |
   Bi-GRU               Bi-GRU
    |                    |
   Bi-GRU              Bi-GRU
    |                    |
  Global maxpool    Global maxpool
    |                    |
  global average pool global average pool
    |                    |
    ----------------------
          |
        person/全连接
          |
          |
          |
        output
```

* 该baseline在线上是0.23的成绩，单模型，一个cv。

## 2. 改进baseline(添加BN+SDP)

老实说，该组件我是没有见过，[qrfaction](https://qrfaction.github.io)指出，该组件能够很好的使得输入标准化并且[空间dropout](https://arxiv.org/pdf/1411.4280.pdf)能够加噪，而能够不丢失语义信息，因此，我们采用了该组件

```python
norm = BatchNormalization()
q1 = embedd_word(question1)
q1 = norm(q1)
q1 = SpatialDropout1D(0.2)(q1)
q2 = embedd_word(question2)
q2 = norm(q2)
q2 = SpatialDropout1D(0.2)(q2)
```

* 值得注意的是，仅仅使用该组件能够使得线上0.19，10cv能够达到0.171。

## 改进聚合组件

聚合组件，是用来将两个句子输入的向量进行聚合，那么聚合的函数我们使用的是皮尔逊系数、harmonic系数、Tanimoto系数以及Dice，通过该聚合以及两种池化，在线上能够达到0.1693的成绩。

整体的模型：


![](/img/in-post/paipai/paipai.png)

## 半监督
模型，我们调试了很多很多次，不停更换各种组件，貌似我们单模型是最好的？模型不能够再次提高，我们采取半监督的方式来提高我们的线上成绩，最后是0.1635。

## 图特征
主办方没有对图特征进行明确说明，但是如果不用的话，估计进不去决赛hhh，不知道谁先使用了下，其他队伍也跟着使用了。

我们采用的图特征有一下几个

1.  qi , qj 的公共边权重和

2.  qi , qj 的所接所有边权重和

3.  去掉edge(qi,qj)后qi与qj之间的最短带权路径

4.  pagerank值

5.  图的邻接矩阵分解获得qi,qj的embedding 向量

6.  q_id所属连通分量的node个数和edge个数

图特征能够给我们带来2个千分点。

## 数据增强

我们团队在数据增强这点做得非常失败，同时在答辩现场也没有理解bird的方法，等蚂蚁结束看下bird他们的源代码吧！


## 最终loss下降整体图


![](/img/in-post/paipai/loss.png)


## 感想

感谢我们团队4人的付出，尤其是[qrfaction](https://qrfaction.github.io)，在模型调参直觉上，一调一个准。我们团队经过这次的比赛获得的经验有

1. 模型保存，一定要迭代着保存模型！！！(血的教训)
2. 模型集成在初赛时一定要调完(后期的提交机会尤其重要)
3. 认识了很多大佬，同时感觉与北航、北邮的差距不是一般的大！
4. 下面任务是打比赛，打kaggle，至少弄一个kaggle master。

## 花絮

在上海的三天，第一天晚上我们排练ppt没有出去，第二天一天都在酒店里，进行答辩，第二天晚上忙里偷闲去了东方明珠周围玩了一波无人机。

夜上海

![](/img/in-post/paipai/IMG_1304_temp.png)

夜上海全景图


![](/img/in-post/paipai/IMG_1303.png)



