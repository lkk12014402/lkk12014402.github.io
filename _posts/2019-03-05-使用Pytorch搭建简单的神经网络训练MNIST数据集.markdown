---
layout:     post
title:      "使用Pytorch搭建简单的卷积神经网络训练MNIST数据集"
subtitle:   "Pytorch"
date:       2019-03-05
author:     "lkk"
header-img: ""
tags:
    - Pytorch
    - 深度学习
    - 卷积神经网络
    - Python
---

##MNIST数据集详解

MNIST 数据集已经是一个被”嚼烂”了的数据集, 很多教程都会对它”下手”, 几乎成为一个 “典范”. 不过有些人可能对它还不是很了解, 下面来介绍一下：

MNIST 数据集可在 http://yann.lecun.com/exdb/mnist/ 获取, 它包含了四个部分:

![](/img/8389494-852b21740a506378.jpg)


- Training set images: train-images-idx3-ubyte.gz (9.9 MB, 解压后 47 MB, 包含 60,000 个样本)
- Training set labels: train-labels-idx1-ubyte.gz (29 KB, 解压后 60 KB, 包含 60,000 个标签)
- Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 解压后 7.8 MB, 包含 10,000 个样本)
- Test set labels: t10k-labels-idx1-ubyte.gz (5KB, 解压后 10 KB, 包含 10,000 个标签)

MNIST 数据集来自美国国家标准与技术研究所, National Institute of Standards and Technology (NIST)。训练集 (training set) 由来自 250 个不同人手写的数字构成, 其中 50% 是高中学生, 50% 来自人口普查局 (the Census Bureau) 的工作人员。测试集(test set) 也是同样比例的手写数字数据。

我们可以看出这个其实并不是普通的文本文件或是图片文件，而是一个压缩文件，下载并解压出来，我们看到的是二进制文件。下面根据官方文档，介绍一下数据的存储格式。

针对训练标签集，官网上陈述有

![](/img/8389494-3ae56a29c9090ad3.jpg)

官网说法，训练集是有60000个用例的，也就是说这个文件里面包含了60000个标签内容，每一个标签的值为0到9之间的一个数；回到我们的训练标签集上，按上面说的，我们先解析每一个属性的含义，offset代表了字节偏移量，也就是这个属性的二进制值的偏移是多少；type代表了这个属性的值的类型；value代表了这个属性的值是多少；description是对这个的说明；所以呢，这里对上面的进行一下说明，它的说法是“从第0个字节开始有一个32位的整数，它的值是0x00000801，它是一个魔数；从第4个字节开始有一个32位的整数，它的值是60000，它代表了数据集的数量；从第8个字节开始有一个unsigned byte，它的值是??，是一个标签值….”；我们现在针对我们看到的文件进行解说，看图

![](/img/8389494-1730445a99acbfa9.jpg)

首先我们知道用sublime打开这个文件（是解压过的），是用十六进制表示的，也就是说里面的每一个数字代表了四个位，两个数字代表了一个字节；我们首先看到偏移量为0字节处0000 0801它就是代表了魔数，它的值为00000801，这里补充说一下什么是魔数，其实它就是一个校验数，用来判断这个文件是不是MNIST里面的train-labels.idx1-ubyte文件；接着往下看偏移量为4字节处0000 ea60,我们知道按照上面说过的这个应该是表示容量数，也就是60000,而60000的十六进制就是ea60,满足；再看偏移量为8字节处05，它就表示我们的标签值了，也就是说第一个图片的标签值为5,后面的也是依此类推；接下来我们来看训练图片集，同样从官网上可以看到

![](/img/8389494-3f596ed6ea9e6028.jpg)


其解说与上面的标签文件类似，但是这里还要补充说明一下，在MNIST图片集中，所有的图片都是28×28的，也就是每个图片都有28×28个像素；看回我们的上述图片，其表示，我们的train-images-idx3-ubyte文件中偏移量为0字节处有一个4字节的数为0000 0803表示魔数；接下来是0000ea60值为60000代表容量，接下来从第8个字节开始有一个4字节数，值为28也就是0000 001c，表示每个图片的行数；从第12个字节开始有一个4字节数，值也为28,也就是0000001c表示每个图片的列数；从第16个字节开始才是我们的像素值，用图片说话；而且每784个字节代表一幅图片

补充说明：在图示中我们可以看到有一个MSB first，其全称是”Most Significant Bit first”,相对称的是一个LSB first，“Least Significant Bit”; MSB first是指最高有效位优先，也就是我们的大端存储，而LSB对应小端存储；关于大端，小端


下面使用python来对MNIST数据集进行解析：

1.将数据集解析为图片形式存储

```python
import numpy as np
import struct
import matplotlib.pyplot as plt

from PIL import Image
import os

# 训练集文件
train_images_idx3_ubyte_file = r'./data_mnist/raw/train-images-idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = r'./data_mnist/raw/train-labels-idx1-ubyte'

# # 测试集文件
test_images_idx3_ubyte_file = './data_mnist/raw/t10k-images-idx3-ubyte'
# # 测试集标签文件
test_labels_idx1_ubyte_file = './data_mnist/raw/t10k-labels-idx1-ubyte'


def decode_data(data_file, label_file):

    output_path = './parse_mnist_data/train'
    # if not os.path.exists(output_path):
    #     os.mkdir(output_path)

    bin_data = open(data_file, 'rb').read()
    bin_label = open(label_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    image_offset = 0
    image_fmt_header = '>iiii'
    magic_number, image_num, num_rows, num_cols = struct.unpack_from(image_fmt_header, bin_data, image_offset)
    print('数据：魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, image_num, num_rows, num_cols))

    # 解析文件头信息，依次为魔数和标签数
    label_offset = 0
    label_fmt_header = '>ii'
    magic_number, label_num = struct.unpack_from(label_fmt_header, bin_data, label_offset)
    print('标签：魔数:%d, 图片数量: %d张' % (magic_number, label_num))

    assert image_num == label_num

    # 解析数据集
    image_size = num_rows * num_cols
    image_offset += struct.calcsize(image_fmt_header)
    fmt_image = '>' + str(image_size) + 'B'

    # 解析标签集
    label_offset += struct.calcsize(label_fmt_header)
    fmt_label = '>B'

    for i in range(image_num):
        if (i + 1) % 1000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        image = np.array(struct.unpack_from(fmt_image, bin_data, image_offset)).reshape((num_rows, num_cols))
        image_offset += struct.calcsize(fmt_image)

        label = struct.unpack_from(fmt_label, bin_label, label_offset)[0]
        label_offset += struct.calcsize(fmt_label)

        output_image_path = output_path + '/' + str(label)

        if not os.path.exists(output_image_path):
            os.makedirs(output_image_path)

        im = Image.fromarray(image).convert('L')
        # print(im.mode)

        ouput_image = os.path.join(output_image_path, '{}-{}.jpeg'.format(i, label))
        # print(ouput_image)
        im.save(ouput_image)


if __name__ == '__main__':
    decode_data(train_images_idx3_ubyte_file, train_labels_idx1_ubyte_file)
```

2.只读到内存中查看

```python
def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    每2个数字代表一个字节，前4个字节是 0000 0803 代表魔数，表明这是LeCunn官方数据。
    接下来的4个字节 0000 ea60 代表有60，000个图片。再接下来， 0000 001c代表28像素，第一行最后四个字节也是代表28像素。

    从第二行开始之后的784个字节是第一张图片对应的像素值。每行有16个字节，所以一张图片占 784/16=49行。
    每张图片解析出来，类似如下，可以使用数组形式来描述它

    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)




def run():
    train_images = load_train_images()
    train_labels = load_train_labels()
    # test_images = load_test_images()
    # test_labels = load_test_labels()

    # 查看前十个数据及其标签以读取是否正确
    for i in range(10):
        print(train_labels[i])
        plt.imshow(train_images[i], cmap='gray')
        plt.show()
    print('done')

if __name__ == '__main__':
    run()

```

3.借助numpy库解析更为简洁高效

参考：https://blog.csdn.net/simple_the_best/article/details/75267863



##搭建简单卷积神经网络训练MNIST数据集

> 环境：Pytorch 1.0 （有木有GPU均可）

1.导入相应模块、下载数据

```python
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torchvision
import time

print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 定义超参数
batch_size = 32
learning_rate = 1e-2
num_epoches = 20

# 下载训练集 MNIST 手写数字训练集
train_dataset = datasets.MNIST(
    root='./data_mnist', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(
    root='./data_mnist', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

dataloaders = {"train": train_loader, "test": test_loader}
dataset_sizes = {"train": len(train_dataset), "test": len(test_dataset)}

print(dataset_sizes)
````


2.可视化

```python
import matplotlib.pyplot as plt
import numpy as np

# 输出图像的函数

classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 随机得到一些训练图片
dataiter = iter(train_loader)
images, labels = dataiter.next()

# 显示图片
imshow(torchvision.utils.make_grid(images))
# 打印图片标签
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
```

![png](/img/output_3_0.png)


3.定义卷积神经网络

```python
# 定义 Convolution Network 模型
class Cnn(nn.Module):
    def __init__(self, in_channel, n_class):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 6, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.ReLU(True), nn.MaxPool2d(2, 2))

        self.fc = nn.Sequential(
            nn.Linear(400, 120), nn.Linear(120, 84), nn.Linear(84, n_class))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

```


4.初始化模型

```python
model = Cnn(1, 10)  # 图片大小是28x28
model = model.to(device)
```

5.查看网络结构

```python
from torchsummary import summary
summary(model, input_size=(1, 28, 28))
```

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 6, 28, 28]              60
              ReLU-2            [-1, 6, 28, 28]               0
         MaxPool2d-3            [-1, 6, 14, 14]               0
            Conv2d-4           [-1, 16, 10, 10]           2,416
              ReLU-5           [-1, 16, 10, 10]               0
         MaxPool2d-6             [-1, 16, 5, 5]               0
            Linear-7                  [-1, 120]          48,120
            Linear-8                   [-1, 84]          10,164
            Linear-9                   [-1, 10]             850
================================================================
Total params: 61,610
Trainable params: 61,610
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.11
Params size (MB): 0.24
Estimated Total Size (MB): 0.35
----------------------------------------------------------------
```


6.定义loss和optimizer

```python
# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
```

7.定义训练函数

```python
def run_model(model, criterion, optimizer, data_set, data_size, scheduler=None, phase='train', num_epochs=25):

    since = time.time()

    metrics = {'train_acc': [], 'train_loss': []}
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        if phase == 'train':
            #scheduler.step()
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in data_set[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / data_size[phase]
        epoch_acc = running_corrects.double() / data_size[phase]

        metrics['train_acc'].append(epoch_acc)
        metrics['train_loss'].append(epoch_loss)
        
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model, metrics
```

或者使用下面的函数，边训练边用测试集测试

```python
def run_model(model, criterion, optimizer, data_set, data_size, scheduler=None, num_epochs=25):
    since = time.time()

#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
    confusion_matrix = tnt.meter.ConfusionMeter(10, normalized=False)
    metrics = {'train_acc': [], 'train_loss': [], 'test_acc': [], 'test_loss': [], 'cm': []}
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
#                 scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_set[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        confusion_matrix.add(outputs.data, labels.data)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / data_size[phase]
            epoch_acc = running_corrects.double().item() / data_size[phase]

#             metrics['train_acc'].append(epoch_acc)
#             metrics['train_loss'].append(epoch_loss)
        
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'train':
                metrics['train_acc'].append(epoch_acc)
                metrics['train_loss'].append(epoch_loss)
            else:
                metrics['test_acc'].append(epoch_acc)
                metrics['test_loss'].append(epoch_loss)
                cm = confusion_matrix.value().copy()
                metrics['cm'].append(cm)
#             # deep copy the model
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())

#         print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
#     model.load_state_dict(best_model_wts)

    return model, metrics
```


8.开始训练

```python
model, metrics = run_model(model, criterion, optimizer, dataloaders, dataset_sizes, phase='train', num_epochs=25)
````

```python

    Epoch 0/24
    ----------
    train Loss: 1.0729 Acc: 0.6512
    Epoch 1/24
    ----------
    train Loss: 0.1788 Acc: 0.9454
    Epoch 2/24
    ----------
    train Loss: 0.1198 Acc: 0.9635
    Epoch 3/24
    ----------
    train Loss: 0.0980 Acc: 0.9699
    Epoch 4/24
    ----------
    train Loss: 0.0855 Acc: 0.9742
    Epoch 5/24
    ----------
    train Loss: 0.0762 Acc: 0.9761
    Epoch 6/24
    ----------
    train Loss: 0.0688 Acc: 0.9786
    Epoch 7/24
    ----------
    train Loss: 0.0645 Acc: 0.9802
    Epoch 8/24
    ----------
    train Loss: 0.0598 Acc: 0.9813
    Epoch 9/24
    ----------
    train Loss: 0.0568 Acc: 0.9825
    Epoch 10/24
    ----------
    train Loss: 0.0533 Acc: 0.9835
    Epoch 11/24
    ----------
    train Loss: 0.0503 Acc: 0.9843
    Epoch 12/24
    ----------
    train Loss: 0.0478 Acc: 0.9848
    Epoch 13/24
    ----------
    train Loss: 0.0461 Acc: 0.9850
    Epoch 14/24
    ----------
    train Loss: 0.0439 Acc: 0.9866
    Epoch 15/24
    ----------
    train Loss: 0.0423 Acc: 0.9870
    Epoch 16/24
    ----------
    train Loss: 0.0409 Acc: 0.9872
    Epoch 17/24
    ----------
    train Loss: 0.0388 Acc: 0.9878
    Epoch 18/24
    ----------
    train Loss: 0.0377 Acc: 0.9880
    Epoch 19/24
    ----------
    train Loss: 0.0356 Acc: 0.9887
    Epoch 20/24
    ----------
    train Loss: 0.0351 Acc: 0.9881
    Epoch 21/24
    ----------
    train Loss: 0.0335 Acc: 0.9892
    Epoch 22/24
    ----------
    train Loss: 0.0325 Acc: 0.9900
    Epoch 23/24
    ----------
    train Loss: 0.0314 Acc: 0.9900
    Epoch 24/24
    ----------
    train Loss: 0.0308 Acc: 0.9903
    Training complete in 20m 59s

```

9.可视化结果

```python
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator


def plot_metrics(metrics, title=None):
    max_epochs = len(metrics['train_acc']) + 1
    epochs = range(1, max_epochs)
    epochs_dx = np.linspace(epochs[0], epochs[-1], num=max_epochs * 4, endpoint=True)
    s_train_acc = interp1d(epochs, metrics['train_acc'], kind='cubic')
    s_train_loss = interp1d(epochs, metrics['train_loss'], kind='cubic')
    fig, ax = plt.subplots(1, 2)
    fig.subplots_adjust(right=2, top=0.85)
    if title is not None:
        st = fig.suptitle(title, fontsize=16)
        st.set_x(1)
        
    ax[0].plot(epochs, metrics['train_acc'], 'b.', label='train')
    ax[0].plot(epochs_dx, s_train_acc(epochs_dx), 'b')
    ax[0].legend(loc="lower right")
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epochs")
    ax[0].xaxis.set_major_locator(MultipleLocator(1))  # only integers in axis multiples of 1

    ax[1].plot(epochs, metrics['train_loss'], 'b.', label='train')
    ax[1].plot(epochs_dx, s_train_loss(epochs_dx), 'b')
    ax[1].legend(loc="upper right")
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Epochs")
    ax[1].xaxis.set_major_locator(MultipleLocator(1))
    plt.show()

plot_metrics(metrics, "Mnist-----train")
```

![](/img/output_10_0.png)

