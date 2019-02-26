---
layout:     post
title:      "如何正确使用colab"
subtitle:   "colab"
date:       2018-07-29
author:     "hadxu"
header-img: "img/hadxu.jpg"
tags:
    - Python
    - Google
    - colab
---

# 如何正确使用colab

今年刚开始，Google开放了colab。这是什么呢？这是一个云上编程平台，只要你有一个连接Google的浏览器，就能够随时编程，同时代码一次编写，随时运行，因为是保存在Google driver中的。
再者，google为每一个运行时提供了12小时的GPU支持，gpu型号为K80，虽然不是特别好，但也是够了，相对于我们的CPU的话。

**colab的编程环境与我们常常使用的jupyter notebook是一致的**

我们来看看colab中的Python包。

```shell
Package                  Version  
------------------------ ---------
absl-py                  0.3.0    
altair                   2.1.0    
astor                    0.7.1    
beautifulsoup4           4.6.0    
bleach                   2.1.3    
cachetools               2.1.0    
certifi                  2018.4.16
chardet                  3.0.4    
crcmod                   1.7      
cycler                   0.10.0   
decorator                4.3.0    
entrypoints              0.2.3    
future                   0.16.0   
gast                     0.2.0    
google-api-core          1.2.1    
google-api-python-client 1.6.7    
google-auth              1.4.2    
google-auth-httplib2     0.0.3    
google-auth-oauthlib     0.2.0    
google-cloud-bigquery    1.1.0    
google-cloud-core        0.28.1   
google-cloud-language    1.0.2    
google-cloud-storage     1.8.0    
google-cloud-translate   1.3.1    
google-colab             0.0.1a1  
google-resumable-media   0.3.1    
googleapis-common-protos 1.5.3    
grpcio                   1.13.0   
h5py                     2.8.0    
html5lib                 1.0.1    
httplib2                 0.11.3   
idna                     2.6      
ipykernel                4.6.1    
ipython                  5.5.0    
ipython-genutils         0.2.0    
Jinja2                   2.10     
jsonschema               2.6.0    
jupyter-client           5.2.3    
jupyter-core             4.4.0    
Keras                    2.1.6    
Markdown                 2.6.11   
MarkupSafe               1.0      
matplotlib               2.1.2    
mistune                  0.8.3    
mpmath                   1.0.0    
nbconvert                5.3.1    
nbformat                 4.4.0    
networkx                 2.1      
nltk                     3.2.5    
notebook                 5.2.2    
numpy                    1.14.5   
oauth2client             4.1.2    
oauthlib                 2.1.0    
olefile                  0.45.1   
opencv-python            3.4.2.17 
packaging                17.1     
pandas                   0.22.0   
pandas-gbq               0.4.1    
pandocfilters            1.4.2    
patsy                    0.5.0    
pexpect                  4.6.0    
pickleshare              0.7.4    
Pillow                   4.0.0    
pip                      18.0     
plotly                   1.12.12  
pluggy                   0.6.0    
portpicker               1.2.0    
prompt-toolkit           1.0.15   
protobuf                 3.6.0    
psutil                   5.4.6    
ptyprocess               0.6.0    
py                       1.5.4    
pyasn1                   0.4.4    
pyasn1-modules           0.2.2    
Pygments                 2.1.3    
pyparsing                2.2.0    
pystache                 0.5.4    
python-dateutil          2.5.3    
pytz                     2018.5   
PyWavelets               0.5.2    
PyYAML                   3.13     
pyzmq                    16.0.4   
requests                 2.18.4   
requests-oauthlib        1.0.0    
rsa                      3.4.2    
scikit-image             0.13.1   
scikit-learn             0.19.2   
scipy                    0.19.1   
seaborn                  0.7.1    
setuptools               39.1.0   
simplegeneric            0.8.1    
six                      1.11.0   
statsmodels              0.8.0    
sympy                    1.1.1    
tensorboard              1.9.0    
tensorflow               1.9.0    
tensorflow-hub           0.1.1    
termcolor                1.1.0    
terminado                0.8.1    
testpath                 0.3.1    
toolz                    0.9.0    
tornado                  4.5.3    
tox                      3.1.2    
traitlets                4.3.2    
typing                   3.6.4    
uritemplate              3.0.0    
urllib3                  1.22     
vega-datasets            0.5.0    
virtualenv               16.0.0   
wcwidth                  0.1.7    
webencodings             0.5.1    
Werkzeug                 0.14.1   
wheel                    0.31.1   
xgboost                  0.7.post4
```

可以发现，基本数据挖掘方面的包都已经包含。那么，如果我需要的包没有怎么办？别急，使用如下方式，比如我想使用```Pytorch```,
可以这样安装
```python
!pip install torch
```
而且，都是外网，速度非常快。

**当我们有环境之后，需要数据，那么数据哪里来？**

一个非常好的办法就是讲数据放在google driver中，那么，如何来读取数据？

1. 将需要的数据放入到Google driver中，并且记录文件id。(右键点击文件，点击```get charebale link```,获取后面id所对应的内容。即```https://drive.google.com/open?id=0B8cPzLyASEU-c3RhcnRlcl9maWxl```为```0B8cPzLyASEU-c3RhcnRlcl9maWxl```)

2. 打开colab notebook

```python
# 安装google python driver
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# 授权
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# 读取文件
train_downloaded = drive.CreateFile({'id': '获取的idxxx'})
train_downloaded.GetContentFile('train.csv')

# 与平常read_csv无异
df_train = pd.read_csv('train.csv')

df_train.head()
```

通过这种方式我们能够随时随地进行编程。赶快学起来吧，用其来打Kaggle非常不错。
