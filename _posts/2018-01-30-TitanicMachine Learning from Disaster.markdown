---
layout:     post
title:      "TitanicMachine Learning from Disaster"
subtitle:   "kaggle"
date:       2018-01-30
author:     "hadxu"
header-img: "img/in-post/cs224n/cs224n_head.png"
tags:
    - Kaggle
    - Python
---

# TitanicMachine Learning from Disaster
***
不知道为啥，突然喜欢打比赛了，然而自己的实际操作能力太弱了，这几天整理了一下```TitanicMachine Learning from Disaster```的比赛，从开始建模，到最后提交成绩，将整个算法的流程梳理了一遍，希望对读者有用。

## baseline
* 首先我们来看一下该数据集

```python
# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score

# Figures inline and set visualization style
%matplotlib inline
sns.set()
```

* 导入数据集

```python
# Import test and train datasets
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# View first lines of training data
df_train.head(n=4)
```

* 结果为

![](/img/in-post/2018-01-3001.jpg)

* 查看df信息

```python
df_train.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
```

> 可以看出来，有些列是```Nan```的，我们首先对生存者进行分析

```python
sns.countplot(x='Survived',data=df_train)
```

![](/img/in-post/Jietu20180130-212117.jpg)

> 可以看到活着的人很少，于是我们先提交全是为0的结果。

```python
df_test['Survived'] = 0
df_test[['PassengerId','Survived']].to_csv('no_survivors.csv',index=False)
```

>准确率在60%左右，不是特别差，😀

## 第一次改进

> 我们直觉就是，在男女中，生存的比例是多少？

```python
sns.countplot(x='Sex',data=df_train)
sns.factorplot(x='Survived',col='Sex',kind='count',data=df_train)
```
![](/img/in-post/Jietu20180130-212706.jpg)


* 还可以探索各种各样的特征。

## 第二次改进

* 将训练集和测试集放在一起

```python
# Store target variable of training data in a safe place
survived_train = df_train.Survived

# Concatenate training and test sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])

data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())

# Check out info of data
data.info()
```

* 将变量离散化**非常重要**

```python
data = pd.get_dummies(data, columns=['Sex'], drop_first=True)
```

* 提取特征

```python
data = data[['Sex_male', 'Fare', 'Age','Pclass', 'SibSp']]
```

* 训练集测试集分开

```python
data_train = data.iloc[:891]
data_test = data.iloc[891:]
```

```python
X = data_train.values
test = data_test.values
y = survived_train.values
```

* 训练

```
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)
```

* 预测

```python
Y_pred = clf.predict(test)
df_test['Survived'] = Y_pred
```

* 在这里有一个疑问，我们为什么选取树的深度为3？

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y)
dep = np.arange(1, 9)
train_accuracy = np.empty(len(dep))
test_accuracy = np.empty(len(dep))

# Loop over different values of k
for i, k in enumerate(dep):
    # Setup a k-NN Classifier with k neighbors: knn
    clf = tree.DecisionTreeClassifier(max_depth=k)

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    #Compute accuracy on the training set
    train_accuracy[i] = clf.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = clf.score(X_test, y_test)

# Generate plot
plt.title('clf: Varying depth of tree')
plt.plot(dep, test_accuracy, label = 'Testing Accuracy')
plt.plot(dep, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Depth of tree')
plt.ylabel('Accuracy')
plt.show()
```

结果如

![](/img/in-post/Jietu20180130-215725.jpg)



## 第三次改进(更多特征)

* 将姓名提取出来

```python
data['Title'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
sns.countplot(x='Title', data=data);
plt.xticks(rotation=45);
```

![](/img/in-post/Jietu20180130-214113.jpg)

* 将身份提出出来

```python
data['Title'] = data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
data['Title'] = data['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',
                                            'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')
sns.countplot(x='Title', data=data);
plt.xticks(rotation=45);
```

![](/img/in-post/Jietu20180130-214222.jpg)

* 是否有cabin

```python
data['Has_Cabin'] = ~data.Cabin.isnull()
```

* 删除无用特征

```python
data.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1, inplace=True)
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())
data['Embarked'] = data['Embarked'].fillna('S')
data.info()
```

* 将年龄、Fare进行分区间

```python
data['CatAge'] = pd.qcut(data.Age, q=4, labels=False )
data['CatFare']= pd.qcut(data.Fare, q=4, labels=False)
data = data.drop(['Age', 'Fare'], axis=1)
```

* 构造特征

```python
data['Fam_Size'] = data.Parch + data.SibSp
data = data.drop(['SibSp','Parch'], axis=1)
```

* 进行量化

```python
data_dum = pd.get_dummies(data, drop_first=True)
```

* 构造数据集

```
# Split into test.train
data_train = data_dum.iloc[:891]
data_test = data_dum.iloc[891:]

# Transform into arrays for scikit-learn
X = data_train.values
test = data_test.values
y = survived_train.values
```

* 训练

```python
dep = np.arange(1,9)
param_grid = {'max_depth' : dep}

# Instantiate a decision tree classifier: clf
clf = tree.DecisionTreeClassifier()

# Instantiate the GridSearchCV object: clf_cv
clf_cv = GridSearchCV(clf, param_grid=param_grid, cv=5)

# Fit it to the data
clf_cv.fit(X, y)

# Print the tuned parameter and score
print("Tuned Decision Tree Parameters: {}".format(clf_cv.best_params_))
print("Best score is {}".format(clf_cv.best_score_))
```

* 保存

```
Y_pred = clf_cv.predict(test)
df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('dec_tree_feat_eng.csv', index=False)
```


**最终的效果大概是78.468，效果还不错，接下来考虑使用模型融合。**





