---
layout:     post
title:      "Kaggle tips"
subtitle:   "kaggle"
date:       2018-01-31
author:     "hadxu"
header-img: "img/in-post/cs224n/cs224n_head.png"
tags:
    - Kaggle
    - Python
---

# 整理了一些kaggle比赛的一些tips

## stacking

* 该技巧就是将预测结果叠加在进行预测。

> 首先将训练集分为训练集和测试集```training,valid,ytraining,yvalid```,然后训练几个模型分别进行训练

```python
model1=RandomForestRegressor(random_state=2)
model2=LinearRegression()
#fit models
model1.fit(training,ytraining)
model2.fit(training,ytraining)
```

并进行预测

```python
preds1=model1.predict(valid)
preds2=model2.predict(valid)
#make predictions for test data
test_preds1=model1.predict(test)
test_preds2=model2.predict(test)
```

将预测的结果作为训练集，再进行训练

```python
stacked_predictions=np.column_stack((preds1,preds2))
meta_model=LinearRegression()
#fit meta model on stacked predictions
meta_model.fit(stacked_predictions,yvalid)
```

最后在测试集上得到结果

```python
stacked_test_predictions=np.column_stack((test_preds1,test_preds2))
final_predictions=meta_model.predict(stacked_test_predictions)
```


## bagging
* 该技巧就是将每个阶段的不同预测平均起来

```python
def bagging(train , y, test,bags=10,seed=1 ,model=RandomForestRegressor()):
   
   bagged_prediction=np.zeros(test.shape[0])
   for n in range (0, bags):
        model.set_params(random_state=seed + n)# update seed 
        model.fit(train,y) # fit model
        preds=model.predict(test) # predict on test data
        bagged_prediction+=preds # add predictions to bagged predictions 
    # 求平均
   bagged_prediction/= bags   
   return bagged_prediction
```




