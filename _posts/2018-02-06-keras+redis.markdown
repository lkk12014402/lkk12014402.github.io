---
layout:     post
title:      "Keras+Redis+Flask"
subtitle:   "Keras"
date:       2018-02-06
author:     "hadxu"
header-img: "img/in-post/cs224n/cs224n_head.png"
tags:
    - Redis
    - Python
    - keras
    - Flask
---

# Keras+Redis可部署的深度学习应用

学习深度学习，算法是非常重要的，但是更重要的就是将算法运用起来，为我们生活服务。前几天逛```Twitter```，发现了一篇文章关于深度学习应用的部署，非常感兴趣，其实早在15年我就已经做过类似的项目，但是那个项目非常简单，只支持简单的识别，这篇文章讲解了高并发的深度学习架构。我们来看看吧。

### 安装需要的环境

* Flask
* Redis
* Keras

安装Redis

```
$ wget http://download.redis.io/redis-stable.tar.gz
$ tar xvzf redis-stable.tar.gz
$ cd redis-stable
$ make
$ sudo make install
```

启动Redis

```
$ redis-server
```

### 代码编写

* 定义常量

```python
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"


IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 32
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25
```

> 表示图片都是```224*224*3,float类型```,同时指定batch_size以及停顿时间。

* 建立Flask以及数据库连接

```python
app = flask.Flask(__name__)
db = redis.StrictRedis(host="localhost", port=6379, db=0)
model = None
```

* 处理图片

```python
def base64_encode_image(a):
	# base64 encode the input NumPy array
	return base64.b64encode(a).decode("utf-8")

def base64_decode_image(a, dtype, shape):
	# if this is Python 3, we need the extra step of encoding the
	# serialized NumPy string as a byte object
	if sys.version_info.major == 3:
		a = bytes(a, encoding="utf-8")
	a = np.frombuffer(base64.decodestring(a), dtype=dtype)
	a = a.reshape(shape)
	return a

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")
 
	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)
 
	# return the processed image
	return image
```

* 读取模型

```python
print("* Loading model...")
model = ResNet50(weights="imagenet")
print("* Model loaded")
```

* 开启预测

```python
while True:
    # attempt to grab a batch of images from the database, then
    # initialize the image IDs and batch of images themselves
    queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
    imageIDs = []
    batch = None
```

> 从队列中读取BATCH_SIZE个数据

* 读取数据

```python
for q in queue:
    # deserialize the object and obtain the input image
    q = json.loads(q.decode("utf-8"))
    image = base64_decode_image(q["image"], IMAGE_DTYPE,
        (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANS))
    if batch is None:
        batch = image

    # otherwise, stack the data
    else:
        batch = np.vstack([batch, image])
    imageIDs.append(q["id"])
```

* 预测

```python
if len(imageIDs)>0:
    print("* Batch size: {}".format(batch.shape))
    preds = model.predict(batch)
    results = imagenet_utils.decode_predictions(preds)
    for (imageID, resultSet) in zip(imageIDs, results):
        # initialize the list of output predictions
        output = []

        # loop over the results and add them to the list of
        # output predictions
        for (imagenetID, label, prob) in resultSet:
            r = {"label": label, "probability": float(prob)}
            output.append(r)
    db.set(imageID, json.dumps(output))
db.ltrim(IMAGE_QUEUE, len(imageIDs), -1)
```

> 先提取一部分数据，然后预测，将结果放到output中，接下来从队列中删除已经预测过的。

* 从网络读取数据

```python
@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image = image.copy(order="C")
            k = str(uuid.uuid4())
            d = {"id": k, "image": base64_encode_image(image)}
            db.rpush(IMAGE_QUEUE, json.dumps(d))
            while True:
                output = db.get(k)
                if output is not None:
                    output = output.decode("utf-8")
                    data["predictions"] = json.loads(output)
                    db.delete(k)
                    break
                time.sleep(CLIENT_SLEEP)
        data["success"] = True
    return flask.jsonify(data)
```

> 首先读取图片信息，读取进来转换为PIL类型，并处理，同时从队列中读取处理完的数据。

* 运行

```python
if __name__ == "__main__":
	print("* Starting model service...")
	t = Thread(target=classify_process, args=()) # 开启线程
	t.daemon = True
	t.start()
	print("* Starting web service...")
	app.run()
```

最终的效果如图:

```
curl -X POST -F image=@jemma.png 'http://localhost:5000/predict'
```

![](/img/in-post/flask-server.jpg)

该系统还是可以的。地址[flask+redis+keras](https://github.com/HadXu/machine-learning/tree/master/flask_imageNet/keras%20restful%20API)

还有一篇文章，还没有看完[Deep learning in production with Keras, Redis, Flask, and Apache](https://www.pyimagesearch.com/2018/02/05/deep-learning-production-keras-redis-flask-apache/)




