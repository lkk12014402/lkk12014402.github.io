---
layout:     post
title:      "StackOverflow高分问答"
subtitle:   "Python"
date:       2017-04-21
author:     "hadxu"
header-img: "img/hadxu.jpg"
tags:
    - Python
---

# Python备忘录 持续更新中

----------
> 这些都是stackoverflow上的高vote答案。

1. 如何合并两个字典，使之成为一个字典

    	x = {'a':1, 'b': 2}
    	y = {'b':10, 'c': 11}
		z = {**x, **y}
		>>> z
		{'a': 1, 'b': 10, 'c': 11}

2. 字节转json

		r = requests.get('http://localhost/index.json')
		json.loads(r.content.decode('utf-8'))

3. 以值来排序字典

		import operator
		x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
		sorted_x = sorted(x.items(), key=operator.itemgetter(1))

4. 判断一个文件夹是否存在，如果不存在就创建新文件夹

		if not os.path.exists(directory):
    		os.makedirs(directory)

5. 列举数组并打印出下标

		for idx, val in enumerate(ints):
    		print(idx, val)

6. 在Python中，支持多级比较

		>>> 1<2<3
		>>> True
		>>> 1<5<2
		>>> False

7. 找到一个列表中某个元素的下标

		>>> ["foo", "bar", "baz"].index("bar")
		>>> 1

8. 判断一个key或value是否在一个字典中

		if 'key1' in dict:
		  print "blah"
		else:
		  print "boo"
		
		如果是value的话，是这样的：
		if 'value1' in dict.values():
		  print "blah"
		else:
		  print "boo"

9. 获取当前的时间

		import datetime
		datetime.datetime.now()
		>>>datetime.datetime(2017, 2, 15, 11, 22, 46, 801132)
		>>>datetime.now().strftime('%Y-%m-%d %H:%M:%S')
10. 字典按值排序

		import operatoe		
		x = [{'name':'Homer', 'age':39}, {'name':'Bart', 'age':10}] 
		sorted(x,key=operator.itemgetter('age'))
		>>>[{'age': 10, 'name': 'Bart'}, {'age': 39, 'name': 'Homer'}]
		
11. 在Python中使用switch关键字

	在Python中是没有switch关键字的，但是如果我想实现这样的switch语句怎么办？可以这样：

		def f(x):
    		return {
		        'a': 1,
		        'b': 2,
		    }[x]		
12. 找出文件夹下所有以`.txt`结尾的文件：

		import os
		for file in os.listdir("/mydir"):
	    if file.endswith(".txt"):
	        print(file)
13. 将tuple转化为字典

		x = [(1,2),(3,4)]
		d = {key: value for (key, value) in x}
		>>>d
		>>>{1: 2, 3: 4}
14. 随机生成验证码：

		''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))
15. 统计一个列表中某个元素出现的次数

		[1, 2, 3, 4, 1, 4, 1].count(1)
		>>>3
		或者这样：
		from collections import Counter
		Counter(x)
16. 如果碰到类似于这样的错误：
		`UnicodeEncodeError: 'ascii' codec can't encode character u'\xa0' in position 20: ordinal not in range(128)`
		
	可以这样解决：

		u' '.join((agent_contact, agent_telno)).encode('utf-8').strip()

17. 将两个列表合并成一个字典：

		keys = ('name', 'age', 'food')
		values = ('Monty', 42, 'spam')

	可以这样：
		
		dictionary = dict(zip(keys, values))

18. 如果有一个字典格式被包含在一个字符串里面，就像这样`s = "{'muffin' : 'lolz', 'foo' : 'kitty'}"`，如何解析出来里面的字典呢？
		>>>import ast
		>>>ast.literal_eval(s)
		>>> {'foo': 'kitty', 'muffin': 'lolz'}
19. 有没有更加Pythonic的方法来合并两个字典，如果key相等就value相加？

		>>> from collections import Counter
		>>> A = Counter({'a':1, 'b':2, 'c':3})
		>>> B = Counter({'b':3, 'c':4, 'd':5})
		>>> A + B
		Counter({'c': 7, 'b': 5, 'd': 5, 'a': 1})
20. 如何得到一个程序的运行时间？

		import time
		start_time = time.time()
		main()
		print("--- %s seconds ---" % (time.time() - start_time))
21. 如何得到一个字典最大值的item？

		import operator
		stats = {'a':1000, 'b':3000, 'c': 100}
		max(stats.iteritems(), key=operator.itemgetter(1))
		>>>{'b':3000}
22. 翻转字典里面的key和value

		inv_map = {v: k for k, v in my_map.items()}


23. 如果爬取网页或者读二进制文件的时候，发现取出来的为16进制的那种汉字，则需要进行下面的解答：

		content = f.read() #取出来的为二进制
	    print(content.decode('utf-8')) #可以在电脑上显示


----------
##下面的是我在实习的时候碰见的问题
1. 在Python中，都知道调用C库使用cffi，但是如果出现库之间的依赖的问题，可以这样解决：
		
		from ctypes import CDLL, RTLD_GLOBLE
		from cffi import FFI
		CDLL('***.so',mode=RTLD_GLOBLE)
		CDLL('******.so', mode=RTLD_GLOBLE) #有多少这样的依赖，就加载进来
		ffi = FFI()
		ffi.cdef('各种说明，方法的定义')
		lib = ffi.dlopen('******.so') #这里的动态链接库就是我们要使用的库
		lib.xxxxxx() #调用库里的方法

2. 在Python中，我已经得到了类似于这样的数据`data = [{'id': 123, 'name': 'hadxu'}, {'id': 345, 'name': 'qwert'}]`，要将他转换为excel,csv那样的格式，可以这样做：

		import pandas as pd
		df = pd.DataFrame(data)
		df.to_csv('xx.csv', sep=',')
在这里有个一注意点，那就是，如果里面有中文，如果想用excel来打开，就不要加`encoding='utf-8'`,如果不想用excel打开，用sublime来打开的话，可以加`urf-8`，当然了，可以这样用`utf-8`编码，在windows上用记事本打开，另存为设置编码为ANSI编码，这样windows也是可以看见的。不要问我怎么知道，为了对付组长的不懂因为，必须要中文，才这么干的。

3. 在实习过程中，在数据库中有很多检测项，实际上通过博士的算法得到很多检测后的值，但是对于客户来说不要值，只需要知道通过或者不通过，于是可以这样写：

		da = {'face': 10, 'eye': 30, 'noise': 40, 'bk': 60, 'glasses': 99}
    	check_data = {'face': 12, 'eye': 13, 'noise': 14, 'bk': 30, 'glasses': 60}
    	res = {k: '通过' if da.get(k) > check_data.get(k) else '不通过' for k in da.keys()}
字典da的就是标准值，字典check_data的是检测得到的值，如果check_data的值大于da对应的值，就是不通过。
