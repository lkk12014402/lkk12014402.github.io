---
layout:     post
title:      "RSA背后原理以及Python实现"
subtitle:   "Cryptography"
date:       2018-04-01
author:     "lkk"
header-img: ""
tags:
    - Cryptography
    - python
---


# RSA加密算法背后的数学原理

### Prime Numbers

一个素数定义为大于1的因子只能够是它本身。比如2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281

有无穷无尽这样的素数，也就意味着没有最大的素数。**没有最大，只有更大**。
RSA算法就是基于大素数而提出的一种算法。

#### 产生素数的方法很多，其中比较著名的为筛选法。

```python
import math

def isPrime(num):
	if num<2:
		return False

	for i in range(2,int(math.sqrt(num))+1):
		if num%i == 0:
			return False
	return True

def primeSieve(sieveSize):
	sieve = [True]*sieveSize
	sieve[0] = False
	sieve[1] = False
    # 重点在这
	for i in range(2,int(math.sqrt(sieveSize))+1):
		pointer = i * 2
		while pointer < sieveSize:
			sieve[pointer] = False
			pointer += i

	primes = []
	for i in range(sieveSize):
		if sieve[i] == True:
			primes.append(i)

	return primes
```

### 检测一个数是否为素数

```isPrime()```函数是用来检测一个数是否为素数，但是如果一个数特别特别大，使用该函数是非常非常慢的。

**Miller Rabin**是目前主流的测试一个数是否为素数的算法

[Miller–Rabin](https://en.wikipedia.org/wiki/Miller–Rabin_primality_test
)

```C++
write n − 1 as 2r·d with d odd by factoring powers of 2 from n − 1
WitnessLoop: repeat k times:
   pick a random integer a in the range [2, n − 2]
   x ← ad mod n
   if x = 1 or x = n − 1 then
      continue WitnessLoop
   repeat r − 1 times:
      x ← x2 mod n
      if x = 1 then
         return composite
      if x = n − 1 then
         continue WitnessLoop
   return composite
return probably prime
```


使用Python语言如下：

```python
import random

def rabinMiller(num):
	s = num - 1
	t = 0
	while s%2 == 0:
		s //= 2
		t += 1
	for trials in range(5):
		a = random.randrange(2,num-1)
		v = pow(a,s,num)
		if v!=1:
			i = 0
			while v!=(num-1):
				if i == t-1:
					return False
				else:
					i += 1
					v = (v**2)%num
	return True

def isPrime(num):
	if num<2:
		return False
	lowPrimes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]

	if num in lowPrimes:
		return True
	for prime in lowPrimes:
		if num % prime==0:
			return False

	return rabinMiller(num)

def generateLargePrime(keysize=1024):
	while True:
		num = random.randrange(2**(keysize-1),2**keysize)

		if isPrime(num):
			return num

if __name__ == '__main__':
	# res = generateLargePrime()
	# print(res)
	num = 221
	res = rabinMiller(num)
	print(res)
```

通过这段代码就可以求得我们要求的素数。


## RSA算法

> RSA算法就是基于大素数难以分解而得到的一种算法，通过公钥和私钥来进行对数据的加密解密。

* The public key is used for encrypting.
* The private key is used for decrypting.

1. 选择两个大素数p,q。
2. 计算```n=p*q```
3. 计算```s=lcm(p-1,q-1)```,即p-1和q-1的最小公倍数
4. 选择```1<e<s```,且```gcd(e,s)=1```
5. 计算e的模逆元```e*x % s = 1```
6. 则(n,e)为公钥，(n,d)为私钥

具体介绍为[RSA](https://en.wikipedia.org/wiki/RSA_(cryptosystem))


用Python来实现的话，则这样

```python
def generateRSAkey(keySize=1024):
	p = rabinMiller.generateLargePrime(keySize)
	q = rabinMiller.generateLargePrime(keySize)
	n = p*q

	while True:
		e = random.randrange(2 ** (keySize - 1), 2 ** (keySize))
		if cryptomath.gcd(e, (p - 1) * (q - 1)) == 1:
			break
	d = cryptomath.findModInverse(e, (p - 1) * (q - 1))

	publicKey = (n, e)
	privateKey = (n, d)
```

使用方式如下：

```python

publicKey,privateKey = generateRSAkey(1024)

# 我们需要加密的文本信息
text = 3

en = pow(text,publicKey[1],publicKey[0])

print(en)

de = pow(en, privateKey[1],privateKey[0])

print(de)

>>> 3
```

具体代码在[Cryptography](https://github.com/HadXu/Cryptography/tree/master/code)









