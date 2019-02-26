---
layout:     post
title:      "swift_tutorail"
subtitle:   "swift"
date:       2018-02-23
author:     "hadxu"
header-img: "img/in-post/cs224n/cs224n_head.png"
tags:
    - iOS
    - swift
---

# swift tutorial

> 突然发现iOS开发非常优美，加之对机器学习非常感兴趣，而CoreML的出现，非常让人激动。

一个非常好的资料就是[swift](https://developer.apple.com/library/content/documentation/Swift/Conceptual/Swift_Programming_Language/index.html#//apple_ref/doc/uid/TP40014097-CH3-ID0)


```swift

var str:String? = "hello"

str = nil
str = "hey"

let constantString = "hello"

print(constantString)

let name="johnwerqq"

if name.count > 7 {
    print("long name")
}

switch name.count {
case 7...10:
    print("Long")
case 5..<7:
    print("Medium")
default:
    print("some length")
}

var number = 0
while number < 10 {
    number * number
    number += 1
}

for number in 0...10 {
    number
}

for number in [2,5,1,9,6] {
    number
}

let animal:[String] = ["Cow", "Dog", "Bunny"]

animal[2]

var cutness = ["Cow":"Not very", "Dog":"Cute", "Bunny":"Very cute"]

cutness["Dog"]

for n in animal {
    cutness[n]
}

func doMath(on a:Double, and b:Double, op:String) -> Double{
    print("Performathin",op,"on",a,"and",b)
    var result:Double = 0
    switch op {
    case "+": result = a + b
    case "-": result = a - b
    case "*": result = a * b
    case "/": result = a / b
    default:
        print("bad op",op)
    }
    return result
}

let res = doMath(on:2.0, and:1.0, op:"-")


var image = [
    [3,7,10],
    [6,4,3],
    [8,6,5]
]

func myfun(image:inout [[Int]]) {
    for row in 0..<image.count {
        for col in 0..<image[row].count {
            image[row][col]
            if (image[row][col] < 5) {
                image[row][col] = 5
            }
        }
        
    }
}

myfun(image: &image)

image


var str:String? = "hi"

str!.count

func fun(spell:String) -> String {
    return spell
}

var f = {
    (spell:String) -> String in
    return spell
}


f("hello")


struct Animal {
    var name:String = ""
    var heightInches = 0.0
    var heightCM:Double {
        get{
            return 2.54*heightInches
        }
        set(newHeightCM){
                heightInches = newHeightCM / 2.54
        }
    }
}

var dog = Animal(name:"dog",heightInches:50)
dog.heightCM
dog.heightInches

dog.heightCM = 254

dog.heightInches



class SuperNumber: NSNumber {
    override func getValue(_ value: UnsafeMutableRawPointer) {
        super.getValue(value)
    }
}

extension NSNumber {
    func superCoolGetter() -> Int {
        return 5
    }
}

let n = NSNumber(value:4)
n.superCoolGetter()

protocol dancable {
    func dance()
}

class Person: dancable {
    func dance() {
        print("hello")
    }
}

let t = Person()
t.dance()


enum TypeOfVeggies:String {
    case Carrots
    case Tomatoes
    case Celery
}

func eatVeggies(veggies: TypeOfVeggies) {
    
}

eatVeggies(veggies: TypeOfVeggies.Carrots)

let image = UIImage(named: "sample.png")
```

# iOS tutorial

