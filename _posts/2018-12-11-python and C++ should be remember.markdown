---
layout:     post
title:      "Python/C++ trick"
subtitle:   "比赛"
date:       2018-12-11
author:     "hadxu"
header-img: "img/hadxu.jpg"
tags:
    - Python
---

# Python trick


1. Read-only Dict

```Python
>>> from types import MappingProxyType
>>> x = {'one':1,'two':2}
>>> x = MappingProxyType(x)
>>> x
mappingproxy({'one': 1, 'two': 2})
>>> x['one']
1
>>> x['one'] = 4
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'mappingproxy' object does not support item assignment
```

2. define a tuple

```python
>>> x = 2,3,4
>>> x
(2, 3, 4)
>>> x += (5,)
>>> x
(2, 3, 4, 5)
```

3. array.array Basic Typed Arrays

```python
>>> import array
>>> arr = array.array('f',[2,4,7,8,9])
>>> arr
array('f', [2.0, 4.0, 7.0, 8.0, 9.0])
>>> arr[1]
4.0
```

4. namedtuple

```python
>>> from collections import namedtuple
>>> x = namedtuple('point','x y z')(1,2,3)
>>> x
point(x=1, y=2, z=3)
```

5. frozenset

```python
>>> x = frozenset({1,2,3})
>>> x
frozenset({1, 2, 3})
>>> x.add(4)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'frozenset' object has no attribute 'add'
```

6. mkdir tempfile

```python
@contextmanager
def _tempdir():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


with _tempdir() as temp_dir:
    fname = os.path.join(temp_dir, 'a.txt')
    source = 'HIIIII'
    with open(fname, 'w') as f:
        f.write(source)

    with open(fname, 'r') as f:
        text = f.read()
        print(text)
```


7. Numpy Memory layout

**An instance of class ndarray consists of a contiguous one-dimensional segment of computer memory (owned by the array, or by some other object), combined with an indexing scheme that maps N integers into the location of an item in the block.**

也就是说Numpy的存储本质上是一维的，连续的地址空间。那么如何知道维度呢？ **Shape & dtype**

8. Numpy Maze(迷宫求解)

```python
# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from matplotlib.animation import FuncAnimation


def build_maze(shape=(65,65), complexity=0.75, density = 0.50):
    """
    Build a maze using given complexity and density

    Parameters
    ==========

    shape : (rows,cols)
      Size of the maze

    complexity: float
      Mean length of islands (as a ratio of maze size)

    density: float
      Mean numbers of highland (as a ratio of maze surface)

    """
    
    # Only odd shapes
    shape = ((shape[0]//2)*2+1, (shape[1]//2)*2+1)

    # Adjust complexity and density relatively to maze size
    n_complexity = int(complexity*(shape[0]+shape[1]))
    n_density    = int(density*(shape[0]*shape[1]))

    # Build actual maze
    Z = np.zeros(shape, dtype=bool)

    # Fill borders
    Z[0,:] = Z[-1,:] = Z[:,0] = Z[:,-1] = 1

    # Islands starting point with a bias in favor of border
    P = np.random.normal(0, 0.5, (n_density,2))
    P = 0.5 - np.maximum(-0.5, np.minimum(P, +0.5))
    P = (P*[shape[1],shape[0]]).astype(int)
    P = 2*(P//2)
    
    # Create islands
    for i in range(n_density):

        # Test for early stop: if all starting point are busy, this means we
        # won't be able to connect any island, so we stop.
        T = Z[2:-2:2,2:-2:2]
        if T.sum() == T.size:
            break

        x, y = P[i]
        Z[y,x] = 1
        for j in range(n_complexity):
            neighbours = []
            if x > 1:
                neighbours.append([(y, x-1), (y, x-2)])
            if x < shape[1]-2:
                neighbours.append([(y, x+1), (y, x+2)])
            if y > 1:
                neighbours.append([(y-1, x), (y-2, x)])
            if y < shape[0]-2:
                neighbours.append([(y+1, x), (y+2, x)])
            if len(neighbours):
                choice = np.random.randint(len(neighbours))
                next_1, next_2 = neighbours[choice]
                if Z[next_2] == 0:
                    Z[next_1] = Z[next_2] = 1
                    y, x = next_2
            else:
                break
    return Z



def build_graph(maze):
    height, width = maze.shape
    # 获取每一个非墙点的坐标
    graph = {(i, j): [] for j in range(width)
                        for i in range(height) if not maze[i][j]}

    # 记录每一个非墙点上下左右的坐标以及方向
    for row, col in graph.keys():
        if row < height - 1 and not maze[row + 1][col]:
            graph[(row, col)].append(("S", (row + 1, col)))
            graph[(row + 1, col)].append(("N", (row, col)))
        if col < width - 1 and not maze[row][col + 1]:
            graph[(row, col)].append(("E", (row, col + 1)))
            graph[(row, col + 1)].append(("W", (row, col)))
    return graph


def breadth_first(maze, start, goal):
    queue = deque([([start], start)])
    visited = set()
    graph = build_graph(maze)
    while queue:
        path, current = queue.popleft()
        if current == goal:
            return np.array(path)
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            p = list(path)
            p.append(neighbour)
            queue.append((p, neighbour))
    return None

def BellmanFord(Z, start, goal):
    Z = 1 - Z
    G = np.zeros(Z.shape)
    G[start] = 1
    gamma = 0.99
    def diffuse(Z):
        return max(gamma*Z[0], gamma*Z[1], Z[2], gamma*Z[3], gamma*Z[4])

    length = Z.shape[0]+Z.shape[1]
    G_gamma = np.empty_like(G)
    while G[goal] == 0.0:
        np.multiply(G, gamma, out=G_gamma)
        N = G_gamma[0:-2,1:-1]
        W = G_gamma[1:-1,0:-2]
        C = G[1:-1,1:-1]
        E = G_gamma[1:-1,2:]
        S = G_gamma[2:,1:-1]
        G[1:-1,1:-1] = Z[1:-1,1:-1]*np.maximum(N,np.maximum(W,
                                np.maximum(C,np.maximum(E,S))))

    y, x = goal
    P = []
    dirs = [(0,-1), (0,+1), (-1,0), (+1,0)]
    while (x, y) != start:
        P.append((x, y))
        neighbours = [-1, -1, -1, -1]
        if x > 0:
            neighbours[0] = G[y, x-1]
        if x < G.shape[1]-1:
            neighbours[1] = G[y, x+1]
        if y > 0:
            neighbours[2] = G[y-1, x]
        if y < G.shape[0]-1:
            neighbours[3] = G[y+1, x]
        a = np.argmax(neighbours)
        x, y  = x + dirs[a][1], y + dirs[a][0]
    P.append((x, y))
    return G, np.array(P)


if __name__ == '__main__':
    Z = build_maze((41,81))
    Z = 1 - Z

    start,goal = (1, 1), (Z.shape[0]-2, Z.shape[1]-2)
    G, P = BellmanFord(Z, start, goal)
    X, Y = P[:,0], P[:,1]
    plt.figure(figsize=(13, 13*Z.shape[0]/Z.shape[1]))
    ax = plt.subplot(1, 1, 1, frameon=False)
    ax.imshow(Z, interpolation='nearest', cmap=plt.cm.gray_r, vmin=0.0, vmax=1.0)
    cmap = plt.cm.hot
    cmap.set_under(color='k', alpha=0.0)
    ax.imshow(G, interpolation='nearest', cmap=cmap, vmin=0.01, vmax=G[start])
    ax.scatter(X[1:-1], Y[1:-1], s=60,
               lw=1, marker='o', edgecolors='k', facecolors='w')
    ax.scatter(X[[0,-1]], Y[[0,-1]], s=60,
               lw=3, marker='x', color=['w','k'])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    # plt.savefig("maze.png")
    plt.show()
```

# C++

1. constexpr 是在编译的时候就已经得到值不改变，const则是在运行的时候得到值不改变

```cpp
const int y = 7;
int main()
{
	constexpr int x = y*y;
	cout<<"h"<<x<<endl;
	return 0;
}
```

2. 构造函数

```cpp
class Vector{
public:
	Vector(int s):elem{new double[s]},sz{s}{}
	double& operator[](int i){return elem[i];}
	int size(){return sz;}
private:
	double* elem;
	int sz;
};
```

3. **struct是一个public成员的class。**

4. enum

```cpp
enum class Color {red,blue,green};
Color col = Color::blue;
```

5. 编译时 **assert**

```cpp
static_assert(sizeof(int)>0,"two small");
```
6. class

```cpp
class complex{
	double re,im;
public:
	complex(double r,double i):re{r},im{i}{}
	complex(double r):re{r},im{0}{}
	complex():re{0},im{0}{}
	double real() const {return re;}

	complex& operator+=(complex z){re +=z.re,im += z.im;return *this;}
	complex& operator-=(complex z){re -=z.re,im -= z.im;return *this;}
};
```
7. template

```cpp
template <typename T>
int compare(const T &v1,const T &v2){
	if (v1 <v2)
		return -1;
	if (v2 < v1)
		return 1;
	return 0;
}
int main()
{
    cout << compare(1,2) << endl;
    return 0;
}
```

8. Standard-Library Components
需要掌握以下库
```
Runtime
The C Standard Library
Strings
regular
I/O
Container (STL)
numerical computation
concurrent
template meraprogramming
smart points
special containers
```
