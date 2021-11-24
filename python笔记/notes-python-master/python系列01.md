

## 待补充模块

### python基础部分

### numpy中的结构化数据

### numpy中的记录数组

### python进阶中的正则表达式和re模块

### python操作数据库模块

### python修饰符的使用模块

### python的作用域

### operator, functools, itertools, toolz, fn, funcy 模块



## 01. **Python 工具**

### 01.01 Python 简介

### 01.02 Ipython 解释器

### 01.03 Ipython notebook

### 01.04 使用 Anaconda



## [02. **Python 基础**]

### 02.01 Python 入门演示

### 02.02 Python 数据类型

### 02.03 数字

因存储而造成的数值计算误差

### 02.04 字符串

### 02.05 索引和分片

### 02.06 列表

### 02.07 可变和不可变类型

bytearray问题

### 02.08 元组

### 02.09 列表与元组的速度比较

### 02.10 字典

### 02.11 集合

### 02.12 不可变集合

### 02.13 Python 赋值机制

### 02.14 判断语句

### 02.15 循环

### 02.16 列表推导式

### 02.17 函数

### 02.18 模块和包

### 02.19 异常

### 02.20 警告

### 02.21 文件读写

读文件和写文件





## 03 numpy

### 03.01 numpy概述

**Numpy**是**Python**的一个很重要的第三方库，很多其他科学计算的第三方库都是以**Numpy**为基础建立的。

**Numpy**的一个重要特性是它的数组计算。

### 03.02 numpy数组及其索引

#### 数组生成及其属性

**一维数组**

```python
# 方法一，先生成列表，然后再转换
lst = [0, 1, 2, 3]
a = array(lst)
# 方法二，直接传入列表
a = array([1,2,3,4])
```

**数组属性**

```python
#查看类型
type(a)		#numpy.ndarray
#查看数组中的数据类型
a.dtype		#dtype('int32')
#查看每个元素所占字节
a.itemsize		#4
# 查看形状，会返回一个元组，每个元素代表这一维的元素数目
a.shape		#(4,)
shape(a)	#(4,)
#查看元素个数
a.size	#4
size(a)
#查看维数
a.ndim	#1
```

**使用fill方法设定初始值**

```python
a.fill(-4.8) 	#array([-4, -4, -4, -4])
# 这里之所以不是4.8，是因为数组中要求元素的dtype是一样的，如果传入参数的类型与数组不一样，就会按已有的类型进行转换
```

**多维数组**

```python
# 方法一
a = array([[ 0, 1, 2, 3],
           [10,11,12,13]])
# 方法二
a = arange(25)
a.shape = 5,5
#输出a为：
#array([[ 0,  1,  2,  3,  4],
#       [ 5,  6,  7,  8,  9],
#       [10, 11, 12, 13, 14],
#       [15, 16, 17, 18, 19],
#       [20, 21, 22, 23, 24]])
```

#### 索引和切片

**一维数组**

```python
a = array([10, 11, 12, 13])
a[0]	#10
a[0]=14	#array([14, 11, 12, 13])
a[-3:3]	#array([11,12, 13])
a[::2]	#array([10,12])
a[-2:]	##array([12, 13])
```

**多维数组**

多维数组也类似，只用把第一个元素看成最外一层中括号的索引，或最后一个元素看成最里面括号的索引，然后以此推就可以。不同维元素用逗号隔开

```python
#索引
a = array([[ 0, 1, 2, 3],
           [10,11,12,13]])
a[1, 3]	#13
a[1]	#array([10, 11, 12, -1])
#切片
a = array([[ 0, 1, 2, 3, 4, 5],
           [10,11,12,13,14,15],
           [20,21,22,23,24,25],
           [30,31,32,33,34,35],
           [40,41,42,43,44,45],
           [50,51,52,53,54,55]])
a[0, 3:5]	#array([3, 4])
a[4:, 4:]	#array([[44, 45],
       		#		[54, 55]])
a[2::2, ::2]	#array([[20, 22, 24],
       			#		[40, 42, 44]])
```



> **切片是引用，也就是指向原来数组所分配的空间，如果对切片的子数组修改，原来的数组也会修改。**
>
> **但在列表中就不会是引用这种情况。**
>
> **为避免引用，可以使用copy()方法来直接复制一个**
>
> ```python
> a = array([0,1,2,3,4])
> b = a[2:4].copy()
> b[0] = 10	
> a	#array([0, 1, 2, 3, 4])
> ```
>
> **花式索引的返回值是原对象的一个复制而不是引用**



#### 花式索引

**切片只能支持连续或者等间隔的切片操作，要想实现任意位置的操作，需要使用花式索引**

**一维花式索引**

```python
a = arange(0, 80, 10)	#array([ 0, 10, 20, 30, 40, 50, 60, 70])
#方法一：给定索引列表
indices = [1, 2, -3]
y = a[indices]		#array([10, 20, 50])
#方法二：使用布尔数组
mask = array([0,1,1,0,0,1,0,0],
            dtype=bool)
a[mask]		#array([10, 20, 50])
#方法三：布尔表达式
mask = a > 50
a[mask]			#array([ 60, 70])
```

> mask必须是布尔数组

**二维花式索引**

对于二维花式索引，我们需要给定 `row` 和 `col` 的值：

```python
a = array([[ 0, 1, 2, 3, 4, 5],
           [10,11,12,13,14,15],
           [20,21,22,23,24,25],
           [30,31,32,33,34,35],
           [40,41,42,43,44,45],
           [50,51,52,53,54,55]])
#方法一：常规操作
a[(0,1,2,3,4), (1,2,3,4,5)]		#array([ 1, 12, 23, 34, 45])
a[3:, [0,2,5]]		#array([[30, 32, 35],
      				#		 [40, 42, 45],
    				#		 [50, 52, 55]])
#方法二：用mask
mask = array([1,0,1,0,0,1],
            dtype=bool)
a[mask, 2]		#array([ 2, 22, 52])
```



#### where语句的使用

`where` 函数会返回所有非零元素的索引。

**一维情况**

```python
a = array([0, 12, 5, 20])
a > 10		#array([False,  True, False,  True])
where(a > 10)	#(array([1, 3], dtype=int64),)
# 注意where的返回值是一个元组，因为where可以对多维数组使用

loc = where(a > 10)
a[loc]		#array([12, 20])
```

**多维数组**

```python
a = array([[0, 12, 5, 20],
           [1, 2, 11, 15]])
loc = where(a > 10)	#(array([0, 0, 1, 1], dtype=int64), array([1, 3, 2, 3], dtype=int64))
a[loc]		#array([12, 20, 11, 15])
rows, cols = where(a>10)	
rows	#array([0, 0, 1, 1], dtype=int64)
cols	#array([1, 3, 2, 3], dtype=int64)
```

### 03.03 numpy数组类型

之前已经看过整数数组和布尔数组，除此之外还有浮点数数组和复数数组。

#### 复数数组

```python
a = array([1 + 1j, 2, 3, 4])
a.dtype		#dtype('complex128')
a.real		#array([ 1.,  2.,  3.,  4.])
a.imag		#array([ 1.,  0.,  0.,  0.])
a.imag = [1,2,3,4]
a		#array([ 1.+1.j,  2.+2.j,  3.+3.j,  4.+4.j])
#查看复共轭
a.conj()	#array([ 1.-1.j,  2.-2.j,  3.-3.j,  4.-4.j])
```

事实上，上面这些属性方法（a.real，a.imag，a.conj()）可以用在浮点数或者整数数组上，但这样虚部是只读的，并不能修改它的值。

#### 指定数组类型

```python
#构建数组的时候，数组会根据传入的内容自动判断类型
#对于浮点数，默认为双精度,可以在构建的时候指定类型
a = array([0,1.0,2,3],
         dtype=float32)
```

#### numpy类型

|   基本类型 |                            可用的**Numpy**类型 |                                                    备注 |
| ---------: | ---------------------------------------------: | ------------------------------------------------------: |
|     布尔型 |                                         `bool` |                                               占1个字节 |
|       整型 |       `int8, int16, int32, int64, int128, int` |                     `int` 跟**C**语言中的 `long` 一样大 |
| 无符号整型 | `uint8, uint16, uint32, uint64, uint128, uint` |           `uint` 跟**C**语言中的 `unsigned long` 一样大 |
|     浮点数 |  `float16, float32, float64, float, longfloat` | 默认为双精度 `float64` ，`longfloat` 精度大小与系统有关 |
|       复数 |  `complex64, complex128, complex, longcomplex` |              默认为 `complex128` ，即实部虚部都为双精度 |
|     字符串 |                              `string, unicode` |           可以使用 `dtype=S4` 表示一个4字节字符串的数组 |
|       对象 |                                       `object` |                                    数组中可以使用任意值 |
|    Records |                                         `void` |                                                         |

**任意类型的数组**

```python
a = array([1,1.2,'hello', [10,20,30]], 
          dtype=object)
a * 2	#array([2, 2.4, 'hellohello', [10, 20, 30, 10, 20, 30]], dtype=object)
```

#### 类型转换

**asarray函数**

```python
a = array([1.5, -3], 
         dtype=float32)
a	#array([ 1.5, -3. ], dtype=float32)
asarray(a, dtype=float64)	#array([ 1.5, -3. ])
asarray(a, dtype=uint8)		#array([  1, 253], dtype=uint8)
a		#array([ 1.5, -3. ], dtype=float32)
b = asarray(a, dtype=float32)	
b is a 	#True
```

>如上最后三行代码所示，`asarray` 不会修改原来数组的值。但当类型相同的时候，`asarray` 并不会产生新的对象，而是使用同一个引用。
>
>这么做的好处在与，`asarray` 不仅可以作用于数组，还可以将其他类型转化为数组。
>
>有些时候为了保证我们的输入值是数组，我们需要将其使用 `asarray` 转化，当它已经是数组的时候，并不会产生新的对象，这样保证了效率。

**astype 方法**

`astype` 方法返回一个新数组，也不会改变原来数组的值

```python
a.astype(float64)	#array([ 1.5, -3. ])
a.astype(uint8)		#array([  1, 253], dtype=uint8)
a		#array([ 1.5, -3. ], dtype=float32)
```

> `astype` 总是返回原来数组的一份复制，即使转换的类型是相同的

**view方法**

`view` 会将 `a` 在内存中的表示看成是 `uint8` 进行解析

```python
a = array((1,2,3,4), dtype=int32)
a	#array([1, 2, 3, 4])
b = a.view(uint8)	
b	#array([1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0], dtype=uint8)
#修改 a 会修改 b 的值，因为共用一块内存
a[0] = 2**30
a	#array([1073741824,          2,          3,          4])
b	#array([ 0,  0,  0, 64,  2,  0,  0,  0,  3,  0,  0,  0,  4,  0,  0,  0],dtype=uint8)
```

### 03.04 数组方法

> 沿着某一维操作，可以这样理解：
>
> 就是将数组大小相应下标的维度变成1。例如a的大小为（2，3），沿着第一维求和（即a.sum(axis=0)），那就是先将a的大小变成（1，3），然后去除多余维度，变成（3，）

#### 求和

```python
a = array([[1,2,3], 
           [4,5,6]])
sum(a)		#21
sum(a, axis=0)		#array([5, 7, 9])
sum(a, axis=-1)		#array([ 6, 15])
a.sum()		#21
a.sum(axis=0)		#array([5, 7, 9])
```

#### 求积

```python
a.prod()	#720
prod(a, axis=0)		#array([ 4, 10, 18])
```

#### 求最值

```python
%pylab
from numpy.random import rand
a = rand(3, 4)
%precision 3  	#保留三位小数操作
a		#array([[ 0.444,  0.06 ,  0.668,  0.02 ],
      	#		 [ 0.793,  0.302,  0.81 ,  0.381],
        #   	 [ 0.296,  0.182,  0.345,  0.686]])
a.min()			#0.020
a.min(axis=0)	#array([ 0.296,  0.06 ,  0.345,  0.02 ])
a.max()			#0.810
a.max(axis=-1)	#array([ 0.668,  0.81 ,  0.686])
```

#### 最值的位置

使用 `argmin, argmax` 方法

```python
#全局最值的位置
a.argmin()			#3
#局部最值的位置
a.argmin(axis=0)	#array([2, 0, 2, 0], dtype=int64)
```

> 注意，**全局最值的位置，是一个标量**。不管原来数组是几维，他都是从第一行算起，然后第二行。。。

#### 均值

使用 `mean` 方法，可以使用 `mean` 函数，还可以使用 `average` 函数，`average` 函数还支持加权平均

```python
a = array([[1,2,3],[4,5,6]])
a.mean()	#3.5
a.mean(axis=-1)		#array([ 2.,  5.])
mean(a)		#3.5
average(a, axis = 0)	#array([ 2.5,  3.5,  4.5])
average(a, axis = 0, weights=[1,2])		#array([ 3.,  4.,  5.])
```

#### 标准差

用 `std` 方法计算标准差，用 `var` 方法计算方差

```python
a.std(axis=1)	#array([ 0.816,  0.816])
a.var(axis=1)	#array([ 0.667,  0.667])
# 或者使用函数
var(a, axis=1)
std(a, axis=1)
```

#### clip方法

将数值限制在某个范围

```python
a			#array([[1, 2, 3],
      		#		[4, 5, 6]])
a.clip(3,5)	#array([[3, 3, 3],
       		#		[4, 5, 5]])
```

#### ptp方法

计算最大值和最小值之差

```python
a.ptp(axis=1)	#array([2, 2])
a.ptp()			#5
```

#### round方法

近似，默认到整数

```python
a = array([1.35, 2.5, 1.5])
a.round()	#array([1., 2., 2.])
# 近似到一位小数
a.round(decimals=1)	#array([1.4, 2.5, 1.5])
```

> python中的.5近似规则是默认近似到偶数



### 03.05 数组排序

#### sort函数

`sort` 返回的结果是从小到大排列的。

```python
names = array(['bob', 'sue', 'jan', 'ad'])
weights = array([20.8, 93.2, 53.4, 61.8])

sort(weights)	#array([ 20.8,  53.4,  61.8,  93.2])
```

> sort方法与sort函数不一样，前者会改变数组的值：
>
> ```python
> data.sort()
> data	#array([ 20.8,  53.4,  61.8,  93.2])
> ```

#### argsort函数

`argsort` 返回从小到大的排列在数组中的索引位置

```python
ordered_indices = argsort(weights)
ordered_indices		#array([0, 2, 3, 1], dtype=int64)
# 可以直接用它来进行索引
weights[ordered_indices]	# array([ 20.8,  53.4,  61.8,  93.2])
names[ordered_indices]	#array(['bob', 'jan', 'ad', 'sue'], dtype='|S3')
```

> 使用函数并不会改变原来数组的值
>
> `argsort` 方法与 `argsort` 函数的使用没什么区别，也不会改变数组的值。即上面的与weights.argsort()的结果一样

#### 二维数组排序

对于多维数组，sort方法默认沿着最后一维开始排序。改变轴，可以对任意一列进行排序

#### searchsorted函数

`searchsorted(sorted_array, values)` 接受两个参数，其中，第一个必需是已排序的数组。

`searchsorted` 返回的值相当于**保持第一个数组的排序性质不变**，**将第二个数组中的值插入第一个数组中的位置**

```python
sorted_array = linspace(0,1,5)	#0到1之间生成5个数，即0,0.25,0.5,0.75,1
values = array([.1,.8,.3,.12,.5,.25])
searchsorted(sorted_array, values)	#array([1, 4, 2, 1, 2, 1], dtype=int64)
```

> 例如：`0.1` 在 [0.0, 0.25) 之间，所以插入时应当放在第一个数组的索引 `1` 处，故第一个返回值为 `1`。

**举个例子**

```python
from numpy.random import rand
data = rand(100)
data.sort()

bounds = .4, .6		#注意：不加括号，则默认是元组
#返回0.4和0.6两个值对应的插入位置
low_idx, high_idx = searchsorted(data, bounds)
#利用插入位置，将数组中在这两个值之间的所有值都提取出来
data[low_idx:high_idx]
```

### 03.06 数组形状

#### 修改数组形状

**主要有shape和reshape。前者是直接修改原来数组，后者不修改原来数组的值，而是返回一个新的数组。两者都不能改变数组中元素的综述，否则会报错。**

```python
from numpy import *
a = arange(6)
a.shape = 2,3
a.reshape(3,2)
```

#### 使用newaxis增加数组维数

```python
a = arange(3)
shape(a)	#(3,)
y = a[newaxis, :]
shape(y)	#(1,3)
y = a[:, newaxis]
shape(y)	#(3,1)
y = a[newaxis, newaxis, :]
shape(y)	#(1, 1, 3)
```

#### squeeze方法去除多余的轴

squeeze 返回一个将所有长度为1的维度去除的新数组。

```python
a = arange(6)
a.shape = (2,1,3)
b = a.squeeze()
b.shape		#(2,3)
```

#### transpose数组转置

使用 `transpose` 返回数组的转置，本质上是**将所有维度反过来**	

```python
a					#array([[[0, 1, 2]],
      			 	#		[[3, 4, 5]]])
a.transpose()		#array([[[0, 3]],
					#       [[1, 4]],
					#       [[2, 5]]])
# 简写形式如下
a.T
```

>对于复数数组，转置并不返回复共轭，只是单纯的交换轴的位置
>
>转置可以作用于多维数组。例如将形状为(3,4,5)的数组转置为(5,4,3)
>
>转置返回的是对原数组的另一种view，所以改变转置会改变原来数组的值。



#### concatenate数组连接

需要将不同的数组按照一定的顺序连接起来，用`concatenate((a0,a1,...,aN), axis=0)`。这些**数组要用 `()` 包括到一个元组中去，并给定轴。除了给定的轴外，这些数组其他轴的长度必须是一样的。**

默认沿第一维连接，也可以指定维度。

```python
x = array([
        [0,1,2],
        [10,11,12]
    ])
y = array([
        [50,51,52],
        [60,61,62]
    ])
z = concatenate((x,y))		#z.shape为(4,3)
z = concatenate((x,y), axis=1)		#z.shape为(2,6)
#这里 x 和 y 的形状是一样的，还可以将它们连接成三维的数组，但是 concatenate 不能提供这样的功能，不过可以这样:
z = array((x,y))		#z.shape为(2,2,3)
```

针对上面的三种情况，numpy分别提供了三种对应的函数

- vstack：对应第一维拼接，即vstack((x, y))
- hstack：对应第二维拼接
- dstack：对应升维拼接

#### flatten数组

`flatten` 方法的作用是将多维数组转化为1维数组。返回的是数组的复制，因此，**改变 `b` 并不会影响 `a` 的值**

#### flat属性

可以使用数组自带的 `flat` 属性

```python
a.flat	#<numpy.flatiter at 0x195229db650>
b = a.flat
b		#<numpy.flatiter at 0x195229dc0a0>
b[0]	#0
```

`a.flat` 相当于返回了所有元组组成的一个迭代器。与flatten不同，此时修改b的值，a的值也会改变



#### ravel方法

除此之外，还可以使用 `ravel` 方法，`ravel` 使用高效的表示方式

```python
a = array([[0,1],
           [2,3]])
b = a.ravel()
b	#array([0, 1, 2, 3])
#修改 b 会改变 a 
b[0] = 10
a	#array([[10,  1],
    #  		[ 2,  3]])
#另一种情况
a = array([[0,1],
           [2,3]])
aa = a.transpose()
b = aa.ravel()
b	#array([0, 2, 1, 3])
b[0] = 10
aa		#array([[0, 2],
       	#		[1, 3]])
a		#array([[0, 1],
       	#		[2, 3]])
```

可以看到，在这种情况下，修改 `b` 并不会改变 `aa` 的值，原因是我们用来 `ravel` 的对象 `aa` 本身是 `a` 的一个view。

#### atleast_xd函数

保证数组至少有 `x` 维，`x` 可以取值 1，2，3。

```python
x = 1
atleast_1d(x)	#array([1])

a = array([1,2,3])
b = atleast_2d(a)
b.shape		#(1,3)

c = atleast_3d(b)
c.shape		#(1,3,1)
```

### 03.07 对角线

获取对角线使用diagonal方法（不管数组是不是两维相等）

```python
import numpy as np
a = np.array([11,21,31,12,22,32,13,23,33])
a.shape = 3,3
a.diagonal()	#array([11, 22, 33])
a.diagonal(offset=1)	#array([21, 32])
a.diagonal(offset=-1)	#array([12, 23])
```

当然也可以使用花式索引来得到对角线

```python
i = [0,1,2]
a[i, i]
# 修改对角线的值
a[i, i] = 2
#修改次对角线的值
i = np.array([0,1])
a[i, i + 1] = 1
```

### 03.08 数组与字符串的转换

#### tostring 方法

```python
import numpy as np
a = np.array([[1,2],
           [3,4]], 
          dtype = np.uint8)
#转化为字符串
a.tostring()	#'\x01\x02\x03\x04'
#可以使用不同的顺序来转换字符串，例如下面按照Fortran格式，以列来读：
a.tostring(order='F')	#'\x01\x03\x02\x04'
```

#### fromstring 函数

可以使用 `fromstring` 函数从字符串中读出数据，不过要指定类型

```python
s = a.tostring()
a = np.fromstring(s, 
                  dtype=np.uint8)
a	#array([1, 2, 3, 4], dtype=uint8)
#此时，返回的数组是一维的，需要重新设定维度
a.shape = 2,2
```

对于文本文件，推荐使用：

- loadtxt
- genfromtxt
- savetxt

对于二进制文本文件，推荐使用：

- save
- load
- savez

### 03.09 数组属性方法总结

| 1                                   | **基本属性**                                     |
| ----------------------------------- | ------------------------------------------------ |
| `a.dtype`                           | 数组元素类型 `float32,uint8,...`                 |
| `a.shape`                           | 数组形状 `(m,n,o,...)`                           |
| `a.size`                            | 数组元素数                                       |
| `a.itemsize`                        | 每个元素占字节数                                 |
| `a.nbytes`                          | 所有元素占的字节                                 |
| `a.ndim`                            | 数组维度                                         |
| 2                                   | **形状相关**                                     |
| `a.flat`                            | 所有元素的迭代器                                 |
| `a.flatten()`                       | 返回一个1维数组的复制                            |
| `a.ravel()`                         | 返回一个1维数组，高效                            |
| `a.resize(new_size)`                | 改变形状                                         |
| `a.swapaxes(axis1, axis2)`          | 交换两个维度的位置                               |
| `a.transpose(*axex)`                | 交换所有维度的位置                               |
| `a.T`                               | 转置，`a.transpose()`                            |
| `a.squeeze()`                       | 去除所有长度为1的维度                            |
| 3                                   | **填充复制**                                     |
| `a.copy()`                          | 返回数组的一个复制                               |
| `a.fill(value)`                     | 将数组的元组设置为特定值                         |
| 4                                   | **转化**                                         |
| `a.tolist()`                        | 将数组转化为列表                                 |
| `a.tostring()`                      | 转换为字符串                                     |
| `a.astype(dtype)`                   | 转化为指定类型                                   |
| `a.byteswap(False)`                 | 转换大小字节序                                   |
| `a.view(type_or_dtype)`             | 生成一个使用相同内存，但使用不同的表示方法的数组 |
| 5                                   | **复数**                                         |
| `a.imag`                            | 虚部                                             |
| `a.real`                            | 实部                                             |
| `a.conjugate()`                     | 复共轭                                           |
| `a.conj()`                          | 复共轭（缩写）                                   |
| 6                                   | **保存**                                         |
| `a.dump(file)`                      | 将二进制数据存在file中                           |
| `a.dump()`                          | 将二进制数据表示成字符串                         |
| `a.tofile(fid, sep="",format="%s")` | 格式化ASCⅡ码写入文件                             |
| 7                                   | **查找排序**                                     |
| `a.nonzero()`                       | 返回所有非零元素的索引                           |
| `a.sort(axis=-1)`                   | 沿某个轴排序                                     |
| `a.argsort(axis=-1)`                | 沿某个轴，返回按排序的索引                       |
| `a.searchsorted(b)`                 | 返回将b中元素插入a后能保持有序的索引值           |
| 8                                   | **元素数学操作**                                 |
| `a.clip(low, high)`                 | 将数值限制在一定范围内                           |
| `a.round(decimals=0)`               | 近似到指定精度                                   |
| `a.cumsum(axis=None)`               | 累加和                                           |
| `a.cumprod(axis=None)`              | 累乘积                                           |
| 9                                   | **约简操作**                                     |
| `a.sum(axis=None)`                  | 求和                                             |
| `a.prod(axis=None)`                 | 求积                                             |
| `a.min(axis=None)`                  | 最小值                                           |
| `a.max(axis=None)`                  | 最大值                                           |
| `a.argmin(axis=None)`               | 最小值索引                                       |
| `a.argmax(axis=None)`               | 最大值索引                                       |
| `a.ptp(axis=None)`                  | 最大值减最小值                                   |
| `a.mean(axis=None)`                 | 平均值                                           |
| `a.std(axis=None)`                  | 标准差                                           |
| `a.var(axis=None)`                  | 方差                                             |
| `a.any(axis=None)`                  | 只要有一个不为0，返回真，逻辑或                  |
| `a.all(axis=None)`                  | 所有都不为0，返回真，逻辑与                      |

### 03.10 生成数组的函数

#### arange

`arange` 类似于**Python**中的 `range` 函数，只不过返回的不是列表，而是数组：

​				`arange(start, stop=None, step=1, dtype=None)`

产生一个在**区间 `[start, stop)` 之间，以 `step` 为间隔的数组，如果只输入一个参数，则默认从 `0` 开始，并以这个值为结束**

与 `range` 不同， `arange` 允许非整数值输入，产生一个非整型的数组

```
np.arange(0, 2 * np.pi, np.pi / 4)
```

数组的类型默认由参数 `start, stop, step` 来确定，也可以指定具体类型

> 由于存在精度问题，使用浮点数可能出现问题：
>
> ```
> np.arange(1.5, 2.1, 0.3)	#array([ 1.5,  1.8,  2.1])
> ```

#### linspace

`linspace(start, stop, N)` 产生 **`N` 个等距分布在 `[start, stop]`间的元素组成的数组，包括 `start, stop`**。

#### logspace

`logspace(start, stop, N)`产生 N 个对数等距分布的数组，默认以10为底:

```python
np.logspace(0, 1, 5)	#产生的值为 [10^0,10^0.25,10^0.5,10^0.75,10^1]
```



#### meshgrid

有时候需要在二维平面中生成一个网格，这时候可以使用 `meshgrid` 来完成这样的工作

#### ogrid , mgrid



#### ones , zeros

产生一个制定形状的全 `0` 或全 `1` 的数组，还可以制定数组类型

产生一个全是 `5` 的数组:

```python
np.ones([2,3]) * 5	#array([[ 5.,  5.,  5.],
       				#		[ 5.,  5.,  5.]])
```



#### empty

`empty(shape, dtype=float64, order='C')` 使用 `empty` 方法产生一个制定大小的数组（**数组所指向的内存未被初始化，所以值随机**），再用 `fill` 方法填充:

```python
a = np.empty(2)	#array([-0.03412165,  0.05516321])
a.fill(5)		#array([ 5.,  5.])
#另一种替代方法使用索引，不过速度会稍微慢一些
a[:] = 5		#array([ 5.,  5.])
```



#### empty`_`like, ones`_`like, zeros`_`like

`empty_like(a) ones_like(a) zeros_like(a)`产生一个跟 `a` 大小一样，类型一样的对应数组。

```python
a = np.arange(0, 10, 2.5)
np.empty_like(a)	#array([ 0.,  0.,  0.,  0.])
np.zeros_like(a)	#array([ 0.,  0.,  0.,  0.])
np.ones_like(a)		#array([ 1.,  1.,  1.,  1.])
```



#### identity

`indentity(n, dtype=float64)`产生一个 `n` 乘 `n` 的单位矩阵

```python
np.identity(3)
```



### 03.11 矩阵

1. 使用 `mat` 方法将 `2` 维数组转化为矩阵

```python
import numpy as np
a = np.array([[1,2,4],
              [2,5,3], 
              [7,8,9]])
A = np.mat(a)
A	#matrix([[1, 2, 4],
    #    	[2, 5, 3],
    #    	[7, 8, 9]])
    
#也可以使用 **Matlab** 的语法传入一个字符串来生成矩阵 
A = np.mat('1,2,4;2,5,3;7,8,9')
```

2. 利用分块创造新的矩阵

```python
a = np.array([[ 1, 2],
              [ 3, 4]])
b = np.array([[10,20], 
              [30,40]])
np.bmat('a,b;b,a')
#matrix([[ 1,  2, 10, 20],
#        [ 3,  4, 30, 40],
#        [10, 20,  1,  2],
#        [30, 40,  3,  4]])
```

3. 矩阵与向量的乘法

```python
x = np.array([[1], [2], [3]])
A * x
#matrix([[17],
#        [21],
#        [50]])
```

4. ` A.I` 表示 `A` 矩阵的逆矩阵

5. 矩阵指数表示矩阵连乘

```python
A ** 4
```

### 03.12 一般函数

#### 三角函数

```
sin(x)
cos(x)
tan(x)
sinh(x)
conh(x)
tanh(x)
arccos(x)
arctan(x)
arcsin(x)
arccosh(x)
arctanh(x)
arcsinh(x)
arctan2(x,y)
```

#### 向量操作

```
dot(x,y)
inner(x,y)
cross(x,y)
vdot(x,y)
outer(x,y)
kron(x,y)
tensordot(x,y[,axis])
```

#### 其他操作

```
exp(x)
log(x)
log10(x)
sqrt(x)
absolute(x)
conjugate(x)
negative(x)
ceil(x)
floor(x)
fabs(x)
hypot(x)	#hypot 返回对应点 (x,y) 到原点的距离。
fmod(x)
maximum(x,y)
minimum(x,y)
```

#### 类型处理

```
iscomplexobj
iscomplex
isrealobj
isreal
imag
real
real_if_close
isscalar
isneginf
isposinf
isinf
isfinite
isnan
nan_to_num
common_type
typename
```

#### 修改形状

```
atleast_1d
atleast_2d
atleast_3d
expand_dims
apply_over_axes
apply_along_axis
hstack
vstack
dstack
column_stack
hsplit
vsplit
dsplit
split
squeeze
```

#### 其他有用函数

```
fix
mod
amax
amin
ptp
sum
cumsum
prod
cumprod
diff
angle

unwrap
sort_complex
trim_zeros
fliplr
flipud
rot90
diag
eye
select
extract
insert

roots
poly
any
all
disp
unique
nansum
nanmax
nanargmax
nanargmin
nanmin
```

### 03.13 向量化函数

一般对于自定义的函数，可以直接针对单个数值来调用，用于数组时会报错。此时可以使用numpy中的vectorize函数来将函数向量化，这样就可以传入数组了，向量化的函数会对数组中的每个值都调用原来的函数。

```python
def sinc:
	....

vsinc = np.vectorize(sinc)
vsinc(x)
```



### 03.14 二元运算

#### 四则运算

|   运算   |       函数       |
| :------: | :--------------: |
| `a + b`  |    `add(a,b)`    |
| `a - b`  | `subtract(a,b)`  |
| `a * b`  | `multiply(a,b)`  |
| `a / b`  |  `divide(a,b)`   |
| `a ** b` |   `power(a,b)`   |
| `a % b`  | `remainder(a,b)` |

1.数组与标量运算，相当于数组的每个元素都与这个标量运算

2.两个长度相等的数组做运算，将对应下标的值分别做运算。如果长度不同会报错

```python
a = np.array([1,2])
b = np.array([3,4])
a * b	#array([3, 8])
np.multiply(a, b)	#array([3, 8])
#传入第三个参数，相当于把计算结果存到第三个参数中
np.multiply(a, b, a)	#array([3, 8])
```

#### 比较和逻辑运算

数组的比较全是逐元素操作的，生成的结果依然是一个数组。因此如果要比较两个数组是否相等，要用`if all(a==b):`

在数组中做逻辑运算时，`0` 被认为是 `False`，非零则是 `True`。

 `&` 的运算优先于比较运算如 `>` 等，所以必要时候需要加上括号

```python
a = np.array([1,2,4,8])
b = np.array([16,32,64,128])

(a > 3) & (b < 100)	#array([False, False,  True, False], dtype=bool)
```

### 03.15 ufunc 对象

**Numpy** 有两种基本对象：`ndarray (N-dimensional array object)` 和 `ufunc (universal function object)`。`ndarray` 是**存储单一数据类型的多维数组**，而 `ufunc` 则是**能够对数组进行处理的函数**。例如前一节中的二元操作符add就是一种ufunc对象。

可以查看ufunc对象支持的方法：

```python
dir(np.add)
#['__call__', '__class__', '__delattr__', '__doc__', '__format__', '__getattribute__', '__hash__', '__init__', '__name__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'accumulate', 'at', 'identity', 'nargs', 'nin', 'nout', 'ntypes', 'outer', 'reduce', 'reduceat', 'signature', 'types']
```

针对上面提到的方法，下面详细介绍几个：

**reduce()方法**

```python
op.reduce(a)	#op即是ufunc对象
#将op沿着某个轴应用，使得数组 a 的维数降低一维。

#对一维数组相当于直接求和

#多维数组默认只按照第一维进行运算
a = np.array([[1,2,3],[4,5,6]])
np.add.reduce(a)	#array([5, 7, 9])
#指定维度
np.add.reduce(a, 1)	#array([ 6, 15])
# 作用于字符串
a = np.array(['ab', 'cd', 'ef'], np.object)
np.add.reduce(a)	#'abcdef'
```

**accumulate 方法**

`accumulate` 可以看成保存 `reduce` 每一步的结果所形成的数组。也就是做累加

```python
a = np.array([1,2,3,4])
np.add.accumulate(a)
```

**reduceat 方法**

`reduceat` 方法将操作符运用到指定的下标上，返回一个与 `indices` 大小相同的数组

```python
op.reduceat(a, indices)

a = np.array([0, 10, 20, 30, 40, 50])
indices = np.array([1,4])

np.add.reduceat(a, indices)	#array([60, 90])
#这里，indices 为 [1, 4]，所以 60 表示从下标1（包括）加到下标4（不包括）的结果，90 表示从下标4（包括）加到结尾的结果。
```

**outer 方法**

对于 `a` 中每个元素，将 `op` 运用到它和 `b` 的每一个元素上所得到的结果

```python
op.outer(a, b)

a = np.array([0,1])
b = np.array([1,2,3])

np.add.outer(a, b)	#array([[1, 2, 3],
       				#		[2, 3, 4]])
np.add.outer(b, a)	#array([[1, 2],
       				#		[2, 3],
       				#		[3, 4]])
```



### 03.16 choose 函数实现条件筛选

choose是将第一个参数（数组）中的每个值当成索引，查找第二个参数中相应索引位置处的值，这个值可能是数字或数组，是数字的话就直接变成它，是数组的话就变成该数组中与第一个参数（数组）相同位置处的值。

```python
i0 = np.array([[0,1,2],
               [3,4,5],
               [6,7,8]])
i2 = np.array([[20,21,22],
               [23,24,25],
               [26,27,28]])
control = np.array([[1,0,1],
                    [2,1,0],
                    [1,2,2]])

np.choose(control, [i0, 10, i2])	#array([[10,  1, 10],
       								#		[23, 10,  5],
       								#		[10, 27, 28]])
```

**举个例子**

```python
# 将数组中的所有小于10的值变成10，大于15的值变成15
a = np.array([[ 0, 1, 2], 
              [10,11,12], 
              [20,21,22]])
lt = a < 10
gt = a > 15

choice = lt + 2 * gt	#array([[1, 1, 1],
       					#		[0, 0, 0],
       					#		[2, 2, 2]])
np.choose(choice, (a, 10, 15))	#array([[10, 10, 10],
       							#		[10, 11, 12],
       							#		[15, 15, 15]])
```



### 03.17 数组广播机制

在numpy中两个数组的形状不一样时，有时候也可以做运算，这就是广播机制。但两个数组的维度需要满足条件

对于 **Numpy** 来说，维度匹配当且仅当：

- **要么维度相同**
- **要么有一个的维度是1**

**匹配会从最后一维开始进行，直到某一个的维度全部匹配为止**，因此对于以下情况，**Numpy** 都会进行相应的匹配：

|            A            |           B           |         Result          |
| :---------------------: | :-------------------: | :---------------------: |
| 3d array: 256 x 256 x 3 |      1d array: 3      | 3d array: 256 x 256 x 3 |
| 4d array: 8 x 1 x 6 x 1 |  3d array: 7 x 1 x 5  | 3d array: 8 x 7 x 6 x 5 |
|   3d array: 5 x 4 x 3   |      1d array: 1      |   3d array: 5 x 4 x 3   |
|  3d array: 15 x 4 x 13  | 1d array: 15 x 1 x 13 |  3d array: 15 x 4 x 13  |
|     2d array: 4 x 1     |      1d array: 3      |     2d array: 4 x 3     |



### 03.18 数组读写

#### 从文本中读取数组

文本中的数据一般都是**空格（制表符）或者逗号分隔**，后者一般是csv文件。

```python
import numpy as np
#空格分开
%%writefile myfile.txt
2.1 2.3 3.2 1.3 3.1
6.1 3.1 4.2 2.3 1.8
#逗号分开
%%writefile myfile.txt
2.1, 2.3, 3.2, 1.3, 3.1
6.1, 3.1, 4.2, 2.3, 1.8
```

**原始方法**

```python
#首先将数据转化成一个列表组成的列表，再将这个列表转换为数组
data = []
with open('myfile.txt') as f:
    # 每次读一行
    for line in f:
        fileds = line.split()	#逗号分隔文本则用line.split(',')
        row_data = [float(x) for x in fileds]
        data.append(row_data)

data = np.array(data)
```

**loadtxt 函数**

```python
loadtxt(fname, dtype=<type 'float'>, 
        comments='#', delimiter=None, 
        converters=None, skiprows=0, 
        usecols=None, unpack=False, ndmin=0)
#loadtxt 有很多可选参数，其中 delimiter 就是分隔符参数。
#skiprows 参数表示忽略开头的行数，可以用来读写含有标题的文本
#usecols 定义使用哪几列数据
#comments 定义文本中的注释符是什么（这样就可以忽略文本中注释符后面的东西）
#loadtxt返回一个数组


#loadtxt给converters传参来自定义列数据的转换方法，例子如下
%%writefile myfile.txt
2010-01-01 2.3 3.2
2011-01-01 6.1 3.1

import datetime
def date_converter(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d")
data = np.loadtxt('myfile.txt',
                  dtype=np.object, #数据类型为对象
                  converters={0:date_converter,  #第一列使用自定义转换方法
                              1:float,           #第二第三使用浮点数转换
                              2:float})
data	#array([[datetime.datetime(2010, 1, 1, 0, 0), 2.3, 3.2],
       	#		[datetime.datetime(2011, 1, 1, 0, 0), 6.1, 3.1]], dtype=object)

    
#最后记得移除文件
import os
os.remove('myfile.txt')
```

另外还有一个功能更为全面的 `genfromtxt` 函数，能处理更多的情况，但相应的速度和效率会慢一些。

**读写各种格式的文件**

|   文件格式   |         使用的包         |                      函数                      |
| :----------: | :----------------------: | :--------------------------------------------: |
|     txt      |          numpy           | loadtxt, genfromtxt, fromfile, savetxt, tofile |
|     csv      |           csv            |                 reader, writer                 |
|    Matlab    |         scipy.io         |                loadmat, savemat                |
|     hdf      |      pytables, h5py      |                                                |
|    NetCDF    | netCDF4, scipy.io.netcdf |  netCDF4.Dataset, scipy.io.netcdf.netcdf_file  |
| **文件格式** |       **使用的包**       |                    **备注**                    |
|     wav      |     scipy.io.wavfile     |                    音频文件                    |
| jpeg,png,... | PIL, scipy.misc.pilutil  |                    图像文件                    |
|     fits     |          pyfits          |                    天文图像                    |

此外， `pandas` ——一个用来处理时间序列的包中包含处理各种文件的方法，具体可参见它的文档：

http://pandas.pydata.org/pandas-docs/stable/io.html



#### 将数组写入文本

```python
savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')
```

`savetxt` 可以将数组写入文件，默认使用科学计数法的形式保存

```python
data = np.array([[1,2], 
                 [3,4]])

np.savetxt('out.txt', data)

with open('out.txt') as f:
    for line in f:
        print(line),
#1.000000000000000000e+00 2.000000000000000000e+00
#3.000000000000000000e+00 4.000000000000000000e+00
```

也可以使用类似**C**语言中 `printf` 的方式指定输出的格式

```python
data = np.array([[1,2], 
                 [3,4]])
np.savetxt('out.txt', data, fmt="%d") #保存为整数
#逗号分隔的输出：
np.savetxt('out.txt', data, fmt="%.2f", delimiter=',') #保存为2位小数的浮点数，用逗号分隔
#复数值默认会加上括号
```

#### Numpy 二进制格式

数组可以储存成二进制格式，单个的数组保存为 `.npy` 格式，多个数组保存为多个`.npy`文件组成的 `.npz` 格式，每个 `.npy` 文件包含一个数组。

**与文本格式不同，二进制格式保存了数组的 `shape, dtype` 信息，以便完全重构出保存的数组**。

保存的方法：

- `save(file, arr)` 保存单个数组，`.npy` 格式
- `savez(file, *args, **kwds)` 保存多个数组，无压缩的 `.npz` 格式
- `savez_compressed(file, *args, **kwds)` 保存多个数组，有压缩的 `.npz` 格式

读取的方法：

- `load(file, mmap_mode=None)` 对于 `.npy`，返回保存的数组，对于 `.npz`，返回一个名称-数组对组成的字典。

二进制文件比文本文件小很多。

使用`os.stat('a.npz').st_size`命令查看文件大小。

如果数据整齐，压缩后的体积就越小；数据混乱，压缩后的体积就越大。

```python
#保存多个数组
a = np.array([[1.0,2.0], 
              [3.0,4.0]])
b = np.arange(1000)
np.savez('data.npz', a=a, b=b)

#查看里面包含的文件
!unzip -l data.npz		
#Archive:  data.npz
#	Length      Date    Time    Name
#---------  ---------- -----   ----
#      112  2015/08/10 00:46   a.npy
#     4080  2015/08/10 00:46   b.npy
#---------                     -------
#     4192                     2 files

#载入数据
data = np.load('data.npz')
#像字典一样进行操作
data.keys() #['a', 'b']
data['a']	#array([[ 1.,  2.],
       		#		[ 3.,  4.]])
```



### <u>03.19 结构化数组</u>



### <u>03.20 记录数组</u>



### 03.21 内存映射

**Numpy** 有对内存映射的支持。

内存映射也是一种处理文件的方法，主要的函数有：

- `memmap`
- `frombuffer`
- `ndarray constructor`

内存映射文件与虚拟内存有些类似，通过内存映射文件可以保留一个地址空间的区域，同时将物理存储器提交给此区域，内存文件映射的物理存储器来自一个已经存在于磁盘上的文件，而且在对该文件进行操作之前必须首先对文件进行映射。

使用内存映射文件处理存储于磁盘上的文件时，将不必再对文件执行I/O操作，使得内存映射文件在处理大数据量的文件时能起到相当重要的作用。

#### memmap

```python
memmap(filename,
       dtype=uint8,
       mode='r+'
       offset=0
       shape=None
       order=0)
```

`mode` 表示文件被打开的类型：

- `r` 只读
- `c` 复制+写，但是不改变源文件
- `r+` 读写，使用 `flush` 方法会将更改的内容写入文件
- `w+` 写，如果存在则将数据覆盖

`offset` 表示从第几个位置开始。



## 04 python进阶

### 04.01 sys 模块简介

#### 命令行参数

`sys.argv` 显示传入的参数。第一个参数 （`sys.args[0]`） 表示的始终是执行的文件名，然后依次显示传入的参数。

```python
创建一个文件
%%writefile print_args.py
import sys
print(sys.argv)		

%run print_args.py 1 foo	#输出：['print_args.py', '1', 'foo']
```



#### 异常消息

`sys.exc_info()` 可以显示 `Exception` 的信息，返回一个 `(type, value, traceback)` 组成的三元组，可以与 `try/catch` 块一起使用：

```python
try:
    x = 1/0
except Exception:
    print sys.exc_info()
#(<type 'exceptions.ZeroDivisionError'>, ZeroDivisionError('integer division or modulo by zero',), <traceback object at 0x0000000003C6FA08>)
```

`sys.exc_clear()` 用于清除所有的异常消息。

#### <u>标准输入输出流</u>

- sys.stdin
- sys.stdout
- sys.stderr

#### 退出Python

`sys.exit(arg=0)` 用于退出 Python。`0` 或者 `None` 表示正常退出，其他值表示异常。

#### Python Path

`sys.path` 表示 Python 搜索模块的路径和查找顺序：

```python
sys.path
#['D:\\mynotes_from_github\\myNotes\\python笔记\\notes-python-master\\05-advanced-python',
# 'D:\\mynotes_from_github\\myNotes\\python笔记\\notes-python-master',
# 'c:\\users\\10277\\appdata\\local\\programs\\python\\python38\\python38.zip',
# 'c:\\users\\10277\\appdata\\local\\programs\\python\\python38\\DLLs',
# 'c:\\users\\10277\\appdata\\local\\programs\\python\\python38\\lib',
# 'c:\\users\\10277\\appdata\\local\\programs\\python\\python38',
# '',
# 'c:\\users\\10277\\appdata\\local\\programs\\python\\python38\\lib\\site-packages',
# 'c:\\users\\10277\\appdata\\local\\programs\\python\\python38\\lib\\site-#packages\\win32',
# 'c:\\users\\10277\\appdata\\local\\programs\\python\\python38\\lib\\site-#packages\\win32\\lib',
# 'c:\\users\\10277\\appdata\\local\\programs\\python\\python38\\lib\\site-#packages\\Pythonwin',
# 'c:\\users\\10277\\appdata\\local\\programs\\python\\python38\\lib\\site-#packages\\IPython\\extensions',
# 'C:\\Users\\10277\\.ipython']
```

在程序中可以修改，添加新的路径。

#### 操作系统信息

`sys.platform` 显示当前操作系统信息：

- `Windows: win32`
- `Mac OSX: darwin`
- `Linux:   linux2`

返回 `Windows` 操作系统的版本：```sys.getwindowsversion()```

#### Python 版本信息

```python
sys.version
#'3.8.6 (tags/v3.8.6:db45529, Sep 23 2020, 15:52:53) [MSC v.1927 64 bit (AMD64)]'
sys.version_info
#sys.version_info(major=3, minor=8, micro=6, releaselevel='final', serial=0)
```



### 04.02 与操作系统进行交互：os模块

`os` 模块提供了对系统文件进行操作的方法

#### 文件路径操作

- `os.remove(path)` 或 `os.unlink(path)` ：删除指定路径的文件。路径可以是全名，也可以是当前工作目录下的路径。
- `os.removedirs`：删除文件，并删除中间路径中的空文件夹
- `os.chdir(path)`：将当前工作目录改变为指定的路径
- `os.getcwd()`：返回当前的工作目录(完整路径)
- `os.curdir`：表示当前目录的符号（就是个 . ）
- `os.rename(old, new)`：重命名文件
- `os.renames(old, new)`：重命名文件，如果中间路径的文件夹不存在，则创建文件夹
- `os.listdir(path)`：返回给定目录下的所有文件夹和文件名，不包括 `'.'` 和 `'..'` 以及子文件夹下的目录。（`'.'` 和 `'..'` 分别指当前目录和父目录）
- `os.mkdir(name)`：产生新文件夹
- `os.makedirs(name)`：产生新文件夹，如果中间路径的文件夹不存在，则创建文件夹

#### 系统常量

```python
#当前操作系统的换行符：
os.linesep	#'\r\n'

#当前操作系统的路径分隔符：
os.sep		#'/'

#当前操作系统的环境变量中的分隔符（';' 或 ':'）：
os.pathsep	#':'
```

#### os.path 模块

不同的操作系统使用不同的路径规范，这样当我们在不同的操作系统下进行操作时，可能会带来一定的麻烦，而 `os.path` 模块则帮我们解决了这个问题。

### 测试

- `os.path.isfile(path)` ：检测一个路径是否为普通文件
- `os.path.isdir(path)`：检测一个路径是否为文件夹
- `os.path.exists(path)`：检测路径是否存在
- `os.path.isabs(path)`：检测路径是否为绝对路径

### split 和 join

- `os.path.split(path)`：拆分一个路径为 `(head, tail)` 两部分
- `os.path.join(a, *p)`：使用系统的路径分隔符，将各个部分合成一个路径

### 其他

- `os.path.abspath()`：返回路径的绝对路径
- `os.path.dirname(path)`：返回路径中的文件夹部分
- `os.path.basename(path)`：返回路径中的文件部分
- `os.path.splitext(path)`：将路径与扩展名分开
- `os.path.expanduser(path)`：展开 `'~'` 和 `'~user'`



### 04.03 CSV文件和csv模块

标准库中有自带的 `csv` (逗号分隔值) 模块处理 `csv` 格式的文件

```python
import csv
```

#### 读csv 文件

使用csv.reader()来读取文件

```python
%%file data.csv
"alpha 1",  100, -1.443
"beat  3",   12, -0.0934
"gamma 3a", 192, -0.6621
"delta 2a",  15, -4.515

#打开这个文件，并产生一个文件 reader
fp = open("data.csv")
r = csv.reader(fp)

#按行迭代数据
for row in r:
    print(row)
fp.close()

#默认数据内容都被当作字符串处理，不过可以自己进行处理
data = []
with open('data.csv') as fp:
    r = csv.reader(fp)
    for row in r:
        data.append([row[0], int(row[1]), float(row[2])])    
data
```

#### 写 csv 文件

可以使用 `csv.writer` 写入文件，不过相应地，传入的应该是以写方式打开的文件，不过一般要用 `'wb'` 即二进制写入方式，防止出现换行不正确的问题

```python
data = [('one', 1, 1.5), ('two', 2, 8.0)]
with open('out.csv', 'wb') as fp:
    w = csv.writer(fp)
    w.writerows(data)

#显示结果
!cat 'out.csv'	#one,1,1.5
				#two,2,8.0
```

#### 更换分隔符

```python
data = [('one, \"real\" string', 1, 1.5), ('two', 2, 8.0)]
with open('out.psv', 'wb') as fp:
    w = csv.writer(fp, delimiter="|")
    w.writerows(data)
```

#### 其他选项

**`numpy.loadtxt()` 和 `pandas.read_csv()` 可以用来读写包含很多数值数据的 `csv` 文件**

```python
%%file trades.csv
Order,Date,Stock,Quantity,Price
A0001,2013-12-01,AAPL,1000,203.4
A0002,2013-12-01,MSFT,1500,167.5
A0003,2013-12-02,GOOG,1500,167.5

#使用 pandas 进行处理，生成一个 DataFrame 对象
import pandas
df = pandas.read_csv('trades.csv', index_col=0)
print(df)
#通过名字进行索引
df['Quantity'] * df['Price']
#输出为：
#Order
#A0001    203400
#A0002    251250
#A0003    251250
#dtype: float64
```



### <u>04.05 正则表达式和re模块</u>

[正则表达式](http://baike.baidu.com/view/94238.htm)是用来匹配字符串或者子串的一种模式，匹配的字符串可以很具体，也可以很一般化。

`Python` 标准库提供了 `re` 模块。

#### re.match & re.search

在 `re` 模块中， `re.match` 和 `re.search` 是常用的两个方法：

```python
re.match(pattern, string[, flags])
re.search(pattern, string[, flags])
```

两者都寻找第一个匹配成功的部分，成功则返回一个 `match` 对象，不成功则返回 `None`，不同之处在于 `re.match` 只匹配字符串的开头部分，而 `re.search` 匹配的则是整个字符串中的子串。

#### re.findall & re.finditer

`re.findall(pattern, string)` 返回所有匹配的对象， `re.finditer` 则返回一个迭代器。

#### re.split

`re.split(pattern, string[, maxsplit])` 按照 `pattern` 指定的内容对字符串进行分割。

#### re.sub

`re.sub(pattern, repl, string[, count])` 将 `pattern` 匹配的内容进行替换。

#### re.compile

`re.compile(pattern)` 生成一个 `pattern` 对象，这个对象有匹配，替换，分割字符串的方法。

#### 正则表达式规则

正则表达式由一些普通字符和一些元字符（metacharacters）组成。普通字符包括大小写的字母和数字，而元字符则具有特殊的含义：

|  子表达式  |                      匹配内容                      |
| :--------: | :------------------------------------------------: |
|    `.`     |              匹配除了换行符之外的内容              |
|    `\w`    |               匹配所有字母和数字字符               |
|    `\d`    |            匹配所有数字，相当于 `[0-9]`            |
|    `\s`    |          匹配空白，相当于 `[\t\n\t\f\v]`           |
| `\W,\D,\S` |              匹配对应小写字母形式的补              |
|  `[...]`   | 表示可以匹配的集合，支持范围表示如 `a-z`, `0-9` 等 |
|  `(...)`   |              表示作为一个整体进行匹配              |
|     ¦      |                     表示逻辑或                     |
|    `^`     |             表示匹配后面的子表达式的补             |
|    `*`     |        表示匹配前面的子表达式 0 次或更多次         |
|    `+`     |        表示匹配前面的子表达式 1 次或更多次         |
|    `?`     |         表示匹配前面的子表达式 0 次或 1 次         |
|   `{m}`    |            表示匹配前面的子表达式 m 次             |
|   `{m,}`   |          表示匹配前面的子表达式至少 m 次           |
|  `{m,n}`   |     表示匹配前面的子表达式至少 m 次，至多 n 次     |

例如：

- `ca*t       匹配： ct, cat, caaaat, ...`
- `ab\d|ac\d  匹配： ab1, ac9, ...`
- `([^a-q]bd) 匹配： rbd, 5bd, ...`



### 04.05 datetime 模块

`datetime` 提供了基础时间和日期的处理。

#### date对象

```python
import datetime as dt

#可以使用 date(year, month, day) 产生一个 date 对象
d1 = dt.date(2007, 9, 25)
d2 = dt.date(2008, 9, 25)
#可以格式化 date 对象的输出
print(d1)	#2007-09-25
print(d1.strftime('%A, %m/%d/%y'))		#Tuesday, 09/25/07
print(d1.strftime('%a, %m-%d-%Y'))		#Tue, 09-25-2007
#可以看两个日期相差多久
print(d2 - d1)		#366 days, 0:00:00
#返回的是一个 timedelta 对象
d = d2 - d1
print(d.days)		#366
print(d.seconds)	#0
#查看今天的日期
print(dt.date.today())		#2021-11-05
```

#### time 对象

可以使用 `time(hour, min, sec, us)` 产生一个 `time` 对象

```python
t1 = dt.time(15, 38)
t2 = dt.time(18)
#改变显示格式：
print(t1)		#15:38:00
print(t1.strftime('%I:%M, %p'))		#03:38, PM
print(t1.strftime('%H:%M:%S, %p'))	#15:38:00, PM
```

> 因为没有具体的日期信息，所以 `time` 对象不支持减法操作。

#### datetime 对象

可以使用 `datetime(year, month, day, hr, min, sec, us)` 来创建一个 `datetime` 对象。

```python
#获得当前时间
d1 = dt.datetime.now()	#2021-11-05 14:07:59.170271
#给当前的时间加上 30 天，timedelta 的参数是 timedelta(day, hr, min, sec, us)
d2 = d1 + dt.timedelta(30)	#2021-12-05 14:07:59.170271
#除此之外，我们还可以通过一些指定格式的字符串来创建 datetime 对象：
print(dt.datetime.strptime('2/10/01', '%m/%d/%y'))	#2001-02-10 00:00:00
```

#### datetime 格式字符表

| 字符 |                     含义                      |
| :--: | :-------------------------------------------: |
| `%a` |                 星期英文缩写                  |
| `%A` |                   星期英文                    |
| `%w` |         一星期的第几天，`[0(sun),6]`          |
| `%b` |                 月份英文缩写                  |
| `%B` |                   月份英文                    |
| `%d` |                日期，`[01,31]`                |
| `%H` |                小时，`[00,23]`                |
| `%I` |                小时，`[01,12]`                |
| `%j` |           一年的第几天，`[001,366]`           |
| `%m` |                月份，`[01,12]`                |
| `%M` |                分钟，`[00,59]`                |
| `%p` |                   AM 和 PM                    |
| `%S` |    秒钟，`[00,61]` （大概是有闰秒的存在）     |
| `%U` | 一年中的第几个星期，星期日为第一天，`[00,53]` |
| `%W` | 一年中的第几个星期，星期一为第一天，`[00,53]` |
| `%y` |                没有世纪的年份                 |
| `%Y` |                  完整的年份                   |

### <u>04.06 SQL数据库</u>

### 04.07 对象关系映射

### 04.08 函数进阶：参数传递，高阶函数，lambda 匿名函数，global 变量，递归

#### 函数是基本类型

在 `Python` 中，函数是一种基本类型的对象，这意味着

- 可以将函数作为参数传给另一个函数
- 将函数作为字典的值储存
- 将函数作为另一个函数的返回值

```python
def square(x):
    """Square of x."""
    return x*x
def cube(x):
    """Cube of x."""
    return x*x*x
#作为字典的值：
funcs = {
    'square': square,
    'cube': cube,
}

x = 2
print(square(x))	#4
print(cube(x))		#8
for func in sorted(funcs):		#cube 8
    print(func, funcs[func](x))`#square 4
```

#### 函数参数

**引用传递**

`Python` 中的函数传递方式是 `call by reference` 即引用传递，例如，对于这样的用法：

```
x = [10, 11, 12]
f(x)
```

传递给函数 `f` 的是一个指向 `x` 所包含内容的引用，如果我们修改了这个引用所指向内容的值（例如 `x[0]=999`），那么外面的 `x` 的值也会被改变。不过**如果我们在函数中赋给 `x` 一个新的值（例如另一个列表）**，那么在函数外面的 `x` 的值不会改变，举个例子：

```python
def mod_f(x):
    x[0] = 999
    return x
def no_mod_f(x):
    x = [4, 5, 6]
    return x
x = [1, 2, 3]

print(x)			#[1, 2, 3]
print(mod_f(x))		#[999, 2, 3]
print(x)			#[999, 2, 3]
print(no_mod_f(x))	#[4, 5, 6]
print(x)			#[999, 2, 3]
```

**默认参数是可变的**

函数可以传递默认参数，默认参数的绑定发生在函数定义的时候，**以后每次调用默认参数时都会使用同一个引用。**这样的机制会导致这种情况的发生：

```python
def f(x = []):
    x.append(1)
    return x
#默认参数指向的是[]的一个引用。理论上我们希望每次调用f()时都返回[1]；但事实上，每多调用一次，都会得到一个追加的结果，例如调用三次后得到[1,1,1]。这就是因为默认参数指向的是[]的引用，而每次调用都会修改了这个引用的值。

#可以改写成下面这样：
def f(x = None):
    if x is None:
        x = []
    x.append(1)
    return x
#改写后保证了每次都指向None，不管调用多少次都不会被修改。
```

#### 高阶函数

以函数作为参数，或者返回一个函数的函数是高阶函数，常用的例子有 `map` 和 `filter` 函数：

1.`map(f, sq)` 函数将 `f` 作用到 `sq` 的每个元素上去，返回的是一个对象，需要用list才能**返回结果组成的列表**，相当于：

`[f(s) for s in sq]`

```python
map(square, range(5))	#[0, 1, 4, 9, 16]
```

2.`filter(f, sq)` 函数的作用相当于，对于 `sq` 的每个元素 `s`，返回的是一个对象，需要用list才能**返回所有 `f(s)` 为 `True` 的 `s` 组成的列表**，相当于：

`[s for s in sq if f(s)]`

```python
def is_even(x):
    return x % 2 == 0

filter(is_even, range(5))	#[0, 2, 4]
```

3.`reduce(f, sq)` 函数接受一个二元操作函数 `f(x,y)`，并对于序列 `sq` 每次**迭代**合并两个元素：

```python
# 在python3中reduce被从全局名称空间中移除，现在在functools模块中
from functools import reduce
def my_add(x, y):
    return x + y
#先把1、2传进去得到3，再把上一步得到的3和下一个数（即3）传进去，不断迭代。。
reduce(my_add, [1,2,3,4,5])		#15
```

#### 匿名函数（lambda表达式）

在使用 `map`， `filter`，`reduce` 等函数的时候，为了方便，对一些简单的函数，我们通常使用匿名函数的方式进行处理，其基本形式是：

```python
lambda <variables>: <expression>
```

举个例子：

```python
#例如下面这个表达式：
print(list(map(square, range(5))))		#[0, 1, 4, 9, 16]
#可以用lambda表达式替换：
print map(lambda x: x * x, range(5))	#[0, 1, 4, 9, 16]
#这样可以省去函数的定义
#当然也可以写成这样：
s2 = sum(x**2 for x in range(1, 10))	#285
```

#### global变量

一般来说，函数中是可以直接使用全局变量的值的。但是要在函数中修改全局变量的值，需要加上 `global` 关键字。如果不加上这句 `global` 那么全局变量的值不会改变：

```python
x = 15
def print_newx():
    global x
    x = 18
    print(x)  
def print_newx1():
    x = 20
    print(x)
print(x)		#15
print_newx()	#18
print(x)		#18
print_newx1()	#20
print(x)		#18
```

#### <u>递归</u>

递归是指函数在执行的过程中调用了本身，一般用于分治法

### 04.09 迭代器

#### 迭代器简介

```python
#迭代器对象可以在 for 循环中使用
x = [2, 4, 6]
for n in x:
    print(n)
    
#好处是不需要对下标进行迭代，但是有些情况下，我们既希望获得下标，也希望获得对应的值，那么可以将迭代器传给 enumerate 函数，这样每次迭代都会返回一组 (index, value) 组成的元组：
for i, n in enumerate(x):
    print('pos', i, 'is', n)	#pos 0 is 2
								#pos 1 is 4
								#pos 2 is 6

#迭代器对象必须实现 __iter__ 方法：
i = x.__iter__()
print(i)	#<list_iterator object at 0x0000013BE00AAAF0>
#__iter__() 返回的对象支持 next 方法，返回迭代器中的下一个元素：
print(i.next())		#2
print(i.next())		#4
#当下一个元素不存在时，会 raise 一个 StopIteration 错误

#很多标准库函数返回的是迭代器:
r = reversed(x)		#反向迭代
print(r.next())		#6

#字典对象的 iterkeys, itervalues, iteritems 方法返回的都是迭代器：
#他们分别生成对key、value和items的迭代器
x = {'a':1, 'b':2, 'c':3}
i = x.iteritems()	#<dictionary-itemiterator object at 0x0000000003D51B88>
print(i.next())		#('a', 1)
```

#### 自定义迭代器

自定义一个 list 的取反迭代器：

```python
class ReverseListIterator(object):
    
    def __init__(self, list):
        self.list = list
        self.index = len(list)
        
    def __iter__(self):
        return self
    
    def next(self):
        self.index -= 1
        if self.index >= 0:
            return self.list[self.index]
        else:
            raise StopIteration
            
x = range(10)
for i in ReverseListIterator(x):
    print(i),
```

**只要定义了\__init\__、\__iter\__、next这三个方法，我们可以返回任意迭代值**

但是这样的迭代器会有个问题：

```python
i = Collatz(7)
for x, y in zip(i, i):
    #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素
    #打包成一个个元组，然后返回由这些元组组成的列表
    print(x, y)
#22 11
#34 17
#52 26
#13 40
#20 10
#5 16
#8 4
#2 1
```

也就是想实现同一迭代对象，但用不同的迭代器。解决这个问题的办法是将迭代器和可迭代对象分开处理。

```python
#可迭代对象的定义：
class BinaryTree(object):
    def __init__(self, *args):
		...
    def __iter__(self):
        return InorderIterator(self)
#迭代器的定义：
class InorderIterator(object):
    
    def __init__(self, *args):
		...
    def next(self):
		...
#实例化：
tree = BinaryTree(...)
```



### 04.10 生成器

#### 简介

`while` 循环通常有这样的形式：

```python
<do setup>
result = []
while True:
    <generate value>
    result.append(value)
    if <done>:
        break
```

使用迭代器实现这样的循环：

```python
class GenericIterator(object):
    def __init__(self, ...):
        <do setup>
        # 需要额外储存状态
        <store state>
    def next(self): 
        <load state>
        <generate value>
        if <done>:
            raise StopIteration()
        <store state>
        return value
```

更简单的，可以使用**生成器**：

```python
def generator(...):
    <do setup>
    while True:
        <generate value>
        # yield 说明这个函数可以返回多个值！
        yield value
        if <done>:
            break
```

**生成器使用 `yield` 关键字将值输出，而迭代器则通过 `next` 的 `return` 将值返回**；与迭代器不同的是，生成器会自动记录当前的状态，而迭代器则需要进行额外的操作来记录当前的状态。

对于之前的 `collatz` 猜想，简单循环的实现如下：

```python
def collatz(n):
    sequence = []
    while n != 1:
        if n % 2 == 0:
            n /= 2
        else:
            n = 3*n + 1
        sequence.append(n)
    return sequence

for x in collatz(7):
    print(x),		#22 11 34 17 52 26 13 40 20 10 5 16 8 4 2 1
```

迭代器的版本如下：

```python
class Collatz(object):
    def __init__(self,start):
        self.value = start
    def __iter__(self):
        return self
    def next(self):
        if self.value == 1:
            return StopIteration()
        elif self.value % 2 ==0:
            self.value = self.value / 2
        else :
            self.value = 3*self.value + 1
        return self.value
    
for x in Collatz(7):
    print(x)		#22 11 34 17 52 26 13 40 20 10 5 16 8 4 2 1
```

生成器版本：

```python
def Collatz(n):
    while n != 1:
        if n % 2 == 0:
            n /= 2
        else:
            n = 3*n + 1
        yield n
for x in Collatz(7):
    print(x)		#22 11 34 17 52 26 13 40 20 10 5 16 8 4 2 1
```

事实上，生成器也是一种迭代器。它支持 `next` 方法，返回下一个 `yield` 的值

#### 生成器详解

```python
#case1
def foo():
    print("starting...")
    while True:
        res = yield 4
        print("res:",res)
g = foo()
print(next(g))
print("*"*20)
print(next(g))
#输出为：
#starting...
#4
#********************
#res: None
#4

#case2
def foo():
    print("starting...")
    while True:
        res = yield 4
        print("res:",res)
g = foo()
print(next(g))
print("*"*20)
print(g.send(7))
#输出为：
#starting...
#4
#********************
#res: 7
#4
```

**上面的case1可以看出当程序遇到yield时，就返回yield后面的值。下一次next调用就会从上一次yield停止的地方开始，case1中可以看出，上一次调用中还没有给完成res赋值就被yield了。**

**而case2中用了个send()方法，意思也就是给本次调用开始的地方赋值，所以res就被赋值为7了**



### 04.11 with语句和上下文管理器

```python
# create/aquire some resource
...
try:
    # do something with the resource
    ...
finally:
    # destroy/release the resource
    ...
```

处理文件，线程，数据库，网络编程等等资源的时候，我们经常需要使用上面这样的代码形式，以确保资源的正常使用和释放。

好在`Python` 提供了 `with` 语句帮我们自动进行这样的处理，例如之前在打开文件时我们使用：

```python
with open('my_file', 'w') as fp:
    # do stuff with fp
    data = fp.write("Hello world")
    
#这等效于下面的代码，但是要更简便：
fp = open('my_file', 'w')
try:
    # do stuff with f
    data = fp.write("Hello world")
finally:
    fp.close()
```

#### 上下文管理器

其基本用法如下：

```python
with <expression>:
    <block>
```

`<expression>` 执行的结果应当返回一个实现了上下文管理器的对象，即实现这样两个方法，`__enter__` 和 `__exit__`.

**`__enter__` 方法在 `<block>` 执行前执行，而 `__exit__` 在 `<block>` 执行结束后执行**

举个例子：

```python
#可以这样定义一个简单的上下文管理器：
class ContextManager(object):    
    def __enter__(self):
        print("Entering")
    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting")

#使用 with 语句执行：
with ContextManager():
    print("  Inside the with statement")
#输出为：
#Entering
#  Inside the with statement
#Exiting
```

> **即使 `<block>` 中执行的内容出错，`__exit__` 也会被执行**

#### `__`enter`__` 的返回值

如果在 `__enter__` 方法下添加了返回值，那么我们可以**使用 `as` 把这个返回值传给某个参数**。一个通常的做法是将 `__enter__` 的返回值设为这个上下文管理器对象本身，文件对象就是这样做的：

```python
fp = open('my_file', 'r')
print(fp.__enter__())	#<open file 'my_file', mode 'r' at 0x0000000003B63030>
fp.close()
import os
os.remove('my_file')
```

实现方法也很简单：

```python
class ContextManager(object):    
    def __enter__(self):
        print "Entering"
        return(self)    
    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting")
        
with ContextManager() as value:
    print(value)
#输出为：
#Entering
#<__main__.ContextManager object at 0x0000000003D48828>
#Exiting
```

#### 错误处理

上下文管理器对象将错误处理交给 `__exit__` 进行，可以将错误类型，错误值和 `traceback` 等内容作为参数传递给 `__exit__` 函数

```python
class ContextManager(object):
    
    def __enter__(self):
        print("Entering")
    
    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting")
        if exc_type is not None:
            print("  Exception:", exc_value)

#如果没有错误，exc_type, exc_value, traceback这些值都将是 None, 当有错误发生的时候：
with ContextManager():
    print 1/0
#Entering
#Exiting
#  Exception: integer division or modulo by zero
#ZeroDivisionError       .....
```

在这个例子中，我们只是简单的显示了错误的值，并没有对错误进行处理，所以错误被向上抛出了，**如果不想让错误抛出，只需要将 `__exit__` 的返回值设为 `True`**：

```python
#上面的ContextManager类中__exit__方法加一个return为true即可
class ContextManager(object):
		...
    def __exit__(self, exc_type, exc_value, traceback):
			...
            return True
#这样就只会做错误处理而不会将错误抛出
```

#### <u>数据库的例子</u>

对于数据库的 transaction 来说，如果没有错误，我们就将其 `commit` 进行保存，如果有错误，那么我们将其回滚到上一次成功的状态。

```python
class Transaction(object):
    
    def __init__(self, connection):
        self.connection = connection
    
    def __enter__(self):
        return self.connection.cursor()
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is None:
            # transaction was OK, so commit
            self.connection.commit()
        else:
            # transaction had a problem, so rollback
            self.connection.rollback()
```

建立一个数据库，保存一个地址表：

```python
import sqlite3 as db
connection = db.connect(":memory:")

with Transaction(connection) as cursor:
    cursor.execute("""CREATE TABLE IF NOT EXISTS addresses (
        address_id INTEGER PRIMARY KEY,
        street_address TEXT,
        city TEXT,
        state TEXT,
        country TEXT,
        postal_code TEXT
    )""")
```

插入数据：

```python
with Transaction(connection) as cursor:
    cursor.executemany("""INSERT OR REPLACE INTO addresses VALUES (?, ?, ?, ?, ?, ?)""", [
        (0, '515 Congress Ave', 'Austin', 'Texas', 'USA', '78701'),
        (1, '245 Park Avenue', 'New York', 'New York', 'USA', '10167'),
        (2, '21 J.J. Thompson Ave.', 'Cambridge', None, 'UK', 'CB3 0FA'),
        (3, 'Supreme Business Park', 'Hiranandani Gardens, Powai, Mumbai', 'Maharashtra', 'India', '400076'),
    ])
```

假设插入数据之后出现了问题：

```python
with Transaction(connection) as cursor:
    cursor.execute("""INSERT OR REPLACE INTO addresses VALUES (?, ?, ?, ?, ?, ?)""",
        (4, '2100 Pennsylvania Ave', 'Washington', 'DC', 'USA', '78701'),
    )
    raise Exception("out of addresses")
```

那么最新的一次插入将不会被保存，而是返回上一次 `commit` 成功的状态：

```python
cursor.execute("SELECT * FROM addresses")
for row in cursor:
    print(row)
    
#(0, u'515 Congress Ave', u'Austin', u'Texas', u'USA', u'78701')
#(1, u'245 Park Avenue', u'New York', u'New York', u'USA', u'10167')
#(2, u'21 J.J. Thompson Ave.', u'Cambridge', None, u'UK', u'CB3 0FA')
#(3, u'Supreme Business Park', u'Hiranandani Gardens, Powai, Mumbai', u'Maharashtra', u'India', u'400076')
```

#### contextlib模块

很多的上下文管理器有很多相似的地方，为了防止写入很多重复的模式，可以使用 `contextlib` 模块来进行处理。

最简单的处理方式是**使用 `closing` 函数确保对象的 `close()` 方法始终被调用**：

```python
from contextlib import closing
import urllib

with closing(urllib.urlopen('http://www.baidu.com')) as url:
    html = url.read()

print(html[:100])
```

另一个有用的方法是使用修饰符 `@contextlib`

```python
from contextlib import contextmanager

@contextmanager
def my_contextmanager():
    print "Enter"
    yield
    print "Exit"

with my_contextmanager():
    print("  Inside the with statement")
```

**`yield` 之前的部分可以看成是 `__enter__` 的部分，`yield` 的值可以看成是 `__enter__` 返回的值，`yield` 之后的部分可以看成是 `__exit__` 的部分。**

对于之前的数据库 `transaction` 我们可以这样定义：

```python
@contextmanager
def transaction(connection):
    cursor = connection.cursor()
    try:
        yield cursor
    except:
        connection.rollback()
        raise
    else:
        connection.commit()
    #用try块来做错误处理
```

### 04.12 修饰符

#### 函数是一种对象

在 `Python` 中，函数是也是一种对象。

```python
def foo(x):
    print(x)   
print(type(foo))	#<class 'function'>

#查看函数拥有的方法：
dir(foo)

#在这些方法中，__call__ 是最重要的一种方法：
foo.__call__(42)		#42
#相当于foo(42)

#因为函数是对象，所以函数可以作为参数传入另一个函数：
def bar(f, x):
    x += 1
    f(x)
bar(foo,4)		#5
```

#### 修饰符

修饰符是这样的一种函数，它**接受一个函数作为输入，通常输出也是一个函数**：

```python
# 举个最简单例子：
def dec(f):
    print('I am decorating function', id(f))
    return f
#将 len 函数作为参数传入这个修饰符函数：
declen = dec(len)	#输出：I am decorating function 2322007264208
#使用这个新生成的函数：
declen([10,20,30])	#3
```

上面的例子中，我们仅仅返回了函数的本身，也可以利用这个函数生成一个新的函数，看一个新的例子：

```python
def loud(f):
    def new_func(*args, **kw):
        print('calling with', args, kw)
        rtn = f(*args, **kw)
        print('return value is', rtn)
        return rtn
    return new_func
loudlen = loud(len)
loudlen([10, 20, 30])
#输出为：
#calling with ([10, 20, 30],) {}
#return value is 3
```

#### 用 @ 来使用修饰符

`Python` 使用 `@` 符号来将某个函数替换为修饰符之后的函数：

```python
def dec(f):
    print('I am decorating function', id(f))
    return f
#例如这个函数：
def foo(x):
    print x    
foo = dec(foo)	#I am decorating function 64021672
#可以替换为：
@dec
def foo(x):
    print(x)	#I am decorating function 64021112
```

事实上，如果修饰符返回的是一个函数，那么可以**链式的使用修饰符**：

```python
@dec1
@dec2
def foo(x):
    print x
```

#### 举个例子

定义两个修饰器函数，一个将原来的函数值加一，另一个乘二

```python
def plus_one(f):
    def new_func(x):
        return f(x) + 1
    return new_func

def times_two(f):
    def new_func(x):
        return f(x) * 2
    return new_func

#定义函数，先乘二再加一：
@plus_one
@times_two
def foo(x):
    return int(x)
foo(13)		#27
```

#### 修饰器工厂

`decorators factories` 是返回修饰器的函数，例如：

```python
def super_dec(x, y, z):
    def dec(f):
        def new_func(*args, **kw):
            print(x + y + z)
            return f(*args, **kw)
        return new_func
    return dec
```

它的作用在于**产生一个可以接受参数的修饰器**，例如我们想将 `loud` 输出的内容写入一个文件去，可以这样做：

```python
def super_loud(filename):
    fp = open(filename, 'w')
    def loud(f):
        def new_func(*args, **kw):	#*args存放任意个数据，**kw存放任意个字典
            fp.write('calling with' + str(args) + str(kw))
            # 确保内容被写入
            fp.flush()
            fp.close()
            rtn = f(*args, **kw)
            return rtn
        return new_func
    return loud
```

可以这样使用这个修饰器工厂:

```python
@super_loud('test.txt')
def foo(x):
    print(x)

#调用 `foo` 就会在文件中写入内容：    
foo(12)		#12
```



### <u>04.13 修饰符的使用(面向对象这一部分完成再来看)</u>

#### @classmethod 修饰符

在 `Python` 标准库中，有很多自带的修饰符，例如 `classmethod` 将一个对象方法转换了类方法：

```python
class Foo(object):
    @classmethod
    def bar(cls, x):
        print 'the input is', x
        
    def __init__(self):
        pass

#类方法可以通过 类名.方法 来调用：
Foo.bar(12)		#the input is 12
```

#### @property 修饰符

有时候，我们希望像 **Java** 一样支持 `getters` 和 `setters` 的方法，这时候就可以使用 `property` 修饰符：

```python
class Foo(object):
    def __init__(self, data):
        self.data = data
    
    @property
    def x(self):
        return self.data
        
#此时可以使用 .x 这个属性查看数据（不需要加上括号）：
foo = Foo(23)
foo.x		#23

#这样做的好处在于，这个属性是只读的：
foo.x = 1	#AttributeError: can't set attribute

#如果想让它变成可读写，可以加上一个修饰符 @x.setter：
class Foo(object):
    def __init__(self, data):
        self.data = data 
    @property
    def x(self):
        return self.data    
    @x.setter
    def x(self, value):
        self.data = value
```

#### Numpy 的 @vectorize 修饰符

`numpy` 的 `vectorize` 函数讲一个函数转换为 `ufunc`，事实上它也是一个修饰符



### <u>04.14 operator, functools, itertools, toolz, fn, funcy 模块</u>

#### operator 模块

`operator` 模块提供了各种操作符（`+,*,[]`）的函数版本方便使用

```python
import operator as op
# 加法
print(reduce(op.add, range(10)))	#45
# 乘法
print(reduce(op.mul, range(1,10)))	#362880


my_list = [('a', 1), ('bb', 4), ('ccc', 2), ('dddd', 3)]
# 标准排序
print(sorted(my_list))
# 使用元素的第二个元素排序
print(sorted(my_list, key=op.itemgetter(1)))
# 使用第一个元素的长度进行排序：
print(sorted(my_list, key=lambda x: len(x[0])))

#[('a', 1), ('bb', 4), ('ccc', 2), ('dddd', 3)]
#[('a', 1), ('ccc', 2), ('dddd', 3), ('bb', 4)]
#[('a', 1), ('bb', 4), ('ccc', 2), ('dddd', 3)]
```

#### functools 模块

`functools` 包含很多跟函数相关的工具，比如之前看到的 `wraps` 函数，不过最常用的是 `partial` 函数，这个函数允许我们使用一个函数中生成一个新函数，这个函数使用原来的函数，不过某些参数被指定了：



#### itertools 模块

`itertools` 包含很多与迭代器对象相关的工具，其中比较常用的是排列组合生成器 `permutations` 和 `combinations`，还有在数据分析中常用的 `groupby` 生成器：



#### toolz, fn 和 funcy 模块

这三个模块的作用是方便我们在编程的时候使用函数式编程的风格。



### <u>04.15 作用域</u>

在函数中，`Python` 从命名空间中寻找变量的顺序如下：

- `local function scope`
- `enclosing scope`
- `global scope`
- `builtin scope`



### 04.16 动态编译

#### 标准编程语言

对于 **C** 语言，代码一般要先编译，再执行。

```
.c -> .exe
```

#### 解释器语言

shell 脚本

```
.sh -> interpreter
```

#### Byte Code 编译

**Python, Java** 等语言先将代码编译为 byte code（不是机器码），然后再处理：

```
.py -> .pyc -> interpreter
```

#### eval 函数

```
eval(statement, glob, local)
```

使用 `eval` 函数动态执行代码，返回执行的值：

```python
#例如：
a = 1
eval("a+1")		#输出为：2

#可以接收命名空间参数：
local = dict(a=2)
glob = {}
eval("a+1", glob, local)	#3
#这里 local 中的 a 先被找到。
```

#### exec 函数

```
exec(statement, glob, local)
```

使用 `exec` 可以添加修改原有的变量。

```python
a = 1
exec("b = a+1")
print(b)		#2

local = dict(a=2)
glob = {}
exec("b = a+1", glob, local)
print(local)	#{'a': 2, 'b': 3}
#执行之后，b 在 local 命名空间中。
```

#### 警告

动态执行的时候要注意，不要执行不信任的用户输入，因为它们拥有 `Python` 的全部权限。

#### compile 函数生成 byte code

```
compile(str, filename, mode)
```

```python
#举个例子
a = 1
c = compile("a+2", "", 'eval')
eval(c)		#3

a = 1
c = compile("b=a+2", "", 'exec')
exec(c)
b			#3
```

#### abstract syntax trees

```python
import ast

tree = ast.parse("a+2", "", "eval")
ast.dump(tree)	#"Expression(body=BinOp(left=Name(id='a', ctx=Load()), op=Add(), right=Num(n=2)))"

#改变常数的值：
tree.body.right.n = 3
ast.dump(tree)	#"Expression(body=BinOp(left=Name(id='a', ctx=Load()), op=Add(), right=Num(n=3)))"

a = 1
c = compile(tree, '', 'eval')
eval(c)		#4

#安全的使用方法 literal_eval ，只支持基本值的操作：
ast.literal_eval("[10.0, 2, True, 'foo']")	#[10.0, 2, True, 'foo']
```





