

## 待补充模块

### numpy中的结构化数据

### numpy中的记录数组

### python操作数据库模块

### 



## 01. **Python 工具**

### 01.01 Python 简介

### 01.02 Ipython 解释器

### 01.03 Ipython notebook

### 01.04 使用 Anaconda



## [02. **Python 基础**]

### 02.01 Python 入门演示

略

### 02.02 Python 数据类型

#### 常用数据类型 Common Data Types

|   类型    |           例子           |
| :-------: | :----------------------: |
|   整数    |          `-100`          |
|  浮点数   |         `3.1416`         |
|  字符串   |        `'hello'`         |
|   列表    |   `[1, 1.2, 'hello']`    |
|   字典    | `{'dogs': 5, 'pigs': 3}` |
| Numpy数组 |    `array([1, 2, 3])`    |

#### 其他类型 Others

|    类型    |           例子            |
| :--------: | :-----------------------: |
|   长整型   |     `1000000000000L`      |
|   布尔型   |       `True, False`       |
|    元组    |     `('ring', 1000)`      |
|    集合    |        `{1, 2, 3}`        |
| Pandas类型 |    `DataFrame, Series`    |
|   自定义   | `Object Oriented Classes` |

### 02.03 数字

#### 复数 Complex Numbers

**Python** 使用 `j` 来表示复数的虚部。

```python
a = 1 + 2j
type(a)			#complex
a.real			#1.0
a.imag			#2.0
a.conjugate()	#(1-2j)
```

#### 简单的数学函数

```python
#绝对值
abs(-12.4)
#取整：四舍五入
round(21.6)		
#最小值
print(min(2, 3, 4, 5))
#最大值
print(max(2, 4, 3))
```

>  对于round一般是四舍五入，但是对于.5的情况，是向偶数取整。



### 02.04 字符串

#### 简单操作

```python
#加法
s = 'hello ' + 'world'		#'hello world'
#字符串与数字相乘
"echo" * 3					#'echoechoecho'
#字符串长度
len(s)						#11
```

#### 字符串方法

**Python**是一种面向对象的语言，面向对象的语言中一个必不可少的元素就是方法，而字符串是对象的一种，所以有很多可用的方法。跟很多语言一样，**Python使用以下形式来调用方法：`对象.方法(参数)`**

##### **分割**

s.split()将s按照空格（包括多个空格，制表符`\t`，换行符`\n`等）分割，并返回所有分割得到的字符串

```python
line = "1 2 3 4  5"
numbers = line.split()		#['1', '2', '3', '4', '5']
```

s.split(sep)以给定的sep为分隔符对s进行分割。

```python
line = "1,2,3,4 5"
numbers = line.split(',')	#['1', '2', '3', '4 5']
```

##### 连接

与分割相反，s.join(str_sequence)的作用是以s为连接符将字符串序列str_sequence中的元素连接起来，并返回连接后得到的新字符串：

```python
s = ' '
s.join(numbers)		#'1 2 3 4 5'
```

##### 替换

s.replace(part1, part2)将字符串s中指定的部分part1替换成想要的部分part2，并返回新的字符串。而原来的值没有变，**替换方法只是生成了一个新的字符串**。

```python
s = "hello world"
print(s.replace('world', 'python'))		#'hello python'
print(s)		#'hello world'
```

##### 大小写转换

s.upper()方法返回一个将s中的字母全部大写的新字符串。

s.lower()方法返回一个将s中的字母全部小写的新字符串。

> 这两种方法也不会改变原来s的值

##### 字符串判断

str.isalnum()  所有字符都是数字或者字母，为真返回 Ture，否则返回 False。

str.isalpha()   所有字符都是字母，为真返回 Ture，否则返回 False。

str.isdigit()     所有字符都是数字，为真返回 Ture，否则返回 False。

str.islower()    所有字符都是小写，为真返回 Ture，否则返回 False。

str.isupper()   所有字符都是大写，为真返回 Ture，否则返回 False。

str.istitle()      所有单词都是首字母大写，为真返回 Ture，否则返回 False。

str.isspace()   所有字符都是空白字符，为真返回 Ture，否则返回 False。
         

##### 去除多余空格

s.strip()返回一个将s两端的多余空格除去的新字符串。

s.lstrip()返回一个将s开头的多余空格除去的新字符串。

s.rstrip()返回一个将s结尾的多余空格除去的新字符串。

##### 更多方法

可以使用dir函数查看所有可以使用的方法：

```
dir(s)
```

#### 多行字符串, 使用 `()` 或者 `\` 来换行

Python 用一对 `"""` 或者 `'''` 来生成多行字符串：

```python
a = """hello world.
it is a nice day."""
#在储存时，我们在两行字符间加上一个换行符 '\n'，可以达到换行效果
```

当代码太长或者为了美观起见时，我们可以使用 `()` 或者 `\` 两种方法来将一行代码转为多行代码：

```python
a = ("hello, world. "
    "it's a nice day. "
    "my name is xxx")
#"hello, world. it's a nice day. my name is xxx"
a = "hello, world. " \
    "it's a nice day. " \
    "my name is xxx"
#"hello, world. it's a nice day. my name is xxx"
```

#### 字符串与其他数据类型转换

##### 强制转换为字符串

- `str(ob)`强制将`ob`转化成字符串。
- `repr(ob)`也是强制将`ob`转化成字符串。

##### 使用 `int` 将字符串转为整数

还可以指定按照多少进制来进行转换，最后返回十进制表达的整数

```python
int('23')		#23
int('FF', 16)	#255
int('11111111', 2)	#255
```

##### `float` 可以将字符串转换为浮点数

```
float('3.5')	#3.5
```

#### 格式化字符串

**Python**用字符串的`format()`方法来格式化字符串。

具体用法如下:  **字符串中花括号 `{}` 的部分会被format传入的参数替代，传入的值可以是字符串，也可以是数字或者别的对象。**

```python
'{} {} {}'.format('a', 'b', 'c')
#'a b c'
```

可以用数字指定传入参数的相对位置：

```
'{2} {1} {0}'.format('a', 'b', 'c')
#'c b a'
```

还可以指定传入参数的名称：

```
'{color} {n} {x}'.format(n=10, x=1.5, color='blue')
#'blue 10 1.5'
```

可以用`{<field name>:<format>}`指定格式：

```python
from math import pi
'{0:10} {1:10d} {2:10.2f}'.format('foo', 5, 2 * pi)
#'foo                 5       6.28'
#第一个10表示这个字符串占10个字节
```

也可以使用旧式的 `%` 方法进行格式化：

```python
s = "some numbers:"
x = 1.34
y = 2
# 用百分号隔开，括号括起来
t = "%s %f, %d" % (s, x, y)		#'some numbers: 1.340000, 2'
```



### 02.05 索引和分片

#### 索引

对于一个有序序列，可以通过索引的方法来访问对应位置的值。字符串便是一个有序序列的例子，**Python使用 `[]` 来对有序序列进行索引。**

#### 分片

分片用来从序列中提取出想要的子序列，其用法为：

```
var[lower:upper:step]
```

其范围包括 `lower` ，但不包括 `upper` ，即 `[lower, upper)`， `step` 表示取值间隔大小，如果没有默认为`1`。

#### 使用“0”作为索引开头的原因

##### 使用`[low, up)`形式的原因

假设需要表示字符串 `hello` 中的内部子串 `el` ：

|     方式 | `[low, up)` | `(low, up]` | `(lower, upper)` | `[lower, upper]` |
| -------: | ----------: | ----------: | ---------------: | ---------------: |
|     表示 |     `[1,3)` |     `(0,2]` |          `(0,3)` |          `[1,2]` |
| 序列长度 |  `up - low` |  `up - low` |   `up - low - 1` |   `up - low + 1` |

对长度来说，前两种方式比较好，因为不需要烦人的加一减一。

现在只考虑前两种方法，假设要表示字符串`hello`中的从头开始的子串`hel`：

|     方式 | `[low, up)` | `(low, up]` |
| -------: | ----------: | ----------: |
|     表示 |     `[0,3)` |    `(-1,2]` |
| 序列长度 |  `up - low` |  `up - low` |

第二种表示方法从`-1`开始，不是很好，所以选择使用第一种`[low, up)`的形式。

### 02.06 列表

在**Python**中，列表是一个有序的序列。

**列表用一对 `[]` 生成，中间的元素用 `,` 隔开，其中的元素不需要是同一类型，同时列表的长度也不固定。空列表可以用 `[]` 或者 `list()` 生成。**

```python
l = [1, 2.0, 'hello']
empty_list = []
empty_list = list()
```

#### 列表操作

与字符串类似，列表也支持以下的操作：

##### 长度

```
len(l)
```

##### 加法和乘法

```python
a = [1, 2, 3]
b = [3.2, 'hello']
a + b	#[1, 2, 3, 3.2, 'hello']
```

列表与整数相乘，相当于将列表重复相加：

```python
l * 2
#[1, 2.0, 'hello', 1, 2.0, 'hello']
```

##### 索引和分片

列表和字符串一样可以通过索引和分片来查看它的元素。与字符串不同的是，**列表可以通过索引和分片来修改。**

> 通过分片来修改元素时注意：
>
> 对于连续的分片（即步长为 `1` ）：**Python**采用的是整段替换的方法，两者的元素个数并不需要相同
>
> ```python
> a = [10, 1, 2, 11, 12]
> print(a[1:3])	#[1, 2]
> a[1:3] = []
> print(a)		#[10, 11, 12]
> ```
>
> 对于不连续（间隔step不为1）的片段进行修改时，两者的元素数目必须一致
>
> ```python
> a = [10, 11, 12, 13, 14]
> a[::2] = [1, 2, 3]	#[1, 11, 2, 13, 3]
> ```

##### 删除元素

**Python**提供了删除列表中元素的方法 'del'。

```python
#删除列表中的第一个元素
a = [1002, 'a', 'b', 'c']
del a[0]
a	#['a', 'b', 'c']
```

##### 从属关系

用 `in` 来看某个元素是否在某个序列（不仅仅是列表）中，用not in来判断是否不在某个序列中。

```python
a = [10, 11, 12, 13, 14]
print(10 in a)		#True
print(10 not in a)	#False
```

> 可以作用于字符串判断时候包含某些字符



#### 列表方法

##### 不改变列表的方法

###### 列表中某个元素个数count

`l.count(ob)` 返回列表中元素 `ob` 出现的次数。

###### 列表中某个元素位置index

`l.index(ob)` 返回列表中元素 `ob` 第一次出现的索引位置，如果 `ob` 不在 `l` 中会报错。

##### 改变列表的方法

###### 向列表添加单个元素

`l.append(ob)` 将元素 `ob` 添加到列表 `l` 的最后。append每次只添加一个元素，并不会因为这个元素是序列而将其展开。

```python
a = [10, 11, 12]
a.append(11)
a.append([11, 12])
#[10, 11, 12, 11, [11, 12]]
```

###### 向列表添加序列

`l.extend(lst)` 将序列 `lst` 的元素依次添加到列表 `l` 的最后，作用相当于 `l += lst`。

###### 插入元素

`l.insert(idx, ob)` 在索引 `idx` 处插入 `ob` ，之后的元素依次后移。

###### 移除元素

`l.remove(ob)` 会将列表中第一个出现的 `ob` 删除，如果 `ob` 不在 `l` 中会报错。

###### 弹出元素

`l.pop(idx)` 会将索引 `idx` 处的元素**删除，并返回这个元素**。

###### 排序

`l.sort()` 会将列表中的元素按照一定的规则排序。如果不想改变原来列表中的值，可以使用 `sorted` 函数

```python
a = [10, 1, 11, 13, 11, 2]
a.sort()	#[1, 2, 10, 11, 11, 13]
a = [10, 1, 11, 13, 11, 2]
b = sorted(a)
print(a)	#[10, 1, 11, 13, 11, 2]
print(b)	#[1, 2, 10, 11, 11, 13]
```

排序高级用法：

```
sorted(iterable[, cmp[, key[, reverse]]])
```

- cmp为函数，指定排序时进行比较的函数，可以指定一个函数或者lambda函数

- key为函数，指定取待排序元素的哪一项进行排序

  - ```
    students  =  [( 'john' ,  'A' ,  15 ), ( 'jane' ,  'B' ,  12 ), ( 'dave' ,  'B' ,  10 )]
    sorted (students, key = lambda  student : student[ 2 ])
    ```

- reverse输入bool值，指定是否反向

###### 列表反向

`l.reverse()` 会将列表中的元素从后向前排列。

如果不想改变原来列表中的值，可以使用这样的方法：

```python
a = [1, 2, 3, 4, 5, 6]
b = a[::-1]
```

#### 去除列表中重复元素

```python
#第一种：用列表中自带的内置函数set进行删除：
list1 = [1,1,1,4,4,5,5,7,7,7,7,7,9,9]
list2 = list(set(list1))#重新创建一个变量，接收返回值。使用list方法中的set函数
print(list2)

#第二种：使用遍历
list3 = [1,1,1,4,4,5,5,7,7,7,7,7,9,9]
list4=[]#创建空的列表
for i in list3:#使用for in遍历出列表
    if not i in list4:#将遍历好的数字存储到控的列表中，因为使用了if not ，只有为空的的字符才会存里面，如果llist4里面已经有了，则不会存进去，这就起到了去除重复的效果！！
        list4.append(i)#把i存入新的列表中
print(list4)
```

> 第一种方法要注意，当列表中的元素不可以hash处理的时候就不能使用set方法



### 02.07 可变和不可变类型

|                         可变数据类型                         |                        不可变数据类型                        |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| `list`, `dictionary`, `set`, `numpy array`, `user defined objects` | `integer`, `float`, `long`, `complex`, `string`, `tuple`, `frozenset` |
|       列表、字典、集合、numpy数组、用户自定义数据结构        |     整数、浮点数、长整型、复数、字符串、元组、不可变集合     |

### 02.08 元组

#### 基本操作

与列表相似，元组`Tuple`也是个有序序列，用`()`生成，可以索引，切片，但是**元组是不可变的**（长度和各位置元素值）。

```
t = (10, 11, 12, 13, 14)
t	#(10, 11, 12, 13, 14)
t[0]	#10
```

#### 单个元素的元组生成

由于`()`在表达式中被应用，只含有单个元素的元组容易和表达式混淆，所以采用下列方式定义只有一个元素的元组：

```
a = (10,)
```

#### 元组方法

由于元组是不可变的，所以只能有一些不可变的方法，例如计算元素个数 `count` 和元素位置 `index` ，用法与列表一样。

#### zip()和zip(*)函数

将**可迭代的对象**作为参数，将对象中对应的元素打包成一个个**元组**，然后返回由这些元组组成的列表。如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同。

```python
>>> a = [1,2,3]
>>> b = [4,5,6]
>>> c = [4,5,6,7,8]
>>> zipped = zip(a,b)     # 打包为元组的列表
[(1, 4), (2, 5), (3, 6)]
>>> zip(a,c)              # 元素个数与最短的列表一致
[(1, 4), (2, 5), (3, 6)]
>>> zip(*zipped)          # 与 zip 相反，可理解为解压，为zip的逆过程，可用于矩阵的转置
[(1, 2, 3), (4, 5, 6)]
```

举个例子：

```python
from numpy import *
import pandas as pd
def test(x):
    return x, x+1, x+2

data = [['Alex',10],['Bob',12],['Clarke',13],['ttt',16]]
df = pd.DataFrame(data, columns=['Name','Age'])
print(df['Age'].apply(lambda x: test(x)))
print(list(zip(df['Age'].apply(lambda x: test(x)))))
print(list(zip(*df['Age'].apply(lambda x: test(x)))))
```

输出为：

```python
0    (10, 11, 12)
1    (12, 13, 14)
2    (13, 14, 15)
3    (16, 17, 18)
Name: Age, dtype: object
[((10, 11, 12),), ((12, 13, 14),), ((13, 14, 15),), ((16, 17, 18),)]
[(10, 12, 13, 16), (11, 13, 14, 17), (12, 14, 15, 18)]
```







### 02.09 列表与元组的速度比较

元组的生成速度要比列表的生成速度快得多，相差大概一个数量级。

在遍历上，元组和列表的速度表现差不多。

元组的生成速度会比列表快很多，迭代速度快一点，索引速度差不多。

### 02.10 字典

字典 `dictionary` ，在一些编程语言中也称为 `hash` ， `map` ，是一种由键值对组成的数据结构。**Python 使用 `{}` 或者 `dict()` 来创建一个空的字典**

有了dict之后，可以用索引键值的方法向其中添加元素，也可以通过索引来查看元素的值:

```python
a = dict()
a["one"] = "this is number 1"
a["two"] = "this is number 2"
a['one']	#'this is number 1'
a["one"] = "this is number 1, too"	
#{'one': 'this is number 1, too', 'two': 'this is number 2'}
```

#### 初始化字典

Python使用`key: value`这样的结构来表示字典中的元素结构，事实上，可以直接使用这样的结构来初始化一个字典：

```python
b = {'one': 'this is number 1', 'two': 'this is number 2'}
```

> 出于hash的目的，Python中要求这些键值对的**键**必须是**不可变**的，而值可以是任意的Python对象。

**使用 dict 初始化字典**

除了通常的定义方式，还可以通过 `dict()` 转化来生成字典：

```python
inventory = dict(
    [('foozelator', 123),
     ('frombicator', 18), 
     ('spatzleblock', 34), 
     ('snitzelhogen', 23)
    ])
```

#### 字典方法

##### `get` 方法

之前已经见过，用索引可以找到一个键对应的值，但是当字典中没有这个键的时候，Python会报错，这时候可以使用字典的 `get` 方法来处理这种情况，其用法如下：

`d.get(key, default = None)`

返回字典中键 `key` 对应的值，如果没有这个键，返回 `default` 指定的值（默认是 `None` ）。

```python
a = {}
a["one"] = "this is number 1"
a.get("one")	#"this is number 1"
a.get("three", "undefined")		#"undefined"
```

##### `pop` 方法删除元素

`pop` 方法可以用来弹出字典中某个键对应的值，同时也可以指定默认参数：

```
`d.pop(key, default = None)`
```

**删除并返回字典中键 `key` 对应的值**，如果没有这个键，返回 `default` 指定的值（默认是 `None` ）。

#### `update`方法更新字典

之前已经知道，可以通过索引来插入、修改单个键值对，但是如果想**对多个键值对进行操作**，这种方法就显得比较麻烦，好在有 `update` 方法：

`d.update(newd)`

将字典`newd`中的内容更新到字典`d`中去。

#### `in`查询字典中是否有该键

`in` 可以用来判断字典中是否有某个特定的键：

```python
barn = {'cows': 1, 'dogs': 5, 'cats': 3}
'chickens' in barn
#False
```

#### `keys` 方法，`values` 方法和`items` 方法

```
`d.keys()` 
```

返回一个由所有键组成的列表；

```
`d.values()` 
```

返回一个由所有值组成的列表；

```
`d.items()` 
```

返回一个由所有键值对元组组成的列表；





### 02.11 集合

列表和字符串都是一种有序序列，而集合 `set` 是一种无序的序列。

因为集合是无序的，所以当集合中存在两个同样的元素的时候，Python只会保存其中的一个（唯一性）；同时为了确保其中不包含同样的元素，集合中放入的元素只能是不可变的对象（确定性）。

#### 集合生成

可以用`set()`函数来显示的生成空集合。也可以使用一个列表来初始化一个集合：

```python
#三种方式创建集合
a = set()
a = set([1, 2, 3, 1])	#{1, 2, 3}
#集合会自动去除重复元素 1。
a = {1, 2, 3, 1}
```

> 可以用`{}`的形式来创建集合。但是创建空集合的时候只能用`set`来创建，因为在Python中`{}`创建的是一个空的字典

#### 集合操作

集合的操作右并、交、差、对称差、包含关系，假设有这样两个集合：

```python
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

#并操作
a.union(b)	#{1, 2, 3, 4, 5, 6}
a | b		#{1, 2, 3, 4, 5, 6}

#交操作
a.intersection(b)	
a & b		#{3, 4}

#差
a.difference(b)
a - b	#1, 2}
b.difference(a)	#{5, 6}

#对称差：返回在 a 或在 b 中，但是不同时在 a 和 b 中的元素组成的集合
a.symmetric_difference(b)	#{1, 2, 5, 6}
a ^ b
```

**包含关系**：

要判断 `b` 是不是 `a` 的子集，可以用 `b.issubset(a)` 方法，或者更简单的用操作 `b <= a`

```python
b = {1, 2}
a = {1, 2, 3}

b.issubset(a)
b <= a		#True
```

> 方法只能用来测试子集，但是操作符可以用来判断真子集
>
> ```python
> a <= a	#True
> ```



#### 集合方法

##### `add` 方法向集合添加单个元素

跟列表的 `append` 方法类似，用来向集合添加单个元素

`s.add(a)` 将元素 `a` 加入集合 `s` 中。

> 如果添加的是已有元素，集合不改变

##### `update` 方法向集合添加多个元素

跟列表的`extend`方法类似，用来向集合添加多个元素。

`s.update(seq)`

将`seq`中的元素添加到`s`中。

```python
t = {1, 2, 3}
t.update([5, 6, 7])	#{1, 2, 3, 5, 6, 7}
```

##### `remove` 方法移除单个元素

`s.remove(ob)`

从集合`s`中移除元素`ob`，如果不存在会报错。

##### pop方法弹出任意元素

由于集合没有顺序，不能像列表一样按照位置弹出元素，所以`pop` 方法删除并返回集合中任意一个元素，如果集合中没有元素会报错。

##### discard 方法

作用与 `remove` 一样，但是当元素在集合中不存在的时候不会报错。

##### difference_update方法

`a.difference_update(b)`

从a中去除所有属于b的元素

### 02.12 不可变集合

对应于元组（`tuple`）与列表（`list`）的关系，对于集合（`set`），**Python**提供了一种叫做不可变集合（`frozen set`）的数据结构。

使用 `frozenset` 来进行创建：

```python
s = frozenset([1, 2, 3, 'a', 1])
s	#rozenset({1, 2, 3, 'a'})
```

与集合不同的是，不可变集合一旦创建就不可以改变。不可变集合的一个主要应用是用来作为字典的键

例如用一个字典来记录两个城市之间的距离：

```python
flight_distance = {}
city_pair = frozenset(['Los Angeles', 'New York'])
flight_distance[city_pair] = 2498
flight_distance[frozenset(['Austin', 'Los Angeles'])] = 1233
flight_distance[frozenset(['Austin', 'New York'])] = 1515
flight_distance
#{frozenset({'Los Angeles', 'New York'}): 2498,
# frozenset({'Austin', 'Los Angeles'}): 1233,
# frozenset({'Austin', 'New York'}): 1515}
```

由于集合不分顺序，所以不同顺序不会影响查阅结果:

```python
flight_distance[frozenset(['New York','Austin'])]	#1515
flight_distance[frozenset(['Austin','New York'])]	#1515
```

### 02.13 Python 赋值机制

#### 简单类型

先来看这一段代码在**Python**中的执行过程。

```python
x = 500
y = x
y = 'foo'
```

- `x = 500`：

**Python**分配了一个 `PyInt` 大小的内存 `pos1` 用来储存对象 `500` ，然后，Python在命名空间中让变量 `x` 指向了这一块内存，注意，整数是不可变类型，所以这块内存的内容是不可变的。

|             内存             |  命名空间  |
| :--------------------------: | :--------: |
| `pos1 : PyInt(500)` (不可变) | `x : pos1` |

- `y = x`：

**Python**并没有使用新的内存来储存变量 `y` 的值，而是在命名空间中，让变量 `y` 与变量 `x` 指向了同一块内存空间。

|             内存             |       命名空间        |
| :--------------------------: | :-------------------: |
| `pos1 : PyInt(500)` (不可变) | `x : pos1` `y : pos1` |

- `y = 'foo'`：

**Python**此时分配一个 `PyStr` 大小的内存 `pos2` 来储存对象 `foo` ，然后改变变量 `y` 所指的对象。

|                            内存                             |       命名空间        |
| :---------------------------------------------------------: | :-------------------: |
| `pos1 : PyInt(500)` (不可变) `pos2 : PyStr('foo')` (不可变) | `x : pos1` `y : pos2` |

验证：

1.对这一过程进行验证，可以使用 `id` 函数。`id(x)`返回变量 `x` 的内存地址。

2.也可以使用 `is` 来判断是不是指向同一个事物：`x is y`

注意：

**Python**会为每个出现的对象进行赋值，哪怕它们的值是一样的，例如：

```python
x = 500
id(x)	#2539977812528
y =500
id(y)	#2539977813296
x is y	#False
```

不过，为了提高内存利用效率，对于一些简单的对象，如一些数值较小的int对象，**Python**采用了重用对象内存的办法：

```python
x = 2
id(x)
y = 2
id(y)
x is y	#True
```

#### 容器类型

现在来看另一段代码：

```python
x = [500, 501, 502]
y = x
y[1] = 600
y = [700, 800]
```

- `x = [500, 600, 502]`

Python为3个PyInt分配内存 `pos1` ， `pos2` ， `pos3` （不可变），然后为列表分配一段内存 `pos4` ，它包含3个位置，分别指向这3个内存，最后再让变量 `x` 指向这个列表。

|                                                         内存 |   命名空间 |
| -----------------------------------------------------------: | ---------: |
| `pos1 : PyInt(500)` (不可变) `pos2 : PyInt(501)` (不可变) `pos3 : PyInt(502)` (不可变) `pos4 : PyList(pos1, pos2, pos3)` (可变) | `x : pos4` |

- `y = x`

并没有创建新的对象，只需要将 `y` 指向 `pos4` 即可。

|                                                         内存 |              命名空间 |
| -----------------------------------------------------------: | --------------------: |
| `pos1 : PyInt(500)` (不可变) `pos2 : PyInt(501)` (不可变) `pos3 : PyInt(502)` (不可变) `pos4 : PyList(pos1, pos2, pos3)` (可变) | `x : pos4` `y : pos4` |

- `y[1] = 600`

**原来 `y[1]` 这个位置指向的是 `pos2` ，由于不能修改 `pos2` 的值，所以首先为 `600` 分配新内存 `pos5` 。**

**再把 `y[1]` 指向的位置修改为 `pos5`** 。此时，由于 `pos2` 位置的对象已经没有用了，**Python**会自动调用垃圾处理机制将它回收。

|                                                         内存 |              命名空间 |
| -----------------------------------------------------------: | --------------------: |
| `pos1 : PyInt(500)` (不可变) `pos2 :` 垃圾回收 `pos3 : PyInt(502)` (不可变) `pos4 : PyList(pos1, pos5, pos3)` (可变) `pos5 : PyInt(600)` (不可变) | `x : pos4` `y : pos4` |

- `y = [700, 800]`

首先创建这个列表，然后将变量 `y` 指向它。

|                             内存                             |              命名空间 |
| :----------------------------------------------------------: | --------------------: |
| `pos1 : PyInt(500)` (不可变) `pos3 : PyInt(502)` (不可变) `pos4 : PyList(pos1, pos5, pos3)` (可变) `pos5 : PyInt(600)` (不可变) `pos6 : PyInt(700)` (不可变) `pos7 : PyInt(800)` (不可变) `pos8 : PyList(pos6, pos7)` (可变) | `x : pos4` `y : pos8` |



### 02.14 判断语句

#### 基本用法

```python
x = 0.5
if x > 0:
    print("Hey!")
    print ("x is positive")
```

虽然都是用 `if` 关键词定义判断，但与**C，Java**等语言不同，**Python**不使用 `{}` 将 `if` 语句控制的区域包含起来。**Python**使用的是缩进方法。同时，也不需要用 `()` 将判断条件括起来。

基本结构如下：

```
if <condition 1>:
    <statement 1>
    <statement 2>
elif <condition 2>: 
    <statements>
else:
    <statements>
```

#### 值的测试

**Python**不仅仅可以使用布尔型变量作为条件，它可以直接在`if`中使用任何表达式作为条件：

大部分表达式的值都会被当作`True`，但以下表达式值会被当作`False`：

- False
- **None**
- **0**
- **空字符串，空列表，空字典，空集合**



### 02.15 循环

#### while 循环

```
while <condition>:
    <statesments>
```

**Python**会循环执行`<statesments>`，直到`<condition>`不满足为止。

#### for 循环

```
for <variable> in <sequence>:
    <indented block of code>
```

`for` 循环会遍历完`<sequence>`中所有元素为止

#### continue 语句

遇到 `continue` 的时候，程序会**跳过执行后面的语句**，直接返回到循环的最开始继续执行剩余的循环。

#### break 语句

遇到 `break` 的时候，程序会**跳出循环**，不管循环条件是不是满足

#### else语句

与 `if` 一样， `while` 和 `for` 循环后面也可以跟着 `else` 语句，不过要和`break`一起连用。

- **当循环正常结束时，循环条件不满足， `else` 被执行**；
- **当循环被 `break` 结束时，循环条件仍然满足， `else` 不执行**。



### 02.16 列表推导式

**1.常规for循环生成列表：**

```python
values = [10, 21, 4, 7, 12]
squares = []
for x in values:
    squares.append(x**2)
squares		#[100, 441, 16, 49, 144]
```

**2.列表推导式——简单for循环：**

```
values = [10, 21, 4, 7, 12]
squares = [x**2 for x in values]
squares
```

**3.列表推导式——简单for循环加if筛选：**

例如在上面的例子中，假如只想保留列表中不大于`10`的数的平方：

```
values = [10, 21, 4, 7, 12]
squares = [x**2 for x in values if x <= 10]
```

**4.推导式生成集合和字典：**

```python
square_set = {x**2 for x in values if x <= 10}
print(square_set)	#{16, 49, 100}
square_dict = {x: x**2 for x in values if x <= 10}
print(square_dict)	#{10: 100, 4: 16, 7: 49}
```

**5.求和问题：**计算上面例子中生成的列表中所有元素的和：

```
total = sum([x**2 for x in values if x <= 10])
print(total)		#165
```

但是，**Python**会生成这个列表，然后在将它放到垃圾回收机制中（因为没有变量指向它），这毫无疑问是种浪费。

为了解决这种问题，**Python**使用产生式表达式来解决这个问题：

```
total = sum(x**2 for x in values if x <= 10)
print(total)
```

> 与上面相比，只是去掉了括号，但这里并不会一次性的生成这个列表。



### 02.17 函数

#### 定义函数

函数通常有一下几个特征：

- 使用 `def` 关键词来定义一个函数。
- `def` 后面是函数的名称，括号中是函数的参数，不同的参数用 `,` 隔开， `def foo():` 的形式是必须要有的，参数可以为空；
- 使用缩进来划分函数的内容；
- `docstring` 用 `"""` 包含的字符串，用来解释函数的用途，可省略；
- `return` 返回特定的值，如果省略，返回 `None` 。

传入参数时，Python提供了两种选项，第一种是**按照位置传入参数**，另一种则是**使用关键词模式**，显式地指定参数的值：

```python
def add(x, y):
    """Add two numbers"""
    a = x + y
    return a
add(2, 3)
add(x=2, y=3)
```

#### 设定参数默认值

可以在函数定义的时候给参数设定默认值，例如：

```
def quad(x, a=1, b=0, c=0):
    return a*x**2 + b*x + c
```

可以省略有默认值的参数，也可以修改参数的默认值:

```
quad(2.0)
quad(2.0, b=3)
```

#### 接收不定参数

使用如下方法，可以使函数接受不定数目的参数：

```python
def add(x, *args):
    total = x
    for arg in args:
        total += arg
    return total
print(add(1, 2, 3, 4))
print(add(1, 2))
```

这里，`*args` 表示参数数目不定，可以看成一个元组，把第一个参数后面的参数当作元组中的元素。

这样定义的函数**不能使用关键词传入参数**，要使用关键词，可以这样：

```python
def add(x, **kwargs):
    total = x
    for arg, value in kwargs.items():
        print("adding ", arg)
        total += value
    return total
add(10, y=11, z=12, w=13)
```

这里， `**kwargs` 表示参数数目不定，相当于一个字典，关键词和值对应于键值对。

#### 返回多个值

函数可以返回多个值，事实上，**Python**将**返回的值变成了元组**。

```python
from math import atan2
def to_polar(x, y):
    r = (x**2 + y**2) ** 0.5
    theta = atan2(y, x)
    return r, theta
r, theta = to_polar(3, 4)
print(r, theta)
```

列表也有相似的功能：

```python
a, b, c = [1, 2, 3]
a, b, c		#(1, 2, 3)
```

事实上，不仅仅返回值可以用元组表示，也可以**将参数用元组以这种方式传入**：

```python
def add(x, y):
    """Add two numbers"""
    a = x + y
    return a   
z = (2, 3)
add(*z)		#5
```

> 这里的`*`必不可少。

还可以**通过字典传入参数来执行函数**：

```python
def add(x, y):
    """Add two numbers"""
    a = x + y
    return a

w = {'x': 2, 'y': 3}
add(**w)
```

#### map 方法生成序列

可以通过 `map` 的方式利用函数来生成序列：

```python
def sqr(x): 
    return x ** 2
a = [2,3,4]
map(sqr, a)		#<map at 0x1a6c67d88b0>
```

其用法为：`map(aFun, aSeq)`

将函数 `aFun` 应用到序列 `aSeq` 上的每一个元素上，返回一个列表，不管这个序列原来是什么类型。

事实上，根据函数参数的多少，`map` 可以接受多组序列，将其对应的元素作为参数传入函数:

```python
def add(x, y): 
    return x + y

a = (2,3,4)
b = [10,5,3]
list(map(add,a,b))
#[12, 8, 7]
```

**python3中的map变成了一个类，调用它会返回一个对象，要获取其中的值需要调用list()方法**

### 02.18 模块和包

#### 模块

Python会将所有 `.py` 结尾的文件认定为Python代码文件，考虑下面的脚本 `ex1.py` ：

```python
%%writefile ex1.py

PI = 3.1416
def sum(lst):
    tot = lst[0]
    for value in lst[1:]:
        tot = tot + value
    return tot   
w = [0, 1, 2, 3]
print(sum(w), PI)
```

这个脚本可以当作一个模块，可以使用`import`关键词加载并执行它（这里要求`ex1.py`在当前工作目录），在导入时，**Python**会执行一遍模块中的所有内容。

`ex1.py` 中所有的变量都被载入了当前环境中，不过要使用`ex1.变量名`的方法来查看或者修改这些变量。还可以用`ex1.函数名`调用模块里面的函数:

```python
import ex1
ex1.PI
ex1.sum([2, 3, 4])
```

需要重新导入模块时，可以使用`reload`强制重新载入它，例如：

```python
from importlib import reload
reload(ex1)
```

#### `__name__` 属性

有时候我们想将一个 `.py` 文件既当作脚本，又能当作模块用，这个时候可以使用 `__name__` 这个属性。

**只有当文件被当作脚本执行的时候， `__name__`的值才会是 `'__main__'`**，所以我们可以：

```python
%%writefile ex2.py

PI = 3.1416
def sum(lst):
    """ Sum the values in a list
    """
    tot = 0
    for value in lst:
        tot = tot + value
    return tot
def add(x, y):
    " Add two values."
    a = x + y
    return a
def test():
    w = [0,1,2,3]
    assert(sum(w) == 6)
    print('test passed.')   
if __name__ == '__main__':
    test()
```

> 注：`__main__`这个属性始终是该文件名(带.py这个后缀)。而**`__name__`这个属性，在直接执行脚本的时候也是指向带后缀.py的文件名，而当模块导入到其他文件中时，会指向导入模块给他指向的名称（默认不带.py后缀）**

#### 其他导入方法

可以从模块中导入变量：

```
from ex2 import add, PI
```

使用 `from` 后，可以直接使用 `add` ， `PI`。或者使用 `*` 导入所有变量。

#### 包

假设我们有这样的一个文件夹：

foo/

- `__init__.py`
- `bar.py` (defines func)
- `baz.py` (defines zap)

这意味着 foo 是一个包，我们可以这样导入其中的内容：

```python
from foo.bar import func
from foo.baz import zap
```

`bar` 和 `baz` 都是 `foo` 文件夹下的 `.py` 文件。

导入包要求：

- 文件夹 `foo` 在**Python**的搜索路径中
- **`__init__.py` 表示 `foo` 是一个包，它可以是个空文件。**

#### 常用的标准库

- **re 正则表达式**
- copy 复制
- math, cmath 数学
- decimal, fraction
- sqlite3 数据库
- **os, os.path 文件系统**
- **gzip, bz2, zipfile, tarfile 压缩文件**
- **csv, netrc 各种文件格式**
- xml
- htmllib
- ftplib, socket
- cmd 命令行
- pdb
- profile, cProfile, timeit
- collections, heapq, bisect 数据结构
- mmap
- threading, Queue 并行
- multiprocessing
- subprocess
- **pickle, cPickle**
- struct



### 02.19 异常

#### try & except 块

写代码的时候，出现错误必不可免，即使代码没有问题，也可能遇到别的问题。一旦报错，程序就会停止执行，如果不希望程序停止执行，那么我们可以添加一对 `try & except`：

```python
import math
while True:
    try:
        text = raw_input('> ')
        if text[0] == 'q':
            break
        x = float(text)
        y = math.log10(x)
        print "log10({0}) = {1}".format(x, y)
    except ValueError:
        print "the value must be greater than 0"
```

一旦 `try` 块中的内容出现了异常，那么 `try` 块后面的内容会被忽略，**Python**会寻找 `except` 里面有没有对应的内容，如果找到，就执行对应的块，没有则抛出这个异常。

在上面的例子中，`try` 抛出 `ValueError`，`except` 中有对应的内容，所以这个异常被 `except` 捕捉到，程序可以继续执行。

我们可以在except后面捕捉不同的异常，来分别进行处理。

如果在except后面用了Exception，就会捕获所有的异常

#### 自定义异常

异常是标准库中的类，这意味着我们可以自定义异常类：

```
class CommandError(ValueError):
    pass
```

这里我们定义了一个继承自 `ValueError` 的异常类，异常类一般接收一个字符串作为输入，并把这个字符串当作异常信息。

#### finally

try/except块还有一个可选的关键词 finally。

**不管 try 块有没有异常， finally 块的内容总是会被执行，而且会在抛出异常前执行**，因此可以用来作为安全保证，比如确保打开的文件被关闭。如果**异常被捕获了，在最后执行**。

```python
try:
    print 1 / 0
except ZeroDivisionError:
    print 'divide by 0.'
finally:
    print 'finally was called.'
    
#divide by 0.
#finally was called.
```

### 02.20 警告

出现了一些需要让用户知道的问题，但又不想停止程序，这时候我们可以使用警告：

首先导入警告模块：

```
import warnings
```

在需要的地方，我们使用 `warnings` 中的 `warn` 函数：

```
warn(msg, WarningType = UserWarning)
```

```python
def month_warning(m):
    if not 1<= m <= 12:
        msg = "month (%d) is not between 1 and 12" % m
        warnings.warn(msg, RuntimeWarning)

month_warning(13)
#C:\Users\10277\AppData\Local\Temp/ipykernel_10276/2502358554.py:4: RuntimeWarning: month (13) is not between 1 and 12
#  warnings.warn(msg, RuntimeWarning)
```

有时候我们想要忽略特定类型的警告，可以使用 `warnings` 的 `filterwarnings` 函数：

```
filterwarnings(action, category)
```

将 `action` 设置为 `'ignore'` 便可以忽略特定类型的警告：



### 02.21 文件读写

写入测试文件：

```python
%%writefile test.txt
this is a test file.
hello world!
python is good!
today is a good day.
```

#### 读文件

使用 `open` 函数来读文件，使用文件名的字符串作为输入参数：

```
f = open('test.txt')
```

默认以读的方式打开文件，如果文件不存在会报错。可以使用 `read` 方法来读入文件中的所有内容：

```python
text = f.read()
text
#'this is a test file.\nhello world!\npython is good!\ntoday is a good day.\n'
```

也可以按照行读入内容，`readlines` 方法返回一个列表，每个元素代表文件中每一行的内容：

```
f = open('test.txt')
lines = f.readlines()
```

使用完文件之后，需要将文件关闭。

#### 写文件

我们使用 `open` 函数的写入模式来写文件：

```python
f = open('myfile.txt', 'w')
f.write('hello world!')
f.close()
```

- 使用 `w` 模式时，如果**文件不存在会被创建**。如果文件已经存在， **`w` 模式会覆盖之前写的所有内容**。

- 除了写入模式，还有**追加模式 `a`** ，追加模式不会覆盖之前已经写入的内容，而是在之后继续写入。
- 还可以使用**读写模式 `w+`**
- **二进制读写模式 b**，可以wb组合来写二进制文件，也可以rb组合来读二进制文件

写入结束之后一定要将文件关闭，否则可能出现内容没有完全写入文件中的情况。

#### with 方法

事实上，**Python**提供了更安全的方法，当 `with` 块的内容结束后，**Python**会自动调用它的`close` 方法，确保读写的安全：

```python
with open('newfile.txt','w') as f:
    for i in range(3000):
        x = 1.0 / (i - 1000)
        f.write('hello world: ' + str(i) + '\n')
```

### 02.22 运算符

#### divmod(a,b)

将除法和取模结合起来操作，返回一个二元组，二元组第一个数是整数除法的结果，第二个数是取模结果

```python
print(divmod(353,60))
# (5, 53)
print(*divmod(353,60))
# 5, 53)
```

> 符号\* 在元组、列表、集合前的作用是**解包**
>
> 符号\*\* 在字典前的作用也是解包







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

#### 测试

- `os.path.isfile(path)` ：检测一个路径是否为普通文件
- `os.path.isdir(path)`：检测一个路径是否为文件夹
- `os.path.exists(path)`：检测路径是否存在
- `os.path.isabs(path)`：检测路径是否为绝对路径

#### split 和 join

- `os.path.split(path)`：拆分一个路径为 `(head, tail)` 两部分
- `os.path.join(a, *p)`：使用系统的路径分隔符，将各个部分合成一个路径

#### 其他

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



### 04.05 正则表达式和re模块

[正则表达式](http://baike.baidu.com/view/94238.htm)是用来匹配字符串或者子串的一种模式，匹配的字符串可以很具体，也可以很一般化。

`Python` 标准库提供了 `re` 模块。

#### re.match & re.search

在 `re` 模块中， `re.match` 和 `re.search` 是常用的两个方法：

```python
re.match(pattern, string, flags=0)
re.search(pattern, string, flags=0)

#如果result不为None,则group方法则对result进行数据提取
result.group()
```

两者都寻找第一个匹配成功的部分，成功则返回一个 `match` 对象，不成功则返回 `None`，不同之处在于 **`re.match` 只匹配字符串的开头部分**，如果字符串开始不符合正则表达式，则匹配失败，函数返回None。而 **`re.search` 匹配的则是整个字符串中的子串**，直到找到一个匹配。

>- **pattern** : 一个字符串形式的正则表达式
>- string : 要被匹配的原始字符串。
>- **flags** : 可选，表示匹配模式，比如忽略大小写，多行模式等，具体参数为：
>  1. **re.I** 忽略大小写
>  2. **re.L** 做本地化识别（locale-aware）匹配表示特殊字符集 \w, \W, \b, \B, \s, \S 
>  3. **re.M** 多行模式
>  4. **re.S**  使 . 匹配包括换行在内的所有字符
>  5. **re.U** 根据Unicode字符集解析字符。这个标志影响 \w, \W, \b, \B.
>  6. **re.X** 为了增加可读性，忽略空格和 **#** 后面的注释

```python
import re
 
line = "Cats are smarter than dogs";
 
searchObj = re.search( r'(.*) are (.*?) .*', line, re.M|re.I)
 
if searchObj:
   print "searchObj.group() : ", searchObj.group()
   print "searchObj.group(1) : ", searchObj.group(1)
   print "searchObj.group(2) : ", searchObj.group(2)
else:
   print "Nothing found!!"
#searchObj.group() :  Cats are smarter than dogs
#searchObj.group(1) :  Cats
#searchObj.group(2) :  smarter
```

>前面的一个 **r** 表示字符串为非转义的原始字符串，让编译器忽略反斜杠，也就是忽略转义字符。
>
>**(.\*?)** 第二个匹配分组，**.\*?** 后面多个问号，代表非贪婪模式，也就是说只匹配符合条件的最少字符，例如：
>
>```
>\S+c 匹配字符串aaaacaaaaaaac的结果是aaaacaaaaaaac，而\S+?c则会优先匹配aaaac
>```
>
>后面的一个 **.\*** 没有括号包围，所以不是分组，匹配效果和第一个一样，但是不计入匹配结果中。
>
>matchObj.group(1) 得到第一组匹配结果，也就是(.*)匹配到的
>
>matchObj.group(2) 得到第二组匹配结果，也就是(.*?)匹配到的
>
>因为只有匹配结果中只有两组，所以如果填 3 时会报错。

#### re.findall & re.finditer

`re.findall(string[, pos[, endpos]])` 以列表形式返回所有匹配的对象。

> - **string** : 待匹配的字符串。
> - **pos** : 可选参数，指定字符串的起始位置，默认为 0。
> - **endpos** : 可选参数，指定字符串的结束位置，默认为字符串的长度。

```python
import re
 
pattern = re.compile(r'\d+')   # 查找数字
result1 = pattern.findall('runoob 123 google 456')
result2 = pattern.findall('run88oob123google456', 0, 10)
 
print(result1)	#['123', '456']
print(result2)	#['88', '12']
```

`re.finditer(pattern, string, flags=0)` 则返回一个迭代器。

```python
import re
 
it = re.finditer(r"\d+","12a32bc43jf3") 
for match in it: 
    print (match.group())
```



#### re.split

`re.split(pattern, string[, maxsplit=0, flags=0])` 按照 `pattern` 指定的内容对字符串进行分割。

> maxsplit: 分隔次数，maxsplit=1 分隔一次，默认为 0，不限制次

```shell
>>>import re
>>> re.split('\W+', 'runoob, runoob, runoob.')
['runoob', 'runoob', 'runoob', '']
>>> re.split('(\W+)', ' runoob, runoob, runoob.') 
['', ' ', 'runoob', ', ', 'runoob', ', ', 'runoob', '.', '']
>>> re.split('\W+', ' runoob, runoob, runoob.', 1) 
['', 'runoob, runoob, runoob.']
 
>>> re.split('a*', 'hello world')   # 对于一个找不到匹配的字符串而言，split 不会对其作出分割
['hello world']
```



#### re.sub

`re.sub(pattern, repl, string, count=0,flags=0)` 将 `pattern` 匹配的内容进行替换。

> - repl : 替换的字符串，也可为一个函数。
> - count : 模式匹配后替换的最大次数，默认 0 表示替换所有的匹配。

```python
import re
 
phone = "2004-959-559 # 这是一个国外电话号码"
 
# 删除字符串中的 Python注释 
num = re.sub(r'#.*$', "", phone)
print "电话号码是: ", num		#电话号码是:  2004-959-559 
 
# 删除非数字(-)的字符串 
num = re.sub(r'\D', "", phone)
print "电话号码是 : ", num		#电话号码是 :  2004959559
```

> 前面的一个r表示字符串为非转义的原始字符串，让编译器忽略反斜杠，也就是忽略转义字符。

**repl 参数是一个函数的情况：**

```python
import re
# 将匹配的数字乘以 2
def double(matched):
    value = int(matched.group('value'))
    return str(value * 2)
 
s = 'A23G4HFD567'
print(re.sub('(?P<value>\d+)', double, s))	#A46G8HFD1134
```



#### re.compile

`re.compile(pattern[, flags])` 生成一个 `pattern` 对象，这个对象有匹配，替换，分割字符串的方法。

```shell
>>>import re
>>> pattern = re.compile(r'\d+')                    # 用于匹配至少一个数字
>>> m = pattern.match('one12twothree34four')        # 查找头部，没有匹配
>>> print m
None
>>> m = pattern.match('one12twothree34four', 2, 10) # 从'e'的位置开始匹配，没有匹配
>>> print m
None
>>> m = pattern.match('one12twothree34four', 3, 10) # 从'1'的位置开始匹配，正好匹配
>>> print m                                         # 返回一个 Match 对象
<_sre.SRE_Match object at 0x10a42aac0>
>>> m.group(0)   # 可省略 0
'12'
>>> m.start(0)   # 可省略 0
3
>>> m.end(0)     # 可省略 0
5
>>> m.span(0)    # 可省略 0
(3, 5)
```

> 在上面，当匹配成功时返回一个 Match 对象，其中：
>
> - `group([group1, …])` 方法用于获得一个或多个分组匹配的字符串，当要获得整个匹配的子串时，可直接使用 `group()` 或 `group(0)`；
> - `start([group])` 方法用于获取分组匹配的子串在整个字符串中的起始位置（子串第一个字符的索引），参数默认值为 0；
> - `end([group])` 方法用于获取分组匹配的子串在整个字符串中的结束位置（子串最后一个字符的索引+1），参数默认值为 0；
> - `span([group])` 方法返回 `(start(group), end(group))`。

#### 正则表达式对象

##### re.RegexObject

re.compile() 返回 RegexObject 对象。

##### re.MatchObject

group() 返回被 RE 匹配的字符串。

- **start()** 返回匹配开始的位置
- **end()** 返回匹配结束的位置
- **span()** 返回一个元组包含匹配 (开始,结束) 的位置

#### 正则表达式规则

正则表达式由一些普通字符和一些元字符（metacharacters）组成。普通字符包括大小写的字母和数字，而元字符则具有特殊的含义：

##### 字符类

![img](https://img-blog.csdn.net/2018070616184021?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoZXJyeWRyZWFtc292ZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

##### 数量限定类

![img](https://img-blog.csdn.net/20180706172035936?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoZXJyeWRyZWFtc292ZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

##### 位置限定类

![img](https://img-blog.csdn.net/20180706180119741?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoZXJyeWRyZWFtc292ZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

##### 特殊符号

![img](https://img-blog.csdn.net/20180708091045106?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoZXJyeWRyZWFtc292ZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

##### 特殊字符类

| 实例 | 描述                                                         |
| :--- | :----------------------------------------------------------- |
| .    | 匹配除 "\n" 之外的任何单个字符。要匹配包括 '\n' 在内的任何字符，请使用象 '[.\n]' 的模式。 |
| \d   | 匹配一个数字字符。等价于 [0-9]。                             |
| \D   | 匹配一个非数字字符。等价于 [^0-9]。                          |
| \s   | 匹配任何空白字符，包括空格、制表符、换页符等等。等价于 [ \f\n\r\t\v]。 |
| \S   | 匹配任何非空白字符。等价于 [^ \f\n\r\t\v]。                  |
| \w   | 匹配包括下划线的任何单词字符。等价于'[A-Za-z0-9_]'。         |
| \W   | 匹配任何非单词字符。等价于 '[^A-Za-z0-9_]'。                 |
| \A   | 匹配字符串开始                                               |
| \Z   | 匹配字符串结束，如果是存在换行，只匹配到换行前的结束字符串。 |
| \z   | 匹配字符串结束                                               |
| \G   | 匹配最后匹配完成的位置。                                     |
| \b   | 匹配一个单词边界，也就是指单词和空格间的位置。例如， 'er\b' 可以匹配"never" 中的 'er'，但不能匹配 "verb" 中的 'er'。 |

##### ()组合类

| 模式        | 描述                                                         |
| ----------- | ------------------------------------------------------------ |
| (re)        | 对正则表达式分组并记住匹配的文本。在某些替换场景中，可以用转移数字来获取相应位置的分组文本，例如\1表示获取第一个位置的文本；或者使用RegExp.$1获取；python中用group() |
| (?imx)      | 正则表达式包含三种可选标志：i, m, 或 x 。只影响括号中的区域。 |
| (?-imx)     | 正则表达式关闭 i, m, 或 x 可选标志。只影响括号中的区域。     |
| (?: re)     | 类似 (...), 但是不表示一个组，也就是不会保存下来             |
| (?imx: re)  | 在括号中使用i, m, 或 x 可选标志                              |
| (?-imx: re) | 在括号中不使用i, m, 或 x 可选标志                            |
| (?#...)     | 注释.                                                        |
| (?= re)     | 前向肯定界定符。如果所含正则表达式，以 ... 表示，在当前位置成功匹配时成功，否则失败。但一旦所含表达式已经尝试，匹配引擎根本没有提高；模式的剩余部分还要尝试界定符的右边。 |
| (?! re)     | 前向否定界定符。与肯定界定符相反；当所含表达式不能在字符串当前位置匹配时成功 |
| (?> re)     | 匹配的独立模式，省去回溯。                                   |



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

#### 高阶函数:  map、filter、reduce

以函数作为参数，或者返回一个函数的函数是高阶函数，常用的例子有 `map` 和 `filter` 函数：

1.`map(f, sq)` 函数将 `f` 作用到 `sq` 的每个元素上去，返回的是一个map对象，需要**用list才能返回结果组成的列表**，相当于：

`[f(s) for s in sq]`

```python
list(map(square, range(5)))	#[0, 1, 4, 9, 16]
```

上面这种方法需要f是个函数，还有下面一种用法：

```python
test = pd.read_csv('test_clean.csv', sep='\t')
labelName = test.label.unique()
labelIndex = range(len(labelName))
labelNameToIndex = dict(zip(labelName, labelIndex))
test["labelIndex"] = test.label.map(labelNameToIndex)
```

> **即方法二：iterableObject.map(dict)**



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

另一个有用的方法是使用修饰器 `@contextlib`

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

### 04.12 修饰器

#### 为什么用修饰器？

在我们写函数中，经常会需要写一些与函数本身功能无关的大量重复代码，例如日志、性能测试、运行时间记录等。我们可以抽离出这些功能，然后用修饰器把需要这些功能的函数给修饰一些。概括的讲，装饰器的作用就是为已经存在的对象添加额外的功能。

#### 函数是一种对象

在 `Python` 中，函数是也是一种对象，所以函数可以作为参数传入另一个函数。

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

#### 修饰器

修饰器是这样的一种函数，它**接受一个函数作为输入，通常输出也是一个函数**：

```python
def loud(f):
    def new_func(*args, **kw): # args和kw是f中传进来的参数
        print('calling with', args, kw)
        rtn = f(*args, **kw)
        print('return value is', rtn)
        return rtn
    return new_func
#将 len 函数作为参数传入这个修饰器函数：
loudlen = loud(len)
loudlen([10, 20, 30])
#输出为：
#calling with ([10, 20, 30],) {}
#return value is 3
```

#### 用 @ 来使用修饰器

`Python` 使用 `@` 符号来将某个函数替换为修饰器之后的函数：

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

事实上，如果修饰器返回的是一个函数，那么可以**链式的使用修饰器**：

```python
@dec1
@dec2
def foo(x):
    print x
```

> 相当于：dec1(dec2(foo))

#### 链式修饰器

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

#### 对有参函数进行修饰

```python
def w2(fun):
     #定义一个内嵌的包装函数，给传入的函数加上相应功能的包装  
    def wrapper(*args,**kwargs):
        print("this is the wrapper head")
        fun(*args,**kwargs)
        print("this is the wrapper end")
    # 将包装后的函数返回  
    return wrapper

@w2
def hello(name,name2):
    print("hello"+name+name2)

hello("world","!!!")

#输出:
# this is the wrapper head
# helloworld!!!
# this is the wrapper end
```

#### 有返回值的函数

```python
def w3(fun):
    def wrapper():
        print("this is the wrapper head")
        temp=fun()
        print("this is the wrapper end")
        return temp   #要把值传回去呀！！
    return wrapper

@w3
def hello():
    print("hello")
    return "test"

result=hello()
print("After the wrapper,I accept %s" %result)

#输出:
#this is the wrapper head
#hello
#this is the wrapper end
#After the wrapper,I accept test
```



#### 修饰器工厂（有参数的修饰器）

`decorators factories` 是返回修饰器的函数，例如：

```python
def super_dec(x, y, z):		#x,y,z是修饰器传的参数
    def dec(f):
        def new_func(*args, **kw):	#args,kw是被修饰函数f中的参数
            print(x + y + z)
            return f(*args, **kw)
        return new_func
    return dec
```

> 也就是在前面提到的修饰器函数外再套一层函数，这层函数可以定义一些新的所需参数，然后通过在用修饰器的时候，把新的所需参数直接用修饰器传过去。

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



### 04.13 修饰器的使用

python中内置的修饰器有三个，分别是staticmethod、classmethod和property，作用分别是把类中定义的实例方法变成静态方法、类方法和类属性。

#### @staticmethod修饰符与静态方法

**静态方法是类中的不需要实例化的函数，指一些与对象无关、而只与对象本身有关的函数。**譬如，我想定义一个关于时间操作的类，其中有一个获得当前时间的函数：

```python
import time
class TimeTest(object):
    def __init__(self,hour,minute,second):
        self.hour = hour
        self.minute = minute
        self.second = second
    @staticmethod   
    def showTime():      
        return time.strftime("%H:%M:%S", time.localtime())
print TimeTest.showTime()   
t = TimeTest(2,10,10)
nowTime = t.showTime()
print nowTime
```

> 这样可以让功能与实例解绑，直接通过类来调用。

#### @classmethod 修饰器与类方法

类方法是**将类本身作为对象进行操作的方法**。他和静态方法的区别在于：不管这个方式是从实例调用还是从类调用，它都用第一个参数把类传递过来**。**

```python
class Foo(object):
    @classmethod
    def bar(cls, x):	#第一个参数cls，将类传进来
        print 'the input is', x
        
    def __init__(self):
        pass

#类方法可以通过 类名.方法 来调用：
Foo.bar(12)		#the input is 12
```

> **类函数可以通过类名以及实例两种方法调用！**

#### @property 修饰器

有时候，我们希望像 **Java** 一样支持 `getters` 和 `setters` 的方法，这时候就可以使用 `property` 修饰器：

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

#如果想让它变成可读写，可以加上一个修饰器 @x.setter：
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

#### <u>Numpy 的 @vectorize 修饰器</u>

`numpy` 的 `vectorize` 函数将一个函数转换为 `ufunc`，事实上它也是一个修饰器



### 04.14 operator, functools, itertools 模块

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

> **感觉op里的itemgetter这个方法挺重要**

#### functools 模块

`functools` 包含很多跟函数相关的工具，比如之前看到的 `wraps` 函数，不过最常用的是 `partial` 函数，这个函数允许我们使用一个函数中生成一个新函数，这个函数使用原来的函数，不过某些参数被指定了：

```python
from functools import partial
# 将 reduce 的第一个参数指定为加法，得到的是类似求和的函数
sum_ = partial(reduce, op.add)
# 将 reduce 的第一个参数指定为乘法，得到的是类似求连乘的函数
prod_ = partial(reduce, op.mul)
print sum_([1,2,3,4])	#10
print prod_([1,2,3,4])	#24
```

#### itertools 模块

`itertools` 包含很多与迭代器对象相关的工具，其中比较常用的是排列组合生成器 `permutations` 和 `combinations`，还有在数据分析中常用的 `groupby` 生成器：

```python
from itertools import cycle, groupby, islice, permutations, combinations
```

`cycle` **返回一个无限的迭代器，按照顺序重复输出输入迭代器中的内容**，`islice` 则**返回一个迭代器中的一段内容**：

```python
print(list(islice(cycle('abcd'), 0, 10)))
#['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b']
```

`groupby` 按照指定的 `key` 对一组数据进行分组，返回一个字典，字典的键是 `key`，值是一个迭代器：

```python
animals = sorted(['pig', 'cow', 'giraffe', 'elephant',
                  'dog', 'cat', 'hippo', 'lion', 'tiger'], key=len)

# 按照长度进行分组
for k, g in groupby(animals, key=len):
    print(k, list(g)) 
#3 ['pig', 'cow', 'dog', 'cat']
#4 ['lion']
#5 ['hippo', 'tiger']
#7 ['giraffe']
#8 ['elephant']
```

排列：

```python
print([''.join(p) for p in permutations('abc')])
#['abc', 'acb', 'bac', 'bca', 'cab', 'cba']
```

组合：

```python
print([list(c) for c in combinations([1,2,3,4], r=2)])
#[[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
```

> 排列和组合两个方法都是可迭代的。



#### toolz, fn 和 funcy 模块

这三个模块的作用是方便我们在编程的时候使用函数式编程的风格。



### 04.15 作用域

在函数中，`Python` 从命名空间中寻找变量的顺序如下：

- `local function scope`：最内层，包含局部变量，比如一个函数/方法内部。
- `enclosing scope`：包含了非局部(non-local)也非全局(non-global)的变量。比如两个嵌套函数，一个函数（或类） A 里面又包含了一个函数 B ，那么对于 B 中的名称来说 A 中的作用域就为 nonlocal。
- `global scope`:  当前脚本的最外层，比如当前模块的全局变量。
- `builtin scope`：包含了内建的变量/关键字等，最后被搜索。

![img](https://www.runoob.com/wp-content/uploads/2014/05/1418490-20180906153626089-1835444372.png)

local作用域和global作用域都比较简单，大家都知道。只用注意前面有提到过的global关键字可以将局部变量修改为全局变量。

Python 中只有模块（module），类（class）以及函数（def、lambda）才会引入新的作用域，其它的代码块（如 if/elif/else/、try/except、for/while等）是不会引入新的作用域的，也就是说这些语句内定义的变量，外部也可以访问

#### built-in 作用域

```python
def list_length(a):
    return len(a)

a = [1,2,3]
print(list_length(a))
```

这里函数 `len` 就是在 `built-in` 作用域中，即python的内置作用域

```python
import builtins
builtins.len
#<function len(obj, /)>
```

> 内置作用域是通过一个名为 builtin 的标准模块来实现的，但是这个变量名自身并没有放入内置作用域内，所以必须导入这个文件才能够使用它





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
eval(expression, glob, local)
```

>**expression**：这个参数是一个字符串，python会使用globals字典和locals字典作为全局和局部的命名空间，将expression当做一个python表达式

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

**eval 函数可以实现list、dict、tuple与 str 之间的转化**。安全性是其最大的缺点。如果用户恶意输入，例如：__ import__('os').system('dir')  那么 eval() 之后，当前目录文件都会展现在用户前面。



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





