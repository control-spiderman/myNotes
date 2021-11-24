## 待补充内容

### 压缩文件处理模块

### pandas全系列

### matplotlib中的图像基础



## 05 matplotlib

### 05.01 Pyplot 教程

#### Matplotlib 简介

**`matplotlib`** 是一个 **`Python`** 的 `2D` 图形包。

在线文档：[http://matplotlib.org](http://matplotlib.org/) ，提供了 [Examples](http://matplotlib.org/examples/index.html), [FAQ](http://matplotlib.org/faq/index.html), [API](http://matplotlib.org/contents.html), [Gallery](http://matplotlib.org/gallery.html)，其中 [Gallery](http://matplotlib.org/gallery.html) 是很有用的一个部分，因为它提供了各种画图方式的可视化，方便用户根据需求进行选择。

#### 使用 Pyplot

导入相关的包：

```python
import numpy as np
import matplotlib.pyplot as plt
```

`matplotlib.pyplot` 包含一系列类似 **`MATLAB`** 中绘图函数的相关函数。**每个 `matplotlib.pyplot` 中的函数对当前的图像进行一些修改**，例如：产生新的图像，在图像中产生新的绘图区域，在绘图区域中画线，给绘图加上标记，等等…… `matplotlib.pyplot` 会自动记住当前的图像和绘图区域，因此这些函数会直接作用在当前的图像上。

#### plt.show() 函数

默认情况下，`matplotlib.pyplot` 不会直接显示图像，只有调用 `plt.show()` 函数时，图像才会显示出来。

`plt.show()` 默认是在新窗口打开一幅图像，并且提供了对图像进行操作的按钮。

#### plt.plot() 函数

`plt.plot()` 函数可以用来绘图：

```python
plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()
```

<img src="C:\Users\10277\AppData\Roaming\Typora\typora-user-images\image-20211108001819466.png" alt="image-20211108001819466" style="zoom:50%;" />

**基本用法**

`plot` 函数基本的用法有以下四种：

默认参数

- `plt.plot(x,y)`

指定参数

- `plt.plot(x,y, format_str)`

默认参数，`x` 为 `0~N-1`

- `plt.plot(y)`

指定参数，`x` 为 `0~N-1`

- `plt.plot(y, format_str)`

**字符参数**

和 **`MATLAB`** 中类似，我们还可以用字符来指定绘图的格式：

表示颜色的字符参数有：

| 字符  |     颜色      |
| :---: | :-----------: |
| `‘b’` |  蓝色，blue   |
| `‘g’` |  绿色，green  |
| `‘r’` |   红色，red   |
| `‘c’` |  青色，cyan   |
| `‘m’` | 品红，magenta |
| `‘y’` | 黄色，yellow  |
| `‘k’` |  黑色，black  |
| `‘w’` |  白色，white  |

表示类型的字符参数有：

|  字符  |    类型    |  字符  |   类型    |
| :----: | :--------: | :----: | :-------: |
| `'-'`  |    实线    | `'--'` |   虚线    |
| `'-.'` |   虚点线   | `':'`  |   点线    |
| `'.'`  |     点     | `','`  |  像素点   |
| `'o'`  |    圆点    | `'v'`  | 下三角点  |
| `'^'`  |  上三角点  | `'<'`  | 左三角点  |
| `'>'`  |  右三角点  | `'1'`  | 下三叉点  |
| `'2'`  |  上三叉点  | `'3'`  | 左三叉点  |
| `'4'`  |  右三叉点  | `'s'`  |  正方点   |
| `'p'`  |   五角点   | `'*'`  |  星形点   |
| `'h'`  | 六边形点1  | `'H'`  | 六边形点2 |
| `'+'`  |   加号点   | `'x'`  |  乘号点   |
| `'D'`  | 实心菱形点 | `'d'`  | 瘦菱形点  |
| `'_'`  |   横线点   |        |           |

#### 显示范围

与 **`MATLAB`** 类似，这里可以使用 `axis` 函数指定坐标轴显示的范围：

```python
plt.axis([xmin, xmax, ymin, ymax])
```

```python
plt.plot([1,2,3,4], [1,4,9,16], 'ro')
# 指定 x 轴显示区域为 0-6，y 轴为 0-20
plt.axis([0,6,0,20])
plt.show()
```

#### 传入 `Numpy` 数组

之前我们传给 `plot` 的参数都是列表，事实上，向 `plot` 中传入 `numpy` 数组是更常用的做法。事实上，如果传入的是列表，`matplotlib` 会在内部将它转化成数组再进行处理：

```python
import numpy as np
import matplotlib.pyplot as plt

# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', 
         t, t**2, 'bs', 
         t, t**3, 'g^')

plt.show()
```

<img src="C:\Users\10277\AppData\Roaming\Typora\typora-user-images\image-20211108002438789.png" alt="image-20211108002438789" style="zoom:50%;" />

**传入多组数据**

事实上，在上面的例子中，我们不仅仅向 `plot` 函数传入了数组，还传入了多组 `(x,y,format_str)` 参数，它们在同一张图上显示。

这意味着我们不需要使用多个 `plot` 函数来画多组数组，只需要可以将这些组合放到一个 `plot` 函数中去即可。

#### 线条属性

之前提到，我们可以用字符串来控制线条的属性，事实上还可以通过关键词来改变线条的性质，例如 `linwidth` 可以改变线条的宽度，`color` 可以改变线条的颜色:

```python
x = np.linspace(-np.pi,np.pi)
y = np.sin(x)
plt.plot(x, y, linewidth=2.0, color='r')
plt.show()
```

#### **使用 plt.plot() 的返回值来设置线条属性**

`plot` 函数返回一个 `Line2D` 对象组成的列表，每个对象代表输入的一对组合，例如：

- line1, line2 为两个 Line2D 对象

  `line1, line2 = plt.plot(x1, y1, x2, y2)`

- 返回 3 个 Line2D 对象组成的列表

  `lines = plt.plot(x1, y1, x2, y2, x3, y3)`

我们可以使用这个返回值来对线条属性进行设置：

```python
# 加逗号 line 中得到的是 line2D 对象，不加逗号得到的是只有一个 line2D 对象的列表
line, = plt.plot(x, y, 'r-')
# 将抗锯齿关闭
line.set_antialiased(False)
plt.show()
```

#### **plt.setp() 修改线条性质**

更方便的做法是使用 `plt` 的 `setp` 函数：

```python
lines = plt.plot(x, y)# 使用键值对plt.setp(lines, color='r', linewidth=2.0)# 或者使用 MATLAB 风格的字符串对plt.setp(lines, 'color', 'r', 'linewidth', 2.0)plt.show()
```

可以设置的属性有很多，可以使用 `plt.setp(lines)` 查看 `lines` 可以设置的属性，各属性的含义可参考 `matplotlib` 的文档。

```
plt.setp(lines)
```

#### 子图

关于figure和subplot的理解，看下面这张图就可以了：

<img src="C:\Users\10277\AppData\Roaming\Typora\typora-user-images\image-20211108004641539.png" alt="image-20211108004641539" style="zoom:150%;" />

1.`figure()` 函数会产生一个指定编号为 `num` 的图：

```
plt.figure(num)
```

这里，`figure(1)` 其实是可以省略的，因为默认情况下 `plt` 会自动产生一幅图像。

2.使用 `subplot` 可以在一副图中生成多个子图，其参数为：

```
plt.subplot(numrows, numcols, fignum)
```

当 `numrows * numcols < 10` 时，中间的逗号可以省略，因此 `plt.subplot(211)` 就相当于 `plt.subplot(2,1,1)`。

```python
def f(t):    return np.exp(-t) * np.cos(2*np.pi*t)t1 = np.arange(0.0, 5.0, 0.1)t2 = np.arange(0.0, 5.0, 0.02)plt.figure(1)plt.subplot(211)plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')plt.subplot(212)plt.plot(t2, np.cos(2*np.pi*t2), 'r--')plt.show()
```

<img src="C:\Users\10277\AppData\Roaming\Typora\typora-user-images\image-20211108003449091.png" alt="image-20211108003449091" style="zoom:50%;" />

#### 图形上加上文字

`plt.hist()` 可以用来画直方图。

```python
mu, sigma = 100, 15x = mu + sigma * np.random.randn(10000)# the histogram of the datan, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)plt.xlabel('Smarts')plt.ylabel('Probability')plt.title('Histogram of IQ')plt.text(60, .025, r'$\mu=100,\ \sigma=15$')plt.axis([40, 160, 0, 0.03])plt.grid(True)plt.show()
```

<img src="C:\Users\10277\AppData\Roaming\Typora\typora-user-images\image-20211108004003418.png" alt="image-20211108004003418" style="zoom:50%;" />

对于这幅图形，我们使用 `xlabel` ，`ylabel`，`title`，`text` 方法设置了文字，其中：

- `xlabel` ：x 轴标注
- `ylabel` ：y 轴标注
- `title` ：图形标题
- `text` ：在指定位置放入文字

输入特殊符号支持使用 `Tex` 语法，用 `$<some Tex code>$` 隔开。

除了使用 `text` 在指定位置标上文字之外，还可以使用 `annotate` 函数进行注释，`annotate` 主要有两个参数：

- `xy` ：注释位置
- `xytext` ：注释文字位置

```python
#例如：ax = plt.subplot(111)t = np.arange(0.0, 5.0, 0.01)s = np.cos(2*np.pi*t)line, = plt.plot(t, s, lw=2)plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),            arrowprops=dict(facecolor='black', shrink=0.05),)plt.ylim(-2,2)plt.show()
```

<img src="C:\Users\10277\AppData\Roaming\Typora\typora-user-images\image-20211108003835172.png" alt="image-20211108003835172" style="zoom:50%;" />



### 05.02 使用style来配置pyplot风格

```python
import matplotlib.pyplot as pltimport numpy as np%matplotlib inline
```

`style` 是 `pyplot` 的一个子模块，方便进行风格转换， `pyplot` 有很多的预设风格，可以使用 `plt.style.available` 来查看：

```python
plt.style.available#['Solarize_Light2','_classic_test_patch','bmh','classic','dark_background','fast','fivethirtyeight','ggplot','grayscale','seaborn','seaborn-bright','seaborn-colorblind','seaborn-dark','seaborn-dark-palette','seaborn-darkgrid','seaborn-deep','seaborn-muted','seaborn-notebook','seaborn-paper','seaborn-pastel','seaborn-poster','seaborn-talk','seaborn-ticks','seaborn-white','seaborn-whitegrid','tableau-colorblind10']
```

例如，我们可以模仿 `R` 语言中常用的 `ggplot` 风格：

```python
plt.style.use('ggplot')plt.plot(x, y)plt.show()
```

<img src="C:\Users\10277\AppData\Roaming\Typora\typora-user-images\image-20211108005101451.png" alt="image-20211108005101451" style="zoom: 33%;" />

有时候，我们不希望改变全局的风格，只是想暂时改变一下风格，则可以使用 `context` 将风格改变限制在某一个代码块内：

```python
with plt.style.context(('dark_background')):
    plt.plot(x, y, 'r-o')
    plt.show()

#在代码块外绘图则仍然是全局的风格。
plt.plot(x, y, 'r-o')
plt.show()
```

还可以混搭使用多种风格，不过最右边的一种风格会将最左边的覆盖：

```python
plt.style.use(['dark_background', 'ggplot'])
plt.plot(x, y, 'r-o')
plt.show()
```

**自定义风格文件**

自定义文件需要放在 `matplotlib` 的配置文件夹 `mpl_configdir` 的子文件夹 `mpl_configdir/stylelib/` 下，以 `.mplstyle` 结尾。

`mpl_configdir` 的位置可以这样查看：

```python
import matplotlib
matplotlib.get_configdir()
```

里面的内容以 `属性：值` 的形式保存：

```
axes.titlesize : 24
axes.labelsize : 20
lines.linewidth : 3
lines.markersize : 10
xtick.labelsize : 16
ytick.labelsize : 16
```

假设我们将其保存为 `mpl_configdir/stylelib/presentation.mplstyle`，那么使用这个风格的时候只需要调用：

```
plt.style.use('presentation')
```

### 05.03 处理文本（基础）

`matplotlib` 对文本的支持十分完善，包括数学公式，`Unicode` 文字，栅格和向量化输出，文字换行，文字旋转等一系列操作。

#### 基础文本函数

在 `matplotlib.pyplot` 中，基础的文本函数如下：

- `text()` 在 `Axes` 对象的任意位置添加文本
- `xlabel()` 添加 x 轴标题
- `ylabel()` 添加 y 轴标题
- `title()` 给 `Axes` 对象添加标题
- `figtext()` 在 `Figure` 对象的任意位置添加文本
- `suptitle()` 给 `Figure` 对象添加标题
- `anotate()` 给 `Axes` 对象添加注释（可选择是否添加箭头标记）

```python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
%matplotlib inline

# plt.figure() 返回一个 Figure() 对象
fig = plt.figure(figsize=(12, 9))

# 设置这个 Figure 对象的标题
# 事实上，如果我们直接调用 plt.suptitle() 函数，它会自动找到当前的 Figure 对象
fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

# Axes 对象表示 Figure 对象中的子图
# 这里只有一幅图像，所以使用 add_subplot(111)
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)

# 可以直接使用 set_xxx 的方法来设置标题
ax.set_title('axes title')
# 也可以直接调用 title()，因为会自动定位到当前的 Axes 对象
# plt.title('axes title')

ax.set_xlabel('xlabel')
ax.set_ylabel('ylabel')

# 添加文本，斜体加文本框
ax.text(3, 8, 'boxed italics text in data coords', style='italic',
        bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})

# 数学公式，用 $$ 输入 Tex 公式
ax.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)

# Unicode 支持
ax.text(3, 2, unicode('unicode: Institut f\374r Festk\366rperphysik', 'latin-1'))

# 颜色，对齐方式
ax.text(0.95, 0.01, 'colored text in axes coords',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)

# 注释文本和箭头
ax.plot([2], [1], 'o')
ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))

# 设置显示范围
ax.axis([0, 10, 0, 10])

plt.show()
```

![image-20211108005905079](C:\Users\10277\AppData\Roaming\Typora\typora-user-images\image-20211108005905079.png)

#### 文本属性和布局

我们可以通过下列关键词，在文本函数中设置文本的属性：

|                    关键词 |                                                           值 |
| ------------------------: | -----------------------------------------------------------: |
|                     alpha |                                                        float |
|           backgroundcolor |                                         any matplotlib color |
|                      bbox | rectangle prop dict plus key `'pad'` which is a pad in points |
|                  clip_box |                         a matplotlib.transform.Bbox instance |
|                   clip_on |                                              [True ， False] |
|                 clip_path |            a Path instance and a Transform instance, a Patch |
|                     color |                                         any matplotlib color |
|                    family | [ `'serif'` , `'sans-serif'` , `'cursive'` , `'fantasy'` , `'monospace'` ] |
|            fontproperties |            a matplotlib.font_manager.FontProperties instance |
| horizontalalignment or ha |                        [ `'center'` , `'right'` , `'left'` ] |
|                     label |                                                   any string |
|               linespacing |                                                        float |
|            multialignment |                         [`'left'` , `'right'` , `'center'` ] |
|          name or fontname |    string e.g., [`'Sans'` , `'Courier'` , `'Helvetica'` ...] |
|                    picker |                                [None,float,boolean,callable] |
|                  position |                                                        (x,y) |
|                  rotation |             [ angle in degrees `'vertical'` , `'horizontal'` |
|          size or fontsize | [ size in points , relative size, e.g., `'smaller'`, `'x-large'` ] |
|        style or fontstyle |                     [ `'normal'` , `'italic'` , `'oblique'`] |
|                      text |            string or anything printable with '%s' conversion |
|                 transform |               a matplotlib.transform transformation instance |
|                   variant |                              [ `'normal'` , `'small-caps'` ] |
|   verticalalignment or va |         [ `'center'` , `'top'` , `'bottom'` , `'baseline'` ] |
|                   visible |                                               [True , False] |
|      weight or fontweight | [ `'normal'` , `'bold'` , `'heavy'` , `'light'` , `'ultrabold'` , `'ultralight'`] |
|                         x |                                                        float |
|                         y |                                                        float |
|                    zorder |                                                   any number |

其中 `va`, `ha`, `multialignment` 可以用来控制布局。

- `horizontalalignment` or `ha` ：x 位置参数表示的位置
- `verticalalignment` or `va`：y 位置参数表示的位置
- `multialignment`：多行位置控制

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# build a rectangle in axes coords
left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height

fig = plt.figure(figsize=(10,7))
ax = fig.add_axes([0,0,1,1])

# axes coordinates are 0,0 is bottom left and 1,1 is upper right
p = patches.Rectangle(
    (left, bottom), width, height,
    fill=False, transform=ax.transAxes, clip_on=False
    )

ax.add_patch(p)

ax.text(left, bottom, 'left top',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes,
        size='xx-large')

ax.text(left, bottom, 'left bottom',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes,
        size='xx-large')

ax.text(right, top, 'right bottom',
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes,
        size='xx-large')

ax.text(right, top, 'right top',
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax.transAxes,
        size='xx-large')

ax.text(right, bottom, 'center top',
        horizontalalignment='center',
        verticalalignment='top',
        transform=ax.transAxes,
        size='xx-large')

ax.text(left, 0.5*(bottom+top), 'right center',
        horizontalalignment='right',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes,
        size='xx-large')

ax.text(left, 0.5*(bottom+top), 'left center',
        horizontalalignment='left',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes,
        size='xx-large')

ax.text(0.5*(left+right), 0.5*(bottom+top), 'middle',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=20, color='red',
        transform=ax.transAxes)

ax.text(right, 0.5*(bottom+top), 'centered',
        horizontalalignment='center',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes,
        size='xx-large')

ax.text(left, top, 'rotated\nwith newlines',
        horizontalalignment='center',
        verticalalignment='center',
        rotation=45,
        transform=ax.transAxes,
        size='xx-large')

ax.set_axis_off()
plt.show()
```

<img src="C:\Users\10277\AppData\Roaming\Typora\typora-user-images\image-20211108010347918.png" alt="image-20211108010347918" style="zoom:50%;" />

### 05.04 处理文本（数学表达式）

在字符串中使用一对 `$$` 符号可以利用 `Tex` 语法打出数学表达式，而且并不需要预先安装 `Tex`。在使用时我们通常加上 `r` 标记表示它是一个原始字符串（raw string）

#### 上下标

使用 `_` 和 `^` 表示上下标：

<img src="C:\Users\10277\AppData\Roaming\Typora\typora-user-images\image-20211108010705203.png" alt="image-20211108010705203" style="zoom: 50%;" />

注：

- 希腊字母和特殊符号可以用 '\ + 对应的名字' 来显示
- `{}` 中的内容属于一个部分；要打出花括号是需要使用 `\{\}`

#### 分数，二项式系数，stacked numbers

<img src="C:\Users\10277\AppData\Roaming\Typora\typora-user-images\image-20211108010750546.png" alt="image-20211108010750546" style="zoom:50%;" />

在 Tex 语言中，括号始终是默认的大小，如果要使括号大小与括号内部的大小对应，可以使用 `\left` 和 `\right` 选项：

<img src="C:\Users\10277\AppData\Roaming\Typora\typora-user-images\image-20211108010853519.png" alt="image-20211108010853519" style="zoom:50%;" />

#### 根号

<img src="C:\Users\10277\AppData\Roaming\Typora\typora-user-images\image-20211108010923326.png" alt="image-20211108010923326" style="zoom:50%;" />

#### 特殊字体

默认显示的字体是斜体，不过可以使用以下方法显示不同的字体：

|                  命令 |                   显示 |
| --------------------: | ---------------------: |
|        \mathrm{Roman} |             RomanRoman |
|       \mathit{Italic} |           ItalicItalic |
|   \mathtt{Typewriter} |   𝚃𝚢𝚙𝚎𝚠𝚛𝚒𝚝𝚎𝚛Typewriter |
| \mathcal{CALLIGRAPHY} | CALLIGRAPHY |
|   \mathbb{blackboard} |   𝕓𝕝𝕒𝕔𝕜𝕓𝕠𝕒𝕣𝕕blackboard |
|    \mathfrak{Fraktur} |         𝔉𝔯𝔞𝔨𝔱𝔲𝔯Fraktur |
|    \mathsf{sansserif} |              𝗌𝖺𝗇𝗌𝗌𝖾𝗋𝗂𝖿 |

#### 特殊字符表

参见：http://matplotlib.org/users/mathtext.html#symbols

### <u>05.05 图像基础</u>

导入相应的包：

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
%matplotlib inline
```

#### 导入图像

 `matplotlib` 默认只支持 `PNG` 格式的图像，我们可以使用 `mpimg.imread` 方法读入这幅图像：

```python
img = mpimg.imread('stinkbug.png')
img.shape	#(375, 500, 3)
```

这是一个 `375 x 500 x 3` 的 `RGB` 图像，并且每个像素使用 uint8 分别表示 `RGB` 三个通道的值。不过在处理的时候，`matplotlib` 将它们的值归一化到 `0.0~1.0` 之间。

#### 显示图像

使用 `plt.imshow()` 可以显示图像：

```python
imgplot = plt.imshow(img)
```

<img src="C:\Users\10277\AppData\Roaming\Typora\typora-user-images\image-20211108011449479.png" alt="image-20211108011449479" style="zoom:33%;" />

#### 伪彩色图像

从单通道模拟彩色图像：

```python
lum_img = img[:,:,0]imgplot = plt.imshow(lum_img)
```

<img src="C:\Users\10277\AppData\Roaming\Typora\typora-user-images\image-20211108011506433.png" alt="image-20211108011506433" style="zoom:33%;" />

#### 改变 colormap

```python
imgplot = plt.imshow(lum_img)imgplot.set_cmap('hot')
```

#### 限制显示范围

先查看直方图：

```
plt.hist(lum_img.flatten(), 256, range=(0.0,1.0), fc='k', ec='k')plt.show()
```

#### resize 操作

### 05.06 注释

### 05.07 标签

### 05.08 figures, subplots, axes 和 ticks 对象

### 05.09 不要迷信默认设置

### 05.10 各种绘图实例



## 06 面向对象编程

### 06.01 简介

#### 属性 attributes

属性是与对象绑定的一组数据，可以只读，只写，或者读写，使用时不加括号

#### 方法 method

方法是与属性绑定的一组函数，需要使用括号，作用于对象本身

#### 什么是对象？

python中除了一些保留的关键词（如if、for）之外，几乎都是对象。整数、函数等都是对象。



### 06.02 使用 OOP 对森林火灾建模

#### 对森林建模

##### 随机生长

- 在原来的基础上,我们要先让树生长，即定义 `grow_trees()` 方法
- 定义方法之前，我们要先指定两个属性：
  - 每个位置随机生长出树木的概率
  - 每个位置随机被闪电击中的概率
- 为了方便，我们定义一个辅助函数来生成随机 `bool` 矩阵，大小与森林大小一致
- 按照给定的生长概率生成生长的位置，将 `trees` 中相应位置设为 `True`

##### 火灾模拟

- 定义`start_fires()`：
  - 按照给定的概率生成被闪电击中的位置
  - 如果闪电击中的位置有树，那么将其设为着火点
- 定义`burn_trees()`：
  - **如果一棵树的上下左右有火，那么这棵树也会着火**
- 定义`advance_one_step()`：
  - 进行一次生长，起火，燃烧

```python
class Forest(object):
    def __init__(self, size=(150, 150), p_sapling=0.0025, p_lightning=5.e-6, name=None):
        self.size = size
        self.trees = np.zeros(self.size, dtype=bool)
        self.forest_fires = np.zeros(self.size, dtype=bool)
        self.p_sapling = p_sapling
        self.p_lightning = p_lightning
        if name is not None:
            self.name = name
        else:
            self.name = self.__class__.__name__

    @property		#森林大小的属性
    def num_cells(self):
        return self.size[0] * self.size[1]

    @property		#树的增长速率（树的数量除以森林大小）
    def tree_fraction(self):
        return self.trees.sum() / float(self.num_cells)

    @property		#火的增长速率
    def fire_fraction(self):
        return self.forest_fires.sum() / float(self.num_cells)

    def advance_one_step(self):
        self.grow_trees()
        self.start_fires()
        self.burn_trees()

    def grow_trees(self):
        growth_sites = self._rand_bool(self.p_sapling)
        self.trees[growth_sites] = True

    def start_fires(self):
        lightning_strikes = (self._rand_bool(self.p_lightning) & 
            self.trees)
        self.forest_fires[lightning_strikes] = True
        
    def burn_trees(self):
        fires = np.zeros((self.size[0] + 2, self.size[1] + 2), dtype=bool)
        fires[1:-1, 1:-1] = self.forest_fires
        north = fires[:-2, 1:-1]
        south = fires[2:, 1:-1]
        east = fires[1:-1, :-2]
        west = fires[1:-1, 2:]
        # 一棵树只要其上下左右有一个位置着火了，这棵树也就着火了
        new_fires = (north | south | east | west) & self.trees
        self.trees[self.forest_fires] = False
        self.forest_fires = new_fires

    def _rand_bool(self, p):
        return np.random.uniform(size=self.trees.shape) < p
```

定义一个森林类之后，我们创建一个新的森林类对象：

```
forest = Forest()#显示当前的状态：print(forest.trees)		#[[False False False ..., False False False]...print(forest.forest_fires)	##[[False False False ..., False False False]...
```



### 06.03 定义 class

#### 基本形式

`class` 定义如下：

```python
class ClassName(ParentClass):    
    """class docstring"""    
    def method(self):        
        return
```

- `class` 关键词在最前面
- `ClassName` 通常采用 `CamelCase` 记法
- 括号中的 `ParentClass` 用来表示继承关系
- 冒号不能缺少
- `""""""` 中的内容表示 `docstring`，可以省略
- 方法定义与函数定义十分类似，不过多了一个 `self` 参数表示这个对象本身
- `class` 中的方法要进行缩进

**查看帮助信息**

```python
class Forest(object):    
    """ Forest can grow trees which eventually die."""    
    pass
```

其中 `object` 是最基本的类型。查看帮助：

```python
import numpy as np
np.info(Forest)
#Forest()
##Forest can grow trees which eventually die.
```

#### 添加方法和属性

可以直接从对象出发添加属性，但是这样只会在此对象中存在该属性：

```python
forest = Forest()
forest.trees = np.zeros((150, 150), dtype=bool)
forest.trees
#array([[False, False, False, ..., False, False, False],....forest2.trees#AttributeError: 'Forest' object has no attribute 'trees'
```

**添加方法时，默认第一个参数是对象本身，一般为 `self`，可能用到也可能用不到，然后才是其他的参数**：

```python
class Forest(object):    
    """ Forest can grow trees which eventually die."""    
    def grow(self):        
        print("the tree is growing!")            
    def number(self, num=1):        
        if num == 1:            
            print('there is 1 tree.')        
        else:            
            print('there are', num, 'trees.')
```



### 06.04 特殊方法

**Python** 使用 `__` 开头的名字来定义特殊的方法和属性，它们有：

- `__init__()`
- `__repr__()`
- `__str__()`
- `__call__()`
- `__iter__()`
- `__add__()`
- `__sub__()`
- `__mul__()`
- `__rmul__()`
- `__class__`
- `__name__`

#### 构造方法 `__init__()`

之前说到，**在产生对象之后，我们可以向对象中添加属性。事实上，还可以通过构造方法，在构造对象的时候直接添加属性**：

```python
class Leaf(object):    
    """    A leaf falling in the woods.    """    
    def __init__(self, color='green'):        
        self.color = color
```

**添加属性的方法**：

```python
#默认属性值：
leaf1 = Leaf()
print(leaf1.color)
#传入有参数的值：
leaf2 = Leaf('orange')
print(leaf2.color)
```

事实上，`__new__()` 才是真正产生新对象的方法，`__init__()` 只是**对对象进行了初始化**，所以：

```python
leaf = Leaf()
```

相当于

```python
my_new_leaf = Leaf.__new__(Leaf)
Leaf.__init__(my_new_leaf)
leaf = my_new_leaf
```

#### 表示方法 `__repr__()` 和 `__str__()`

```python
class Leaf(object):    
    """    A leaf falling in the woods.    """    
    def __init__(self, color='green'):        
        self.color = color    
    def __str__(self):        
        "This is the string that is printed."        
        return "A {} leaf".format(self.color)    
    def __repr__(self):        
        "This string recreates the object."        
        return "{}(color='{}')".format(self.__class__.__name__, self.color)
```

`__str__()` 是**使用 `print` 函数显示的结果**：

```python
leaf = Leaf()
print(leaf)
#A green leaf
```

`__repr__()` 返回的是**不使用 `print` 方法的结果**：

```python
leaf
#Leaf(color='green')
```



### 06.05 属性

#### 只读属性

只读属性，顾名思义，指的是只可读不可写的属性，之前我们定义的属性都是可读可写的，对于**只读属性，我们需要使用 `@property` 修饰符**来得到：

```python
class Leaf(object):
    def __init__(self, mass_mg):
        self.mass_mg = mass_mg
    
    # 这样 mass_oz 就变成属性了
    @property
    def mass_oz(self):
        return self.mass_mg * 3.53e-5
    
leaf = Leaf(200)
print(leaf.mass_oz)		#0.00706
```

这里 `mass_oz` 就是一个只读不写的属性（注意是属性不是方法），而 `mass_mg` 是可读写的属性。

**注意三点：**

```python
#是属性不是方法
leaf.mass_oz()	#会报错
#TypeError: 'float' object is not callable

#是只读属性，不可写：
leaf.mass_oz = 0.001	#会报错
#AttributeError: can't set attribute

#可以修改 mass_mg 属性来改变 mass_oz
leaf.mass_mg = 150
print(leaf.mass_oz)
#输出为：0.005295
```

#### 可读写属性

对于 `@property` 生成的只读属性，我们可以**使用相应的 `@attr.setter` 修饰符来使得这个属性变成可写**的：

```python
class Leaf(object):
    def __init__(self, mass_mg):
        self.mass_mg = mass_mg
    
    # 这样 mass_oz 就变成属性了
    @property
    def mass_oz(self):
        return self.mass_mg * 3.53e-5
    
    # 使用 mass_oz.setter 修饰符
    @mass_oz.setter
    def mass_oz(self, m_oz):
        self.mass_mg = m_oz / 3.53e-5
        
leaf = Leaf(200)
print(leaf.mass_oz)		#0.00706

leaf.mass_mg = 150
print(leaf.mass_oz)		#0.005295

leaf.mass_oz = 0.01		#相当于给第二个mass_oz传参，m_oz=0.01
print(leaf.mass_mg)		#283.28611898
```

一个等价的替代如下：

```python
class Leaf(object):    
    def __init__(self, mass_mg):        
        self.mass_mg = mass_mg    
    def get_mass_oz(self):        
        return self.mass_mg * 3.53e-5    
    def set_mass_oz(self, m_oz):        
        self.mass_mg = m_oz / 3.53e-5    
        mass_oz = property(get_mass_oz, set_mass_oz)
```



### 06.06 继承

一个继承类定义的基本形式如下：

```python
class ClassName(ParentClass):    
    """class docstring"""    
    def method(self):        
        return
```

在里面有一个 `ParentClass` 项，用来进行继承，被继承的类是父类，定义的这个类是子类。 **对于子类来说，继承意味着它可以使用所有父类的方法和属性，同时还可以定义自己特殊的方法和属性。**

如果想对父类的方法进行修改，只需要**在子类中重定义这个类**即可：

```python
class Leaf(object):    
    def __init__(self, color="green"):        
        self.color = color    
    def fall(self):        
        print("Splat!")
        
class MapleLeaf(Leaf):    
    def change_color(self):        
        if self.color == "green":            
            self.color = "red"    
    def fall(self):        
        self.change_color()        
        print("Plunk!")
```



### 06.07 super() 函数及改写森林火灾模拟

`super(CurrentClassName, instance)`	**返回该类实例对应的父类对象**。

```python
class Leaf(object):
    def __init__(self, color="green"):
        self.color = color
    def fall(self):
        print("Splat!")

class MapleLeaf(Leaf):
    def change_color(self):
        if self.color == "green":
            self.color = "red"
    def fall(self):
        self.change_color()
        super(MapleLeaf, self).fall()
```

这里，我们先改变树叶的颜色，然后再找到这个实例对应的父类，并调用父类的 `fall()` 方法：

```python
mleaf = MapleLeaf()

print(mleaf.color)		#green
mleaf.fall()			#Splat!
print(mleaf.color)		#red
```

#### 使用继承重写森林火灾模拟

将森林 `Forest` 作为父类，并定义一个子类 `BurnableForest`

```python
import numpy as np

class Forest(object):
    """ Forest can grow trees which eventually die."""
    def __init__(self, size=(150,150), p_sapling=0.0025):
        self.size = size
        self.trees = np.zeros(self.size, dtype=bool)
        self.p_sapling = p_sapling
        
    def __repr__(self):
        my_repr = "{}(size={})".format(self.__class__.__name__, self.size)
        return my_repr
    
    def __str__(self):
        return self.__class__.__name__
    
    @property
    def num_cells(self):
        """Number of cells available for growing trees"""
        return np.prod(self.size)
    
    @property
    def tree_fraction(self):
        """
        Fraction of trees
        """
        num_trees = self.trees.sum()
        return float(num_trees) / self.num_cells
    
    def _rand_bool(self, p):
        """
        Random boolean distributed according to p, less than p will be True
        """
        return np.random.uniform(size=self.trees.shape) < p
    
    def grow_trees(self):
        """
        Growing trees.
        """
        growth_sites = self._rand_bool(self.p_sapling)
        self.trees[growth_sites] = True    
        
    def advance_one_step(self):
        """
        Advance one step
        """
        self.grow_trees()
```

#### 子类定义

- 将与燃烧相关的属性都被转移到了子类中去。
- 修改两类的构造方法，将闪电概率放到子类的构造方法上，同时在子类的构造方法中，用 `super` 调用父类的构造方法。
- 修改 `advance_one_step()`，父类中只进行生长，**在子类中用 `super` 调用父类的 `advance_one_step()` 方法，并添加燃烧的部分。**

```python
class BurnableForest(Forest):
    """
    Burnable forest support fires
    """    
    def __init__(self, p_lightning=5.0e-6, **kwargs):
        super(BurnableForest, self).__init__(**kwargs)
        self.p_lightning = p_lightning        
        self.fires = np.zeros((self.size), dtype=bool)
    
    def advance_one_step(self):
        """
        Advance one step
        """
        super(BurnableForest, self).advance_one_step()
        self.start_fires()
        self.burn_trees()
        
    @property
    def fire_fraction(self):
        """
        Fraction of fires
        """
        num_fires = self.fires.sum()
        return float(num_fires) / self.num_cells
    
    def start_fires(self):
        """
        Start of fire.
        """
        lightning_strikes = (self._rand_bool(self.p_lightning) & 
            self.trees)
        self.fires[lightning_strikes] = True
        
    def burn_trees(self):
        """
        Burn trees.
        """
        fires = np.zeros((self.size[0] + 2, self.size[1] + 2), dtype=bool)
        fires[1:-1, 1:-1] = self.fires
        north = fires[:-2, 1:-1]
        south = fires[2:, 1:-1]
        east = fires[1:-1, :-2]
        west = fires[1:-1, 2:]
        new_fires = (north | south | east | west) & self.trees
        self.trees[self.fires] = False
        self.fires = new_fires
```



### 06.08 重定义森林火灾模拟

在前面的例子中，我们定义了一个 `BurnableForest`，实现了一个循序渐进的生长和燃烧过程。

假设我们现在想要定义一个立即燃烧的过程（每次着火之后燃烧到不能燃烧为止，之后再生长，而不是每次只燃烧周围的一圈树木），由于燃烧过程不同，我们需要从 `BurnableForest` 中派生出两个新的子类 `SlowBurnForest`（原来的燃烧过程） 和 `InsantBurnForest`，为此

- 将 `BurnableForest` 中的 `burn_trees()` 方法改写，不做任何操作，直接 `pass`（因为在 `advance_one_step()` 中调用了它，所以不能直接去掉）
- 在两个子类中定义新的 `burn_trees()` 方法。

```python
import numpy as npfrom scipy.ndimage.measurements 
import labelclass 
Forest(object):    
    """ Forest can grow trees which eventually die."""    
    def __init__(self, size=(150,150), p_sapling=0.0025):        
        self.size = size        
        self.trees = np.zeros(self.size, dtype=bool)        
        self.p_sapling = p_sapling            
    def __repr__(self):        
        my_repr = "{}(size={})".format(self.__class__.__name__, self.size)        
        return my_repr        
    def __str__(self):        
        return self.__class__.__name__        
    @property    
    def num_cells(self):        
        """Number of cells available for growing trees"""        
        return np.prod(self.size)        
    @property    
    def tree_fraction(self):        
        """        Fraction of trees        """        
        num_trees = self.trees.sum()        
        return float(num_trees) / self.num_cells        
    def _rand_bool(self, p):        
        """        Random boolean distributed according to p, less than p will be True        """        
        return np.random.uniform(size=self.trees.shape) < p        
    def grow_trees(self):        
        """        Growing trees.        """        
        growth_sites = self._rand_bool(self.p_sapling)        
        self.trees[growth_sites] = True                
    def advance_one_step(self):        
        """        Advance one step        """
        self.grow_trees()
class BurnableForest(Forest):    
    """    Burnable forest support fires    """        
    def __init__(self, p_lightning=5.0e-6, **kwargs):        
        super(BurnableForest, self).__init__(**kwargs)        
        self.p_lightning = p_lightning                
        self.fires = np.zeros((self.size), dtype=bool)        
    def advance_one_step(self):        
        """        Advance one step        """        
        super(BurnableForest, self).advance_one_step()        
        self.start_fires()        
        self.burn_trees()            
    @property    
    def fire_fraction(self):        
        """        Fraction of fires        """        
        num_fires = self.fires.sum()        
        return float(num_fires) / self.num_cells        
    def start_fires(self):        
        """        Start of fire.        """        
        lightning_strikes = (self._rand_bool(self.p_lightning) & self.trees)        
        self.fires[lightning_strikes] = True        
    def burn_trees(self):            
        pass    
class SlowBurnForest(BurnableForest):    
    def burn_trees(self):       
        """        Burn trees.        """        
        fires = np.zeros((self.size[0] + 2, self.size[1] + 2), dtype=bool)        
        fires[1:-1, 1:-1] = self.fires        
        north = fires[:-2, 1:-1]        
        south = fires[2:, 1:-1]        
        east = fires[1:-1, :-2]        
        west = fires[1:-1, 2:]        
        new_fires = (north | south | east | west) & self.trees        
        self.trees[self.fires] = False        
        self.fires = new_firesclass InstantBurnForest(BurnableForest):    
    def burn_trees(self):        
        # 起火点        
        strikes = self.fires        
        # 找到连通区域        
        groves, num_groves = label(self.trees)        
        fires = set(groves[strikes])        
        self.fires.fill(False)        
        # 将与着火点相连的区域都烧掉        
        for fire in fires:            
            self.fires[groves == fire] = True        
            self.trees[self.fires] = False        
            self.fires.fill(False)
```



### 06.09 接口

接口只是定义了一些方法，而没有去实现，多用于程序设计时，只是设计需要有什么样的功能，但是并没有实现任何功能，这些功能需要被另一个类（B）继承后，由 类B去实现其中的某个功能或全部功能。

在python中接口由抽象类和抽象方法去实现，接口是不能被实例化的，只能被别的类继承去实现相应的功能。

接口在python中并没有那么重要，因为如果要继承接口，需要把其中的每个方法全部实现，否则会报编译错误，还不如直接定义一个class，其中的**方法实现全部为pass**，让子类重写这些函数。


### 06.10 共有，私有和特殊方法和属性

- 我们之前已经见过 `special` 方法和属性，即以 `__` 开头和结尾的方法和属性
- 私有方法和属性，以 `_` 开头，不过不是真正私有，而是可以调用的，但是不会被代码自动完成所记录（即 Tab 键之后不会显示）
- 其他都是共有的方法和属性
- 以 `__` 开头不以 `__` 结尾的属性是更加特殊的方法，调用方式也不同：

```python
class MyClass(object):    
    def __init__(self):        
        print("I'm special!")    
    def _private(self):        
        print("I'm private!")    
    def public(self):        
        print("I'm public!")    
    def __really_special(self):        
        print("I'm really special!")        
m = MyClass()					
#I'm special!
m.public()						
#m.public()
m._private()					
#I'm private!
m._MyClass__really_special()	
#I'm really special!
```



### 06.11 多重继承

多重继承，指的是**一个类别可以同时从多于一个父类继承行为与特征的功能**，`Python` 是支持多重继承的：

```python
class Leaf(object):    
    def __init__(self, color='green'):        
        self.color = colorclass ColorChangingLeaf(Leaf):    
    def change(self, new_color='brown'):        
        self.color = new_color    
    def fall(self):        
        print("Spalt!")
class DeciduousLeaf(Leaf):    
    def fall(self):        
        print("Plunk!")
class MapleLeaf(ColorChangingLeaf, DeciduousLeaf):    
    pass
```

在上面的例子中， `MapleLeaf` 就使用了多重继承，它可以使用两个父类的方法：

```python
leaf = MapleLeaf()
leaf.change("yellow")
print(leaf.color)	
#yellow
leaf.fall()			
#Plunk!
```

> 如果同时实现了不同的接口，那么，**最后使用的方法以继承的顺序为准，放在前面的优先继承**

事实上，这个顺序可以通过该类的 `__mro__` 属性或者 `mro()` 方法来查看：

```python
MapleLeaf.__mro__
#(__main__.MapleLeaf,
# __main__.ColorChangingLeaf,
# __main__.DeciduousLeaf,
# __main__.Leaf,
# object)
MapleLeaf.mro()		#输出同样内容
```

考虑更复杂的例子：

```python
class A(object):    
    pass
class B(A):    
    pass
class C(A):    
    pass
class C1(C):    
    pass
class B1(B):    
    pass
class D(B1, C):    
    pass
```

调用顺序：

```python
D.mro()
#[__main__.D, __main__.B1, __main__.B, __main__.C, __main__.A, object]
```



## 07 pandas

### 07.01 十分钟上手Pandas

`pandas` 是一个 `Python Data Analysis Library`。

```python
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

####  Pandas 对象

`pandas` 中有三种基本结构：

- `Series`
  - 1D labeled homogeneously-typed array
- `DataFrame`
  - General 2D labeled, size-mutable tabular structure with potentially heterogeneously-typed columns
- `Panel`
  - General 3D labeled, also size-mutable array

#### Series

一维 `Series` 可以用一维列表初始化：

```python
s = pd.Series([1,3,5,np.nan,6,8])
s
#0    1.0
#1    3.0
#2    5.0
#3    NaN
#4    6.0
#5    8.0
#dtype: float64
```

默认情况下，`Series` 的下标都是数字（可以使用额外参数指定），类型是统一的。

#### DataFrame

`DataFrame` 则是个二维结构，这里首先构造一组时间序列，作为我们第一维的下标：

```python
dates = pd.date_range('20210101', periods=6)
dates
#DatetimeIndex(['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04',
#               '2021-01-05', '2021-01-06'],
#              dtype='datetime64[ns]', freq='D')
```

然后创建一个 `DataFrame` 结构：

```python
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD')
df
#				A			B			C			D
2013-01-01	-0.605936	-0.861658	-1.001924	1.528584
2013-01-02	-0.165408	0.388338	1.187187	1.819818
2013-01-03	0.065255	-1.608074	-1.282331	-0.286067
2013-01-04	1.289305	0.497115	-0.225351	0.040239
2013-01-05	0.038232	0.875057	-0.092526	0.934432
2013-01-06	-2.163453	-0.010279	1.699886	1.291653                  
```

默认情况下，如果不指定 `index` 参数和 `columns`，那么他们的值将用从 `0` 开始的数字替代。

除了向 `DataFrame` 中传入**二维数组**，我们也可以**使用字典传入数据**：

```python
df2 = pd.DataFrame({'A' : 1.,
                    'B' : pd.Timestamp('20130102'),
                    'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                    'D' : np.array([3] * 4,dtype='int32'),
                    'E' : pd.Categorical(["test","train","test","train"]),
                    'F' : 'foo' })
df2
#	A		B		C	D	E		F
0	1	2013-01-02	1	3	test	foo
1	1	2013-01-02	1	3	train	foo
2	1	2013-01-02	1	3	test	foo
3	1	2013-01-02	1	3	train	foo
```

字典的每个 `key` 代表一列，其 `value` 可以是各种能够转化为 `Series` 的对象。

与 `Series` 要求所有的类型都一致不同，**`DataFrame` 值要求每一列数据的格式相同**

#### 查看数据

**头尾数据**

`head` 和 `tail` 方法可以分别查看最前面几行和最后面几行的数据（默认为 5）

```python
df.head()

df.tail(3)
```

**下标，列标，数据**

下标使用 `index` 属性查看：

```python
df.index
#DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
#               '2013-01-05', '2013-01-06'],
#              dtype='datetime64[ns]', freq='D')
```

列标使用 `columns` 属性查看：

```python
df.columns
#Index([u'A', u'B', u'C', u'D'], dtype='object')
```

数据值使用 `values` 查看：

```python
df.values
#array([[-0.60593585, -0.86165752, -1.00192387,  1.52858443],
#       [-0.16540784,  0.38833783,  1.18718697,  1.81981793],
#       [ 0.06525454, -1.60807414, -1.2823306 , -0.28606716],
#       [ 1.28930486,  0.49711531, -0.22535143,  0.04023897],
#       [ 0.03823179,  0.87505664, -0.0925258 ,  0.93443212],
#       [-2.16345271, -0.01027865,  1.69988608,  1.29165337]])
```

**统计数据**

查看简单的统计数据：

```python
df.describe()
#			A			B			C			D
count	6.0000	6.000000	6.000000	6.000000
mean	-0.257001	-0.119917	0.047490	0.888110
std	1.126657	0.938705	1.182629	0.841529
min	-2.163453	-1.608074	-1.282331	-0.286067
25%	-0.495804	-0.648813	-0.807781	0.263787
50%	-0.063588	0.189030	-0.158939	1.113043
75%	0.058499	0.469921	0.867259	1.469352
max	1.289305	0.875057	1.699886	1.819818
```

**转置**

```python
df.T
```

#### 排序

`sort_index(axis=0, ascending=True)` 方法按照下标大小进行排序，`axis=0` 表示按第 0 维进行排序。

```python
df.sort_index(ascending=False)
#			A				B			C			D
2013-01-06	-2.163453	-0.010279	1.699886	1.291653
2013-01-05	0.038232	0.875057	-0.092526	0.934432
2013-01-04	1.289305	0.497115	-0.225351	0.040239
2013-01-03	0.065255	-1.608074	-1.282331	-0.286067
2013-01-02	-0.165408	0.388338	1.187187	1.819818
2013-01-01	-0.605936	-0.861658	-1.001924	1.528584

df.sort_index(axis=1, ascending=False)
#				D			C			B			A
2013-01-01	1.528584	-1.001924	-0.861658	-0.605936
2013-01-02	1.819818	1.187187	0.388338	-0.165408
2013-01-03	-0.286067	-1.282331	-1.608074	0.065255
2013-01-04	0.040239	-0.225351	0.497115	1.289305
2013-01-05	0.934432	-0.092526	0.875057	0.038232
2013-01-06	1.291653	1.699886	-0.010279	-2.163453
```

`sort_values(by, axis=0, ascending=True)` 方法按照 `by` 的值的大小进行排序，例如按照 `B` 列的大小：

```python
df.sort_values(by="B")
#				A			B			C			D
2013-01-03	0.065255	-1.608074	-1.282331	-0.286067
2013-01-01	-0.605936	-0.861658	-1.001924	1.528584
2013-01-06	-2.163453	-0.010279	1.699886	1.291653
2013-01-02	-0.165408	0.388338	1.187187	1.819818
2013-01-04	1.289305	0.497115	-0.225351	0.040239
2013-01-05	0.038232	0.875057	-0.092526	0.934432
```

#### **索引**

虽然 `DataFrame` 支持 `Python/Numpy` 的索引语法，但是推荐使用 `.at, .iat, .loc, .iloc 和 .ix` 方法进行索引。

**读取数据**

选择单列数据：

```python
df["A"]
#2013-01-01   -0.605936
#2013-01-02   -0.165408
#2013-01-03    0.065255
#2013-01-04    1.289305
#2013-01-05    0.038232
#2013-01-06   -2.163453
#Freq: D, Name: A, dtype: float64

df.A	#也可以用 df.A
```

使用切片读取多行

```python
df[0:3]
```

`index` 名字也可以进行切片：

```python
df["20130101":"20130103"]
#				A			B			C			D
#2013-01-01	-0.605936	-0.861658	-1.001924	1.528584
#2013-01-02	-0.165408	0.388338	1.187187	1.819818
#2013-01-03	0.065255	-1.608074	-1.282331	-0.286067
```

##### **使用 `label` 索引**

`loc` 可以方便的使用 `label` 进行索引：

```python
df.loc[dates[0]]
#A   -0.605936
#B   -0.861658
#C   -1.001924
#D    1.528584
#Name: 2013-01-01 00:00:00, dtype: float64
```

多列数据：

```python
df.loc[:,['A','B']]
#				A			B
#2013-01-01	-0.605936	-0.861658
#2013-01-02	-0.165408	0.388338
#2013-01-03	0.065255	-1.608074
#2013-01-04	1.289305	0.497115
#2013-01-05	0.038232	0.875057
#2013-01-06	-2.163453	-0.010279
```

选择多行多列：

```python
df.loc['20130102':'20130104',['A','B']]
```

数据降维：

```python
df.loc['20130102',['A','B']]
#得到标量值：
df.loc[dates[0],'B']
```

得到标量值可以用 `at`，速度更快：

```python
%timeit -n100 df.loc[dates[0],'B']
%timeit -n100 df.at[dates[0],'B']

print(df.at[dates[0],'B'])
#16 µs ± 3.75 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
#12.5 µs ± 837 ns per loop (mean ± std. dev. of 7 runs, 100 loops each)
#0.17455716005526253
```

##### **使用位置索引**

`iloc` 使用位置进行索引：

```python
df.iloc[3]
#A    1.289305
#B    0.497115
#C   -0.225351
#D    0.040239
#Name: 2013-01-04 00:00:00, dtype: float64
```

连续切片：

```python
df.iloc[3:5,0:2]
#				A			B
#2013-01-04	1.289305	0.497115
#2013-01-05	0.038232	0.875057
```

索引不连续的部分：

```python
df.iloc[[1,2,4],[0,2]]
#				A			C
#2013-01-02	-0.165408	1.187187
#2013-01-03	0.065255	-1.282331
#2013-01-05	0.038232	-0.092526
```

标量值：

```python
df.iloc[1,1]	#0.3883378290420279

#当然，使用 iat 索引标量值更快：
%timeit -n100 df.iloc[1,1]
%timeit -n100 df.iat[1,1]

df.iat[1,1]
#19.7 µs ± 3.67 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
#15.1 µs ± 668 ns per loop (mean ± std. dev. of 7 runs, 100 loops each)
#0.4176847733412457
```

##### **布尔型索引**

所有 `A` 列大于 0 的行：

```python
df[df.A > 0]
#				A			B			C			D
#2013-01-03	0.065255	-1.608074	-1.282331	-0.286067
#2013-01-04	1.289305	0.497115	-0.225351	0.040239
#2013-01-05	0.038232	0.875057	-0.092526	0.934432
```

只留下所有大于 0 的数值：

```python
df[df > 0]
#			A	B	C	D
#2013-01-01	NaN	NaN	NaN	1.528584
#2013-01-02	NaN	0.388338	1.187187	1.819818
#2013-01-03	0.065255	NaN	NaN	NaN
#2013-01-04	1.289305	0.497115	NaN	0.040239
#2013-01-05	0.038232	0.875057	NaN	0.934432
#2013-01-06	NaN	NaN	1.699886	1.291653
```

使用 `isin` 方法做 `filter` 过滤：

```python
df2 = df.copy()
df2['E'] = ['one', 'one','two','three','four','three']

df2
#				A			B			C			D		E
#2013-01-01	-0.605936	-0.861658	-1.001924	1.528584	one
#2013-01-02	-0.165408	0.388338	1.187187	1.819818	one
#2013-01-03	0.065255	-1.608074	-1.282331	-0.286067	two
#2013-01-04	1.289305	0.497115	-0.225351	0.040239	three
#2013-01-05	0.038232	0.875057	-0.092526	0.934432	four
#2013-01-06	-2.163453	-0.010279	1.699886	1.291653	three

df2[df2['E'].isin(['two','four'])]
#				A			B			C			D		E
#2013-01-03	0.065255	-1.608074	-1.282331	-0.286067	two
#2013-01-05	0.038232	0.875057	-0.092526	0.934432	four
```

#### 设定数据的值

```
s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))
```

像字典一样，直接指定 `F` 列的值为 `s1`，此时以 `df` 已有的 `index` 为标准将二者进行合并，`s1` 中没有的 `index` 项设为 `NaN`，多余的项舍去：

```python
df['F'] = s1
df
#				A			B			C			D		F
2013-01-01	-0.605936	-0.861658	-1.001924	1.528584	NaN
2013-01-02	-0.165408	0.388338	1.187187	1.819818	1
2013-01-03	0.065255	-1.608074	-1.282331	-0.286067	2
2013-01-04	1.289305	0.497115	-0.225351	0.040239	3
2013-01-05	0.038232	0.875057	-0.092526	0.934432	4
2013-01-06	-2.163453	-0.010279	1.699886	1.291653	5
```

或者使用 `at` 或 `iat` 修改单个值：

```
df.at[dates[0],'A'] = 0

df.iat[0, 1] = 0
```

设定一整列：

```python
df.loc[:,'D'] = np.array([5] * len(df))
```

设定满足条件的数值：

```python
df2 = df.copy()
df2[df2 > 0] = -df2
df2
#				A			B			C		D	F
2013-01-01	0.000000	0.000000	-1.001924	-5	NaN
2013-01-02	-0.165408	-0.388338	-1.187187	-5	-1
2013-01-03	-0.065255	-1.608074	-1.282331	-5	-2
2013-01-04	-1.289305	-0.497115	-0.225351	-5	-3
2013-01-05	-0.038232	-0.875057	-0.092526	-5	-4
2013-01-06	-2.163453	-0.010279	-1.699886	-5	-5
```

#### 缺失数据

```python
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1],'E'] = 1

df1
#				A			B			C		D	F	E
#2013-01-01	0.000000	0.000000	-1.001924	5	NaN	1
#2013-01-02	-0.165408	0.388338	1.187187	5	1	1
#2013-01-03	0.065255	-1.608074	-1.282331	5	2	NaN
#2013-01-04	1.289305	0.497115	-0.225351	5	3	NaN
```

丢弃所有缺失数据的行得到的新数据：

```python
df1.dropna(how='any')
#				A			B			C		D	F	E
#2013-01-02	-0.165408	0.388338	1.187187	5	1	1
```

填充缺失数据：

```
df1.fillna(value=5)
```

检查缺失数据的位置：

```python
pd.isnull(df1)
#				A		B		C		D		F		E
#2013-01-01	False	False	False	False	True	False
#2013-01-02	False	False	False	False	False	False
#2013-01-03	False	False	False	False	False	True
#2013-01-04	False	False	False	False	False	True
```

#### 计算操作

##### 统计信息

均值：

```python
#每一列的均值：
df.mean()
#A   -0.156012
#B    0.023693
#C    0.047490
#D    5.000000
#F    3.000000
#dtype: float64

#每一行的均值：
df.mean(1)
```

多个对象之间的操作，如果维度不对，`pandas` 会自动调用 `broadcasting` 机制：

```python
s = pd.Series([1,3,5,np.nan,6,8], index=dates).shift(2)
s
#2021-01-01    NaN
#2021-01-02    NaN
#2021-01-03    1.0
#2021-01-04    3.0
#2021-01-05    5.0
#2021-01-06    NaN
#Freq: D, dtype: float64
```

> 注：shift()函数是对数据进行移动操作，默认向后移动一个，空出来的用nan填充。

相减 `df - s`：

```python
df.sub(s, axis='index')
#			A	B	C	D	F
#2013-01-01	NaN	NaN	NaN	NaN	NaN
#2013-01-02	NaN	NaN	NaN	NaN	NaN
#2013-01-03	-0.934745	-2.608074	-2.282331	4	1
#2013-01-04	-1.710695	-2.502885	-3.225351	2	0
#2013-01-05	-4.961768	-4.124943	-5.092526	0	-1
#2013-01-06	NaN	NaN	NaN	NaN	NaN
```

##### apply 操作

与 `R` 中的 `apply` 操作类似，接收一个函数，默认是对将函数作用到每一列上：

```
df.apply(np.cumsum)
#				A			B			C		D	F
2013-01-01	0.000000	0.000000	-1.001924	5	NaN
2013-01-02	-0.165408	0.388338	0.185263	10	1
2013-01-03	-0.100153	-1.219736	-1.097067	15	3
2013-01-04	1.189152	-0.722621	-1.322419	20	6
2013-01-05	1.227383	0.152436	-1.414945	25	10
2013-01-06	-0.936069	0.142157	0.284941	30	15
```

> np.cumsum是做累加操作，默认按行来，参数为axis=1则是按列来

求每列最大最小值之差：

```python
df.apply(lambda x: x.max() - x.min())
#A    3.452758
#B    2.483131
#C    2.982217
#D    0.000000
#F    4.000000
#dtype: float64
```

##### 直方图

```python
s = pd.Series(np.random.randint(0, 7, size=10))
s
#0    2
#1    5
#2    6
#3    6
#4    6
#5    3
#6    5
#7    0
#8    4
#9    4
#dtype: int64

s.value_counts()
#6    3
#5    2
#4    2
#3    1
#2    1
#0    1
#dtype: int64

#绘制直方图:
h = s.hist()
```

<img src="C:\Users\10277\AppData\Roaming\Typora\typora-user-images\image-20211112103900727.png" alt="image-20211112103900727" style="zoom:50%;" />



##### 字符串方法

当 `Series` 或者 `DataFrame` 的某一列是字符串时，我们可以用 `.str` 对这个字符串数组进行字符串的基本操作：

```python
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])

s.str.lower()
```

#### 合并

##### 连接

```python
df = pd.DataFrame(np.random.randn(10, 4))
df
#		0			1			2			3
0	-2.346373	0.105651	-0.048027	0.010637
1	-0.682198	0.943043	0.147312	-0.657871
2	0.515766	-0.768286	0.361570	1.146278
3	-0.607277	-0.003086	-1.499001	1.165728
4	-1.226279	-0.177246	-1.379631	-0.639261
5	0.807364	-1.855060	0.325968	1.898831
6	0.438539	-0.728131	-0.009924	0.398360
7	1.497457	-1.506314	-1.557624	0.869043
8	0.945985	-0.519435	-0.510359	-1.077751
9	1.597679	-0.285955	-1.060736	0.608629
```

可以**使用 `pd.concat` 函数将多个 `pandas` 对象进行连接**：

```python
pieces = [df[:2], df[4:5], df[7:]]

pd.concat(pieces)
#		0			1			2			3
0	-2.346373	0.105651	-0.048027	0.010637
1	-0.682198	0.943043	0.147312	-0.657871
4	-1.226279	-0.177246	-1.379631	-0.639261
7	1.497457	-1.506314	-1.557624	0.869043
8	0.945985	-0.519435	-0.510359	-1.077751
9	1.597679	-0.285955	-1.060736	0.608629
```

##### 数据库中的 Join

`merge` 可以实现数据库中的 `join` 操作：

```python
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})

print(left)
print(right)
#   key  lval
#0  foo     1
#1  foo     2
#   key  rval
#0  foo     4
#1  foo     5

pd.merge(left, right, on='key')
#	key	lval	rval
#0	foo	1	4
#1	foo	1	5
#2	foo	2	4
#3	foo	2	5
```

##### append

向 `DataFrame` 中添加行：

```python
df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])
#将第三行的值添加到最后：
s = df.iloc[3]

df.append(s, ignore_index=True)
#		A			B			C			D
#0	1.587778	-0.110297	0.602245	1.212597
#1	-0.551109	0.337387	-0.220919	0.363332
#2	1.207373	-0.128394	0.619937	-0.612694
#3	-0.978282	-1.038170	0.048995	-0.788973
#4	0.843893	-1.079021	0.092212	0.485422
#5	-0.056594	1.831206	1.910864	-1.331739
#6	-0.487106	-1.495367	0.853440	0.410854
#7	1.830852	-0.014893	0.254025	0.197422
#8	-0.978282	-1.038170	0.048995	-0.788973
```

##### Grouping

```
df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
                          'foo', 'bar', 'foo', 'foo'],
                   'B' : ['one', 'one', 'two', 'three',
                          'two', 'two', 'one', 'three'],
                   'C' : np.random.randn(8),
                   'D' : np.random.randn(8)})
```

按照 `A` 的值进行分类：

```python
df.groupby('A').sum()
#			C		D		
#A
bar	-2.266021	-2.862813
foo	-0.027163	1.508287
```

按照 `A, B` 的值进行分类：

```python
df.groupby(['A', 'B']).sum()
```

<img src="C:\Users\10277\AppData\Roaming\Typora\typora-user-images\image-20211112105207351.png" alt="image-20211112105207351"  />



#### 改变形状

##### Stack

产生一个多 `index` 的 `DataFrame`：

```python
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                     'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two',
                     'one', 'two', 'one', 'two']]))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df
```

<img src="C:\Users\10277\AppData\Roaming\Typora\typora-user-images\image-20211112105710262.png" alt="image-20211112105710262" style="zoom:80%;" />

`stack` 方法将 `columns` 变成一个新的 `index` 部分：

```python
df2 = df[:4]
stacked = df2.stack()
stacked
#first  second   
#bar    one     A   -0.109174
#               B    0.958551
#       two     A   -0.254743
#               B   -0.975924
#baz    one     A   -0.132039
#               B   -0.119009
#       two     A    0.587063
#               B   -0.819037
#dtype: float64
```

可以**使用 `unstack()` 将最后一级 `index` 放回 `column`**：

```python
stacked.unstack()
```

<img src="C:\Users\10277\AppData\Roaming\Typora\typora-user-images\image-20211112105848377.png" alt="image-20211112105848377" style="zoom:80%;" />

也可以指定其他的级别：

```python
stacked.unstack(1)
```

<img src="C:\Users\10277\AppData\Roaming\Typora\typora-user-images\image-20211112110001724.png" alt="image-20211112110001724" style="zoom:80%;" />



#### 时间序列

金融分析中常用到时间序列数据：

```python
rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)
ts
#2012-03-06    1.096788
#2012-03-07    0.029678
#2012-03-08    0.511461
#2012-03-09   -0.332369
#2012-03-10    1.720321
#Freq: D, dtype: float64
```

标准时间表示：

```python
ts_utc = ts.tz_localize('UTC')
ts_utc
#2012-03-06 00:00:00+00:00    1.096788
#2012-03-07 00:00:00+00:00    0.029678
#2012-03-08 00:00:00+00:00    0.511461
#2012-03-09 00:00:00+00:00   -0.332369
#2012-03-10 00:00:00+00:00    1.720321
#Freq: D, dtype: float64
```

改变时区表示：

```python
ts_utc.tz_convert('US/Eastern')
#2012-03-05 19:00:00-05:00    1.096788
#2012-03-06 19:00:00-05:00    0.029678
#2012-03-07 19:00:00-05:00    0.511461
#2012-03-08 19:00:00-05:00   -0.332369
#2012-03-09 19:00:00-05:00    1.720321
#Freq: D, dtype: float64
```

#### 文件读写

##### csv

```python
#写入文件：
df.to_csv('foo.csv')

#从文件中读取：
pd.read_csv('foo.csv')
```

##### hdf5

```python
#写入文件：
df.to_hdf("foo.h5", "df")
#读取文件：
pd.read_hdf('foo.h5','df').head()
```

##### excel

```python
#写入文件：
df.to_excel('foo.xlsx', sheet_name='Sheet1')
#读取文件：
pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
```

### <u>07.02 一维数据结构：Series</u>

### <u>07.03 二维数据结构：DataFrame</u>

#### 索引和选择：

|                      Operation |        Syntax |    Result |
| -----------------------------: | ------------: | --------: |
|                  Select column |       df[col] |    Series |
|            Select row by label | df.loc[label] |    Series |
| Select row by integer location |  df.iloc[loc] |    Series |
|                     Slice rows |      df[5:10] | DataFrame |
|  Select rows by boolean vector |  df[bool_vec] | DataFrame |





## 08 其他小工具

### 08.01 pprint 模块：打印 Python 对象

`pprint` 是 pretty printer 的缩写，用来打印 Python 数据结构，与 `print` 相比，它打印出来的结构更加整齐，便于阅读。

```python
import pprint

#生成一个 Python 对象：
data = (
    "this is a string", 
    [1, 2, 3, 4], 
    ("more tuples", 1.0, 2.3, 4.5), 
    "this is yet another string"
    )

#使用普通的 print 函数：
print(data)
#('this is a string', [1, 2, 3, 4], ('more tuples', 1.0, 2.3, 4.5), 'this is yet another string')

#使用 pprint 模块中的 pprint 函数：
pprint.pprint(data)
#('this is a string',
# [1, 2, 3, 4],
# ('more tuples', 1.0, 2.3, 4.5),
# 'this is yet another string')
```

### 08.02 pickle, cPickle 模块：序列化 Python 对象

`pickle` 模块实现了一种算法，可以将任意一个 `Python` 对象转化为一系列的字节，也可以将这些字节重构为一个有相同特征的新对象。由于字节可以被传输或者存储，因此 **`pickle` 事实上实现了传递或者保存 `Python` 对象的功能。**

`cPickle` 使用 `C` 而不是 `Python` 实现了相同的算法，因此速度上要比 `pickle` 快一些。但是它不允许用户从 `pickle` 派生子类。如果子类对你的使用来说无关紧要，那么 `cPickle` 是个更好的选择。

```python
try:
    import cPickle as pickle
except:
    import pickle
```

pikle模块和json模块一样，都提供了dumps()、loads()、dump()、load()四种方法，且功能类似。

#### 编码和解码

使用 `pickle.dumps()` 可以将一个对象转换为字符串（`dump string`）：

```python
data = [ { 'a':'A', 'b':2, 'c':3.0 } ]

data_string = pickle.dumps(data)

print("DATA:")
print(data)
print("PICKLE:")
print(data_string)
#DATA:
#[{'a': 'A', 'b': 2, 'c': 3.0}]
#PICKLE:
#b'\x80\x04\x95#\x00\x00\x00\x00\x00\x00\x00]\x94}\x94(\x8c\x01a\x94\x8c\x01A\x94\x8c\x01b\x94K\x02\x8c\x01c\x94G@\x08\x00\x00\x00\x00\x00\x00ua.'
```

虽然 `pickle` 编码的字符串并不一定可读，但是我们可以用 `pickle.loads()` 来从这个字符串中恢复原对象中的内容（`load string`）：

```python
data_from_string = pickle.loads(data_string)

print data_from_string
#[{'a': 'A', 'b': 2, 'c': 3.0}]
```

#### 编码协议

`dumps` 可以接受一个可省略的 `protocol` 参数（默认为 0），目前有 5种编码方式：

当前最高级的编码可以通过 `HIGHEST_PROTOCOL` 查看：

```python
print(pickle.HIGHEST_PROTOCOL)		#5

#如果 protocol 参数指定为负数，那么将调用当前的最高级的编码协议进行编码：
print(pickle.dumps(data, -1))
#b'\x80\x05\x95#\x00\x00\x00\x00\x00\x00\x00]\x94}\x94(\x8c\x01a\x94\x8c\x01A\x94\x8c\x01b\x94K\x02\x8c\x01c\x94G@\x08\x00\x00\x00\x00\x00\x00ua.'
```

从这些格式中恢复对象时，不需要指定所用的协议，`pickle.load()` 会自动识别。

#### 存储和读取 pickle 文件

除了将对象转换为字符串这种方式，`pickle` 还支持将对象写入一个文件中，通常我们将这个文件命名为 `xxx.pkl`，以表示它是一个 `pickle` 文件：

存储和读取的函数分别为：

- `pickle.dump(obj, file, protocol=0)` 将对象序列化并存入 `file` 文件中
- `pickle.load(file)` 从 `file` 文件中的内容恢复对象

将对象存入文件：

```python
with open("data.pkl", "wb") as f:
    pickle.dump(data, f)
```

从文件中读取：

```python
with open("data.pkl") as f:
    data_from_file = pickle.load(f)
    
print(data_from_file)
#[{'a': 'A', 'c': 3.0, 'b': 2}]
```



### 08.03 json 模块：处理 JSON 数据

json模块可以用来处理json数据。模块中主要有四个方法：

- loads()：从字符串中读取json数据，但保存形式还是dict()或者list()这些python中的对象

- dumps()：将python对象转换成json对象

- load()：从json文件中读取数据

- dump() ：将数据存储到json文件

#### JSON 与 Python 的转换

假设我们已经将上面这个 `JSON` 对象写入了一个字符串：

```python
import json
from pprint import pprint

info_string = """
{
    "name": "echo",
    "age": 24,
    "coding skills": ["python", "matlab", "java", "c", "c++", "ruby", "scala"],
    "ages for school": { 
        "primary school": 6,
        "middle school": 9,
        "high school": 15,
        "university": 18
    },
    "hobby": ["sports", "reading"],
    "married": false
}
"""
```

我们可以用 `json.loads()` (load string) 方法从字符串中读取 `JSON` 数据：

```python
info = json.loads(info_string)

pprint(info)
#{'age': 24,
# 'ages for school': {'high school': 15,
#                     'middle school': 9,
#                     'primary school': 6,
#                     'university': 18},
# 'coding skills': ['python', 'matlab', 'java', 'c', 'c++', 'ruby', 'scala'],
# 'hobby': ['sports', 'reading'],
# 'married': False,
# 'name': 'echo'}
```

此时，我们将原来的 `JSON` 数据变成了一个 `Python` 对象，在我们的例子中这个对象是个字典（也可能是别的类型，比如列表）：

```python
type(info)		#dict
```

可以使用 `json.dumps()` 将一个 `Python` 对象变成 `JSON` 对象：

```python
info_json = json.dumps(info)

print(info_json)
#{"name": "echo", "age": 24, "coding skills": ["python", "matlab", "java", "c", "c++", "ruby", "scala"], "ages for school": {"primary school": 6, "middle school": 9, "high school": 15, "university": 18}, "hobby": ["sports", "reading"], "married": false}
```

从中我们可以看到，生成的 `JSON` 字符串中，**数组的元素顺序是不变的**（始终是 `["python", "matlab", "java", "c", "c++", "ruby", "scala"]`），而**对象的元素顺序是不确定的**。

#### 生成和读取 JSON 文件

与 `pickle` 类似，我们可以直接从文件中读取 `JSON` 数据，也可以将对象保存为 `JSON` 格式。

- `json.dump(obj, file)` 将对象保存为 JSON 格式的文件
- `json.load(file)` 从 JSON 文件中读取数据

```python
with open("info.json", "w") as f:
    json.dump(info, f)
```

可以查看 `info.json` 的内容：

```python
with open("info.json") as f:
    print(f.read())
#{"name": "echo", "age": 24, "coding skills": ["python", "matlab", "java", "c", "c++", "ruby", "scala"], "ages for school": {"primary school": 6, "middle school": 9, "high school": 15, "university": 18}, "hobby": ["sports", "reading"], "married": false}
```

从文件中读取数据：

```python
with open("info.json") as f:
    info_from_file = json.load(f)
    
pprint(info_from_file)
#{'age': 24,
# 'ages for school': {'high school': 15,
#                     'middle school': 9,
#                     'primary school': 6,
#                     'university': 18},
# 'coding skills': ['python', 'matlab', 'java', 'c', 'c++', 'ruby', 'scala'],
# 'hobby': ['sports', 'reading'],
# 'married': False,
# 'name': 'echo'}
```



### 08.04 glob 模块：文件模式匹配

`glob` 模块提供了方便的文件模式匹配方法。

例如，找到所有以 `.ipynb` 结尾的文件名：

```python
import glob

glob.glob("*.ipynb")
#['11.01-pprint.ipynb',
# '11.02-pickle-and-cPickle.ipynb',
# '11.03-json.ipynb',
# '11.04-glob.ipynb',
# '11.05-shutil.ipynb',
# '11.06-gzip,-zipfile,-tarfile.ipynb',
# '11.07-logging.ipynb',
# '11.08-string.ipynb',
# '11.09-collections.ipynb',
# '11.10-requests.ipynb']
```

`glob` 函数支持三种格式的语法：

- `*` 匹配单个或多个字符
- `?` 匹配任意单个字符
- `[]` 匹配指定范围内的字符，如：[0-9]匹配数字。

**举个例子：**

```python
#假设我们要匹配第 09 节所有的 `.ipynb` 文件：
glob.glob("../09*/*.ipynb")
#匹配数字开头的文件夹名：
glob.glob("../[0-9]*")
```



### 08.05 shutil 模块：高级文件操作

```python
import shutil
import os
```

`shutil` 是 `Python` 中的高级文件操作模块。

#### 复制文件

```python
with open("test.file", "w") as f:
    pass
print("test.file" in os.listdir(os.curdir))
#True

#shutil.copy(src, dst) 将源文件复制到目标地址：
shutil.copy("test.file", "test.copy.file")
print("test.file" in os.listdir(os.curdir))			#True
print("test.copy.file" in os.listdir(os.curdir))	#True

#如果目标地址中间的文件夹不存在则会报错：
try:
    shutil.copy("test.file", "my_test_dir/test.copy.file")
except IOError as msg:
    print(msg)
#[Errno 2] No such file or directory: 'my_test_dir/test.copy.file'
```

另外的一个函数 `shutil.copyfile(src, dst)` 与 `shutil.copy` 使用方法一致，不过只是简单复制文件的内容，并不会复制文件本身的读写可执行权限，而 `shutil.copy` 则是完全复制。

#### 复制文件夹

将文件转移到 `test_dir` 文件夹：

```python
os.renames("test.file", "test_dir/test.file")
os.renames("test.copy.file", "test_dir/test.copy.file")
```

使用 `shutil.copytree` 来复制文件夹：

```python
shutil.copytree("test_dir/", "test_dir_copy/")

"test_dir_copy" in os.listdir(os.curdir)
#True
```

#### 删除非空文件夹

`os.removedirs` 不能删除非空文件夹：

```python
try:
    os.removedirs("test_dir_copy")
except Exception as msg:
    print msg
#[Errno 39] Directory not empty: 'test_dir_copy'
```

使用 `shutil.rmtree` 来删除非空文件夹：

```
shutil.rmtree("test_dir_copy")
```

#### 移动文件夹

`shutil.move` 可以整体移动文件夹，与 `os.rename` 功能差不多。

#### 产生压缩文件

查看支持的压缩文件格式：

```python
shutil.get_archive_formats()
#[('bztar', "bzip2'ed tar-file"),
# ('gztar', "gzip'ed tar-file"),
# ('tar', 'uncompressed tar file'),
# ('xztar', "xz'ed tar-file"),
# ('zip', 'ZIP file')]
```

产生压缩文件`hutil.make_archive(basename, format, root_dir)`: 

```python
shutil.make_archive("test_archive", "zip", "test_dir/")
#'D:\\mynotes_from_github\\myNotes\\python笔记\\notes-python-master\\11-useful-tools\\test_archive.zip'
```

清理生成的文件和文件夹：

```python
os.remove("test_archive.zip")
shutil.rmtree("test_dir/")
```

### <u>08.06 gzip, zipfile, tarfile 模块：处理压缩文件</u>

```python
import os, shutil, glob
import zlib, gzip, bz2, zipfile, tarfile
```

#### zilb 模块

`zlib` 提供了对字符串进行压缩和解压缩的功能

#### gzip 模块

`gzip` 模块可以产生 `.gz` 格式的文件，其压缩方式由 `zlib` 模块提供。

我们可以通过 `gzip.open` 方法来读写 `.gz` 格式的文件：

```python
content = "Lots of content here"
with gzip.open('file.txt.gz', 'wb') as f:
    f.write(content.encode("utf-8"))
```

> 注意要先将字符串格式编码成字节码

**读：**

```python
with gzip.open('file.txt.gz', 'rb') as f:
    file_content = f.read()

print(file_content)
#b'Lots of content here'
```

将压缩文件内容解压出来：

```python
with gzip.open('file.txt.gz', 'rb') as f_in, open('file.txt', 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)
```

此时，目录下应有 `file.txt` 文件，内容为：

```python
with open("file.txt") as f:
    print(f.read())
#Lots of content here
```

```python
os.remove("file.txt.gz")
```



### 08.07 logging 模块：记录日志

`logging` 模块可以用来记录日志：

```
import logging
```

`logging` 的日志类型有以下几种：

- `logging.critical(msg)`
- `logging.error(msg)`
- `logging.warning(msg)`
- `logging.info(msg)`
- `logging.debug(msg)`

级别排序为：`CRITICAL > ERROR > WARNING > INFO > DEBUG > NOTSET`

默认情况下，`logging` 的日志级别为 `WARNING`，只有**不低于 `WARNING` 级别的日志才会显示在命令行。**

```python
logging.critical('This is critical message')
logging.error('This is error message')
logging.warning('This is warning message')

# 不会显示
logging.info('This is info message')
logging.debug('This is debug message')
#CRITICAL:root:This is critical message
#ERROR:root:This is error message
#WARNING:root:This is warning message
```

可以这样修改默认的日志级别：

```python
logging.root.setLevel(level=logging.INFO)

logging.info('This is info message')
#INFO:root:This is info message
```

可以通过 `logging.basicConfig()` 函数来改变默认的日志显示方式：

```python
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')

logger = logging.getLogger("this program")

logger.critical('This is critical message')
#CRITICAL:this program:This is critical message
```



### 08.08 string 模块：字符串处理

某些地方如果需要匹配标点符号，或者字母数字之类的，但又不能用正则表达式时，就可以考虑这个模块。

```
import string
```

标点符号：

```python
string.punctuation
#'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
```

字母表：

```python
print(string.ascii_letters)
#abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
```

小写和大写：

```python
print(string.ascii_lowercase)
print(string.ascii_uppercase)
#abcdefghijklmnopqrstuvwxyz
#ABCDEFGHIJKLMNOPQRSTUVWXYZ
```

数字：

```python
string.digits
#'0123456789'
```

16 进制数字：

```python
string.hexdigits
#'0123456789abcdefABCDEF'
```

每个单词的首字符大写：

```python
string.capwords("this is a big world")
#'This Is A Big World'
```



### 08.09 collections 模块：更多数据结构

这个模块实现了特定目标的容器，以提供Python标准内建容器 dict、list、set、tuple 的替代选择。

- **Counter**：字典的子类，提供了**可哈希对象的计数功能**
- defaultdict：字典的子类，提供了一个工厂函数，为字典查询提供了默认值
- OrderedDict：字典的子类，保留了他们被添加的顺序
- namedtuple：创建命名元组子类的工厂函数
- deque：类似列表容器，实现了在两端快速添加(append)和弹出(pop)
- ChainMap：类似字典的容器类，将多个映射集合到一个视图里面

#### Counter

可以使用 `Counter(seq)` 对序列中出现的元素个数进行统计。

例如，我们可以**统计一段文本中出现的单词及其出现的次数**：

```python
from string import punctuation

sentence = "One, two, three, one, two, tree, I come from China."
words_count = collections.Counter(sentence.translate(None, punctuation).lower().split())
print(words_count)
#Counter({'two': 2, 'one': 2, 'from': 1, 'i': 1, 'tree': 1, 'three': 1, 'china': 1, 'come': 1})
```

> translate(table[, delete])方法根据参数table给出的表(包含 256 个字符)转换字符串的字符,要过滤掉的字符放到 **deletechars** 参数中。

常用方法：

- elements()：返回一个迭代器，每个元素重复计算的个数，如果一个元素的计数小于1,就会被忽略。
- most_common([n])：返回一个列表，**提供n个访问频率最高的元素和计数**
- subtract([iterable-or-mapping])：从迭代对象中减去元素，输入输出可以是0或者负数
- update([iterable-or-mapping])：从迭代对象计数元素或者从另一个 映射对象 (或计数器) 添加。

```shell
>>> c = collections.Counter('hello world hello world hello nihao'.split())
# 查看元素
>>> list(c.elements())
['hello', 'hello', 'hello', 'world', 'world', 'nihao']
>>> d = collections.Counter('hello world'.split())
>>> c
Counter({'hello': 3, 'world': 2, 'nihao': 1})
>>> d
Counter({'hello': 1, 'world': 1})
# 追加对象，或者使用c.update(d)
>>> c + d
Counter({'hello': 4, 'world': 3, 'nihao': 1})
# 减少对象，或者使用c.subtract(d)
>>> c - d
Counter({'hello': 2, 'world': 1, 'nihao': 1})
```

#### defaultdict

`collections.defaultdict(default_factory)`为字典的没有的key提供一个默认的值。参数应该是一个函数，当没有参数调用时返回默认值。如果没有传递任何内容，则默认为None。

#### OrderedDict

Python字典中的键的顺序是任意的:它们不受添加的顺序的控制。
`collections.OrderedDict`类提供了保留他们添加顺序的字典对象。

#### namedtuple

三种定义命名元组的方法：**第一个参数是命名元组的构造器**（如下的：Person，Human）

```shell
>>> from collections import namedtuple
>>> Person = namedtuple('Person', ['age', 'height', 'name'])
>>> Human = namedtuple('Human', 'age, height, name')
>>> Human2 = namedtuple('Human2', 'age height name')
```

实例化命令元组

```shell
>>> tom = Person(30,178,'Tom')
>>> jack = Human(20,179,'Jack')
>>> tom
Person(age=30, height=178, name='Tom')
>>> jack
Human(age=20, height=179, name='Jack')
>>> tom.age #直接通过  实例名+.+属性 来调用
30
>>> jack.name
'Jack'
```

> **这个可以用来作为entity类，用以构造实体对象**

#### deque

`collections.deque`返回一个新的双向队列对象，从左到右初始化(用方法 append()) ，从 iterable （迭代对象) 数据创建。如果 iterable 没有指定，新队列为空。

`collections.deque`队列支持线程安全，对于从两端添加(append)或者弹出(pop)，复杂度O(1)。

虽然`list`对象也支持类似操作，但是这里优化了定长操作（pop(0)、insert(0,v)）的开销。
如果 maxlen 没有指定或者是 None ，deques 可以增长到任意长度。否则，deque就限定到指定最大长度。一旦限定长度的deque满了，当新项加入时，同样数量的项就从另一端弹出。

支持的方法：

- append(x)：添加x到右端
- appendleft(x)：添加x到左端
- clear()：清楚所有元素，长度变为0
- copy()：创建一份浅拷贝
- count(x)：**计算队列中个数等于x的元素**
- extend(iterable)：在队列右侧添加iterable中的元素
- extendleft(iterable)：在队列左侧添加iterable中的元素，注：在左侧添加时，iterable参数的顺序将会反过来添加
- index(x[,start[,stop]])：返回第 x 个元素（从 start 开始计算，在 stop 之前）。返回第一个匹配，如果没找到的话，升起 ValueError 。
- insert(i,x)：在位置 i 插入 x 。注：如果插入会导致一个限长deque超出长度 maxlen 的话，就升起一个 IndexError 。
- pop()：移除最右侧的元素
- popleft()：移除最左侧的元素
- remove(value)：移去找到的第一个 value。没有抛出ValueError
- reverse()：将deque逆序排列。返回 None 。
- maxlen：队列的最大长度，没有限定则为None。

#### ChainMap

一个 ChainMap 将**多个字典或者其他映射组合在一起**，创建一个单独的可更新的视图。 如果没有 maps 被指定，就提供一个默认的空字典 。`ChainMap`是管理嵌套上下文和覆盖的有用工具。

```shell
>>> from collections import ChainMap
>>> d1 = {'apple':1,'banana':2}
>>> d2 = {'orange':2,'apple':3,'pike':1}
>>> combined_d = ChainMap(d1,d2)
>>> reverse_combind_d = ChainMap(d2,d1)
>>> combined_d 
ChainMap({'apple': 1, 'banana': 2}, {'orange': 2, 'apple': 3, 'pike': 1})
>>> reverse_combind_d
ChainMap({'orange': 2, 'apple': 3, 'pike': 1}, {'apple': 1, 'banana': 2})
>>> for k,v in combined_d.items():
...      print(k,v)
... 
pike 1
apple 1
banana 2
orange 2
>>> for k,v in reverse_combind_d.items():
...      print(k,v)
... 
pike 1
apple 3
banana 2
orange 2
```





### 08.10 requests 模块：HTTP for Human

```python
import requests
```

Python 标准库中的 `urllib2` 模块提供了你所需要的大多数 `HTTP` 功能，但是它的 `API` 不是特别方便使用。

`requests` 模块号称 `HTTP for Human`，它可以这样使用：

```python
r = requests.get("http://httpbin.org/get")
r = requests.post('http://httpbin.org/post', data = {'key':'value'})
r = requests.put("http://httpbin.org/put")
r = requests.delete("http://httpbin.org/delete")
r = requests.head("http://httpbin.org/get")
r = requests.options("http://httpbin.org/get")
```

#### 传入 URL 参数

假如我们想访问 `httpbin.org/get?key=val`，我们可以使用 `params` 传入这些参数：

```python
payload = {'key1': 'value1', 'key2': 'value2'}
r = requests.get("http://httpbin.org/get", params=payload)
```

查看 `url` ：

```python
print(r.url)
#http://httpbin.org/get?key2=value2&key1=value1
```

#### 读取响应内容

`Requests` 会自动解码来自服务器的内容。大多数 `unicode` 字符集都能被无缝地解码。

```python
r = requests.get('https://github.com/timeline.json',timeout=10)
print(r.text)
#{"message":"Hello there, wayfaring stranger. If you’re reading this then you probably didn’t see our blog post a couple of years back announcing that this API would go away: http://git.io/17AROg Fear not, you should be able to get what you need from the shiny new Events API instead.","documentation_url":"https://docs.github.com/v3/activity/events/#list-public-events"}


#查看文字编码：
r.encoding		#'utf-8'


#每次改变文字编码，text 的内容也随之变化：
r.encoding = "ISO-8859-1"
r.text
#u'{"message":"Hello there, wayfaring stranger. If you\xe2\x80\x99re reading this then you probably didn\xe2\x80\x99t see our blog post a couple of years back announcing that this API would go away: http://git.io/17AROg Fear not, you should be able to get what you need from the shiny new Events API instead.","documentation_url":"https://developer.github.com/v3/activity/events/#list-public-events"}'
```

> 注：如果开vpn会get不到数据，报错SSLError

`Requests` 中也有一个内置的 `JSON` 解码器处理 `JSON` 数据：

```
r.json()
#{u'documentation_url': u'https://developer.github.com/v3/activity/events/#list-public-events',
 u'message': u'Hello there, wayfaring stranger. If you\xe2\x80\x99re reading this then you probably didn\xe2\x80\x99t see our blog post a couple of years back announcing that this API would go away: http://git.io/17AROg Fear not, you should be able to get what you need from the shiny new Events API instead.'}
```

如果 `JSON` 解码失败， `r.json` 就会抛出一个异常。











