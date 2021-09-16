# JAVA基础知识学习01

## 1.一个简单的Java应用程序

```java
public class FirstSample{
	public static void main(String[] args){
		System.out.println("we will not use 'Hello world!'")；
	}
}
```
从这段最基本的程序中，可以知道一些Java最基本的规则：

   1.Java区分大小写；
 2. 关键字**public是访问修饰符**，在后面章节会具体介绍；
 3. 关键字class表明Java程序中的**全部内容都包含在类中**，类是构建所有Java应用程序和applet的构建块，Java应用程序中的全部内容都必须放置在类中，而且**关键字class后面紧跟类名**；
 4. Java中**所有函数**都是某个类的方法（或者说成员函数，也就是不存在类外的函数这种东西）；
 5. Java虚拟机（JVM）总是从指定类中的**main方法**的代码块开始执行，因此类的源文件中必须包含一个main方法，main方法**必须声明为public**；
 6. Java标准的命名规范为：类名是以大写字母开头的名词，多个单词组成，则每个单词第一个字母大写；
 7. 源代码的文件名必须与public类（每个源文件只能有一个类声明为public）的**名字相同**，并用.java作为扩展名；
 8. Java中与C++一样，采用大括号来划分程序的各个块，Java中每条代码都以分号结束，回车不是语句结束标志，因此可以一行写多个语句；
 9. void表示这个方法没有返回值。


## 2.注释

Java中一共有三种注释方式：
   1.单行注释：在每行的注释前标记 //

2. 多行注释：在注释段以“/* ”开头，以“ */”结尾
3. 文档注释：在注释段以“/**”开头，以“ */”结尾
4. 

## 3.数据类型
Java必须为每个变量都声明一种数据类型。java中共有8种基本类型。
### 3.1整型
整型共有四种：int(4字节)、long(8字节)、short(2字节)、byte(1字节)。后两者用于特殊场合，比如底层文件处理或内存有限但要处理大数组时。
长整型数值有个后缀l或者L，十六进制数值有个前缀0x或0X，八进制有个前缀0，加上前缀0b或0B可以写二进制数，还可以给大数字加下划线方便阅读。

### 3.2浮点类型
浮点型数值有float和double，后者表示该类数值的精度是前者的两倍，float型数值有个后缀f或F，无后缀默认为double型
### 3.3 char类型

char类型用来表示单个字符（但现在一些Unicode字符需要两个char值）。char类型的字面量值要用单引号括起来。例如'A'的 编码值为65的字符变量，而"A"是一个字符A的字符串。

转义序列都可以出现在加引号的字面量或字符串中。转移序列例如，` \n \t \\ \' \"`等。特别要注意单斜杠在使用时，要转义。

### 3.4 Unicode和char类型

总之，尽量不要在程序中使用char类型，除非确实需要处理UTF-16代码单元。

### 3.5 Boolean类型
Boolean类型有两个值：true和false。用来判定逻辑条件，整型值和布尔值之间不能相互转换。




## 4.变量与常量
### 4.1 变量（声明与赋值）

在申明变量时，要先指定一个变量的类型，然后是变量名，声明的变量后接分号，可以在一行中声明多个变量。
```java
int i , j;//i和j都是整型变量，但尽量不要一行声明多个变量
```
变量初始化，可以直接在变量名后面用等号赋值。变量的声明尽量靠近第一次使用该变量的地方。
```java
int vacationDays = 12
String greeting = "Hello"
```
### 4.2 常量（final）
Java中一般用final来指示常量。一共会有三种情况：
```java
final 类型 变量名 = data；//表示该变量只能被赋值一次
static final 类型 变量名 = data；//这是一个类常量，可以在一个类的多个方法中使用,注意要声明在方法外面、类里面
public static final 类型 变量名=data；//其他类的方法也可以使用该常量
```
## 4.3 枚举类型（enum）
自定义枚举类型，枚举类型包括有限个命名的值。如
```java
enum Size = {SMALL,MEDIUM,LARGE,EXTRA_LARGE};
```
现在Size可以作为一个数据类型，声明该数据类型的变量，其值只能是枚举类型中的某个枚举值或者是特殊值null。如
```java
Size s = Size.MEDIUM;// 变量s是Size类型的
```


## 5 运算符

### 5.1 算数运算符
Java中算术运算符包括+、-、*、/，即加减乘除。其中当参与/运算的两个数都是整数时，表示整数除法，否则为浮点数除法。整数的求余用%表示。
### 5.2 数学函数与常量
Math类中包含了各种各样的数学函数，例如
```java
Math.sqrt(x);//表示求x的平方根
Math.pow(x,a);//表示求x的a次幂
Math.PI;Math.E;//表示pai值和最然对数的底数
```
引入Math包，就可以不用添加前缀“Math”，即
```java
import static java.lang.Math.*;
```
此外Math包中还有三角函数sin、cos、tan、atan；指数exp；对数log；10为底的对数log10等

### 5.3 数值类型间的转换

两个不同类型的操作数做运算时，数据类型优先级如下：有double则全为double，否则有float则全为float，否则有long则全为long，否则为int。
### 5.4 强制类型转换
强制转换方式如下
```java
double x = 9.997;
int nx = (int) x; //此时nx的值为9
```
如果想进行舍入，就要用到Math.round方法
```java
double x = 9.997;
int nx = (int) Math.round(x); //此时nx的值为10。任然需要强制转换，因为round返回的结果为long型
```
### 5.5 结合赋值运算符与自加自减
```java
x+=4;//相当于x= x+4
```
自加自减：x++是先使用原来的值，使用完之后再加1；而++x是先对x加1，再用在表达式中。
### 5.6 关系和Boolean运算符
- Java中检测相等用==，检测不等用！=，其返回的值为true或者false。此外还有>,<,<=,>=运算符。
- 逻辑运算符有：&&表示逻辑与，||表示逻辑或，！表示逻辑非。对于逻辑与运算，第一个表达式为false则直接返回结果，不运算第二个表达式。对于逻辑或，第一个表达式为true则直接返回结果。
- 三元操作符？：其表达式为 condition？expression 1：expression 2  意思是条件为true则返回第一个表达式的值，否则返回第二个的。

### 5.7 位运算符
```java
&("and")     |("or")   ^("xor")    ~("not")
```
\>>和\<<运算符可以将位模式左移或右移
\>>>运算符会用0填充高位，而\>>是用符号位填充高位，不存在\<<<运算符。

### 5.8 括号与运算符级别

感觉记不住可以直接用括号来处理。



## 6 字符串

从概念上讲，Java字符串就是Unicode字符序列。

### 6.1子串

String类的subString方法可以从一个较大的字符串中提取一个子串，例如：

```java
String greeting = "Hello";
String s = greeting.subString(0,3); //s="Hel"
```

### 6.2 拼接

java可以使用+号直接连接两个字符串。当将一个字符串与一个非字符串拼接时，后者会转换成字符串。

如果需要把多个字符串放在一起，并用一个界定符分隔，可以用静态join方法：

```java
String all = String.join("/","S","M","L");
//all = "S/M/L"
```

### 6.3不可变字符串

String类没有提供修改字符串中某个字符的方法。如果想把"Hello"换成"Help!"，只能取前者的前三个字符再和"p!"进行拼接。像这个例子中的方法，实际是在内存中同时存了"Hello"和"Help!"，然后将存放前者的内存地址换成后者的内存地址。对于内存中一些用不到的字符串或数据，Java特有的垃圾回收机制可以判断并回收，避免了内存消耗。这样的不可变字符有一个优点，就是编译器可以让字符串共享。

#### 6.4 检测字符串是否相等

对于字符串的检测，使用equal方法，该方法不仅可以使用字符串的变量，也可以使用字符串的字面量。

```Java
s.equals(t);
"hello".equals(greeting);
"Hello".equalsIgnoreCase("hello");//这三种方法都是可以的
```

### 6.5 空串与null串

空串""是长度为0的字符串，可以用以下方法检查字符串是否为空：

```java
if(str.length() == 0)
if(str.equals("")) //这两种方法都可以
```

空串是一个Java对象，有自己的串长度（0）和内容（空）.但String还可以放一个特殊的值，即null，表示目前没有任何对象与该变量关联，检查一个字符串是否为null，用下面方法：

```java
if(str == null)
```

一般都需要首先判断是否为null，不然会报错。

### 6.6 String API

```java
int compareTo(String other)

boolean equals(Object other)

int indexOf(String str)

...
看书上的50页或者联机文档
```

### 6.7 构造字符串

如果需要将多端小字符拼接成一个字符串，使用原来拼接的方法的话，效率较低，因此可以使用StringBuilder类，例如：

```java
// 构建一个空的字符串构造器
StringBuilder builder = new StringBuilder();
// 每当需要添加一部分内容时，就用append方法
builder.append(ch);
builder.append(str);
// 最后将构建完成的字符串调用toString方法
String completeString = builder.toString();
```

同样，stringBuilder类中也有一些重要方法，可以看书上54页的api文档

### 6.8 码点与代码单元

Java字符串由char值序列组成。常用的Unicode字符使用一个代码单元就可以表示，辅助字符需要用到一对代码单元

1. length方法将返回采用UTF-16编码表示给定字符串所需要的**代码单元数量**（也就是有几个字符，就返回多少）。例如

```java
String greeting = "Hello";
int n = greeting.length(); // is 5
```

2. 若想要得到实际的长度，即**码点数量**，可以调用：

```java
int cpCount = greeting.codePointCount(0,greeting.length());
```

3. 调用s.charAt(n)将返回位置n的代码单元，n介于0和s.length()-1之间，例如

```java
char first = greeting.charAt(0); // first is 'H'
char last = greeting.charAt(4); // last is 'o'
```

4. 如果想得到第i个码点，应该使用下面的语句：

```java
int index = greeting.offsetByCodePoints(0,i);
int cp = greeting.codePointAt(index)；
```

5. 如果想要遍历一个字符串，可以使用codePoints方法，它会生成一个int值的‘流’，每个int值对应一个码点。可以将它转换成一个数组，再完成遍历：

```java
int[] codepoints = str.codePoints().toArray();
```

6. 要把一个码点数组转换成一个字符串，可以使用构造器：

```java
Stirng str = new String(codePoints, 0, codePoints.length);
```



## 7 输入与输出
### 7.1读取输入
要想通过控制台进行输入，首先需要构造一个与“标准输入流”System.in关联的Scanner对象。即
```java
import java.util.*      //Scanner类定义在java.util包中
//使用的类不是定义在基本的java.lang包中时，要使用import导入包
...
Scanner in = new Scanner(System.in);
System.out.print("what is your name?");
String name = in.nextLine();//nextLine方法将读取一行输入
```
使用nextLine方法是因为输入行中可能包含空格，读取一个单词可以用next方法（以空白符作为分隔符），读取整数用nextInt，读取浮点数用nextDouble。

因为输入是可见的，所以Scanner类不适用于从控制台读取密码，java 6 中引入了Console类来实现这个目标，要读取一个密码，可以使用下列代码：

```java
Consloe cons = System.console();//具体如何实现待后续
String username = cons.readLine("user name:");
char[] passwd = cons.readPassword("password:");
```
Scanner中还有方法：hasNext（）、hasNextInt（）等，返回一个boolean值。

### 7.2 格式化输出

格式化数值输出用printf方法，例如，调用
```java
System.out.printf("%8.2f",x);
```
会按照一个字段宽度输出x：包括8个字符，精度为小数点后两个字符。如果除去小数部分，整数部分字符是不足5（8-3），就会打印一个前导空格。
还可以为printf提供多个参数，例如
```java
System.out.printf(“Hello, %s. Next year, you will be %d", name, age);
```
每个以%开头的格式说明符都用相应的参数替换。格式说明符尾部的转换符指示要格式化的数值的类型：f表示浮点数，s表示字符串，d表示10进制整数，c表示字符，b表示布尔，%表示%（即要表示百分号是，得用%%）
另外还可以指定控制格式化输出外观的各种标志。其中+表示打印正数和负数的符号   0表示数字前补0   （表示将负数扩在括号内      ，表示添加分组分隔符（如3,333.33）
### 7.3 文件输入与输出
要想读取一个文件，需要构造一个Scanner对象，如下所示
```java
Scanner in = new Scanner(Path.of("myfile.txt"),StandardCharset.UTF_8);
```
文件名中如果有反斜杠，则需要加一个额外的反斜杠进行转义。上述代码中指定了UTF-8字符编码，如果省略了就会使用默认编码，这一般不建议。
要想写入文件，就需要构造一个PrintWriter对象。在构造器中，需要提供文件名和字符编码：
```java
PrintWriter out = new PrintWriter(("myfile.txt"),StandardCharset.UTF_8);
```
如果文件不存在，就会创建该文件，再写入。
如果用一个不存在的文件构造Scanner，或用一个无法创建的文件名构造PrintWriter，就会出现严重错误，这里可以用throws子句在main方法中做标记。

这里只提到了简单的方法，更高级的内容，后面会有专门一章。



## 8 控制流程

### 8.0 块作用域

注意声明的变量只能在其所声明的块作用域中有效，在作用域外无法引用。而且不能再嵌套的两个块中声明同名的变量。

### 8.1 条件语句
```java
if  (condition)  statement
if  (condition)  statement1 else  statement1 
```
条件用小括号括起来，statement是多个语句就用大括号括起来

### 8.2 循环
while循环：
```java
while (condition) statement //先做判断在执行程序，可能出现一次循环都没做的情况
do statement while (condition); //先执行一次程序再做判断
```
for循环：
```java
for (int i = 1;i<=10;i++)  //for语句至少包含初始化、检测和更新三部分
	System.out.println(i);  //循环部分
```
### 8.3 多重选择：switch语句
switch语句将从与选项值相匹配的case标签开始执行，直到遇到break语句。
case 的标签可以是：
1、类型为char、byte、short或int的常量表达式
2、枚举常量
3、从Java 7开始，case还可以是字符串字面量

```
switch(choise)
{
	case 1:
		...
		break;
	case 2:
		...
		break;
	case 3:
		...
		break;
	...
	default:
		break;
}
```

### 8.4 中断控制流程的语句
除了上述的break外，Java还有一种带标签的break语句。标签放在希望跳出的最外层循环之前，并带冒号。break后面跟着标签名。例如

这种带标签的break语句不但可以用在循环中，还可以用在if语句中。
与break向对应的就是continue，若执行到该语句，则会马上跳到while循环的首部，if循环的更新部分。还有一种就是带标签的continue语句，该语句将跳到与标签匹配的循环的首部。例如
```java
for(count = 1; count <=100; count++){
	System.out.print("enter a number,-1 to quit:");
	n = in.nextInt();
	if (n<0) continue;
	sum += n;
}
```


## 9 数组

数组是存储相同类型值的序列。

### 9.1 声明数组

1. 在声明数组变量时，需要指出数组类型和数组变量的名字。如

```java
int[] a; //只声明额变量a，并没有将a初始化为一个真正的数组
int[] a = new int[100];  //or var a = new int[100]; 
//声明并初始化一个存储100个整数的数组
```
new int[n]会创建一个长度为n的数组。**一旦创建了数组，就不能改变其长度**。如果在程序运行中需要经常扩展数组大小，就需要使用另一种数据结构，即**数组列表**（array list），数组列表后续介绍。

2. 另外有一种创建数组并提供初始值的方法，例如

```java
int[] smallPrimes = {2,3,5,7,11,13};
String[] authors = {"james","tom",""bill,};//这种方法不需要new

new int[] {17,19,23,29,31};//该方法声明了一个匿名数组
smallPrimes = new int[] {17,19,23,29,31}; //重新初始化了smallPrimes，底层方法是先将右端的声明成一个匿名数组，然后再赋值给smallPrimes
```
创建一个数字数组时，所有元素都初始化为0；Boolean数组的元素初始化为false；对象数组的元素则初始化为一个特殊值null，如创建字符串数组时，未赋值时都为null。

3. 访问数组元素：如果a是一个整型数组，a[i]就是数组中下标为i的整数。a[n]的下标为0~n-1.

4. 要获得数组中元素的个数，可以使用array.length

   > 注意与字符串长度str.length()的区别

5. 给数组中元素赋值，可以使用如下方法：

```java
int[] numbers = new int[n];
for (int i=0; i<numbers.length;i++)
	numbers[i] = i+1;
```
### 9.2 for each循环
for each循环是用来处理数组（或其他元素集合）中每个元素的一种方法，这种方法不必考虑指定下标值。语句格式为
```java
for (variable : collection) statement
```
collection这一集合表达式必须是一个数组或者是一个实现了Iterable接口的类对象（如ArrayList）。variable是一个与数组元素同类型的变量，用来存储每个数组元素。例如
```java
for (int element : a) ja
	System.out.println(element);
```
如果要打印数组中的所有值，可以用Array类中的toSting方法。调用Array.toSting(a),返回一个包含数组元素的字符串，这些元素包围在中括号内，并用逗号分隔，例如
```java
System.out.println(Array.toSting(a));//输出为“[2,3,5,7,11,13]”
```
### 9.3 数组拷贝（拷贝变量和拷贝元素）
1）Java中允许将一个数组变量拷贝到另一个数组变量，此时两个变量将引用同一个数组，修改一个变量的元素，另一个也会改变。如：

```java
int[] luckyNumbers = smallPrimes;
luckyNumbers[5] = 12; //now smallPrimes[5] is also 12
```
2）如果要将一个数组的所有值拷贝到另一个新的数组中，就要使用Array类的**copyOf**方法，例如

```java
int[] copiesluckyNumbers = Array.copyOf(luckyNumbers,luckyNumbers.length);
```
上式中的第二个参数是新数组的长度，这个方法可以用来增加数组的大小，超过原数组部分的元素被赋值为0。相反，如果小于原始数组长度，则只拷贝前面的值。

3）如果不想从起始位置拷贝，可以用另一个方法

```java
Array.copyOfRange(luckyNumbers,k,n)// 这种方法即，拷贝从k到n-1的值
```

### 9.4 命令行参数
Java中都会有个带String arg[]参数的main方法。该参数表明main方法接收一个字符串数组，也就是命令行上指定的参数。而程序名不会存储在args 数组中，例如命令行上输入java Message -h world 时，args[0]时“-h”，而不是java或者Message。

```java
public class Messige{
	public static void main(String[] args){
		if(args.length == 0 || args[0].equals("-h"))
			System.out.print("Hello,")
		else if(args[0].equals("-g"))
			System.out.print("Goodbye,")
		for (int i=1;i<args.length; i++)
			System.out.print(" "+args[i]);
	}
}
// 当在命令行输入以下命令：java Message -g cruel world
// args 数组将包含以下内容：
// args[0]:"-g"
// args[1]:"cruel"
// args[2]:"world"
```

### 9.5 数组排序
要对数值型数组进行排序，可以使用Arrays类中的sort方法，如
```java
int[] a = new int[10000];
...
Array.sort(a) // 这个方法底层用了优化的快速排序算法
```
这里有一个例子里面有很多要点，好好看看：
```java
import java.util.*;
public class LotteryDrawing{
	public static void main(String[] args){
		Scanner in = new Scanner(System.in);
		System.out.println("How many number do you need to draw?");
		int k = in.nextInt();
		System.out.print("What is the highest number you can draw?");
		int n = in.nextInt();
		int[] numbers = new int[n];
		for(int i=0;i<numbers.length;i++)
			numbers[i] = i+1;
		int result = new int[k];
		for(int i=0;i<result.length;i++){
			int r = (int) (Math.random()*n);
			result[i] = numbers[r];
			numbers[r] = numbers[n-1];
			n--;
		}
		Arrays.sort(result);
		System.out.println("Bet the following combination. It will make you rich");
		for(int r : result)
			System.out.println(r);
	}
}
```
array类中还有一些常用的方法，参考书上85页的api，例如toString(), copyOf(), copyOfRange(), binarySearch() 等等。

### 9.6 多维数组与不规则数组

1. 多维数组的声明方式与一维数组类似。

```java
double[][] balances = new double[NYEARS][NYEARS];

int[][] magicSquare = {
	{16,3,2,13},
	{5,10,11,8},
	{9,6,7,12},
	{4,15,14,1}
}; // 这两种方法都可以
```

实际上java中没有多维数组，只有一维数组。多维数组被解释为数组的数组。因此可以用来定义一个不规则数组，如下：

```java
// 1
// 1 1
// 1 2 1
// 1 3 3 1
// 1 4 6 4 1
// 1 5 10 10 5 1
// 1 6 15 20 15 6 1
// 要声明上面这样一个数组，用以下方法
int[][] odds = new int[NMAX+1][];
for (int n =0; n <= NMAX; n++)
	odd[n] = new int[n+1]
for(int n=0; n<odds.length; n++)
	for (int k=0; k<odds[n].length; k++){
		...
		odds[n][k] = lotteryOdds;
	}
```



## 10 大数

如果基本的整数和浮点数精度不能满足要求，可以使用java.math包中两个很有用的类：BigInteger和BigDecimal。这两个类可以处理包含任意长度数字序列的数值。两种分别实现任意精度的整数运算和浮点数运算。

1. 可以使用静态方法valueOf将普通的数值转换成大数：

```java
BigInteger a = BigInteger.valueOf(100);
```

2. 对于更大的数可以用一个构造器：

```java
BigInteger reallyBig = new BigInteger("1263456468465651616215654654654654654654")
```

3. 还有一些常量如BigInteger.ZERO、BigInteger.ONE等

不能使用常用的算数运算符（比如+和*）来处理大数。而需要使用大数中的add和multiply方法。

更多的一些方法例如加减乘除，及其注意事项，可以看课本78页的api

