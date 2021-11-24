## GDB调试基本教程

gdb是linux下用来调试C或C++ 的调试工具。

### gdb的进入与退出

先编译好需要调试的程序

```
gcc -g -o test test.c
```

然后使用`gdb test`命令打开gdb调试窗口。

### 基本指令

![image-20210925232602125](C:\Users\10277\AppData\Roaming\Typora\typora-user-images\image-20210925232602125.png)

![image-20210925232634325](C:\Users\10277\AppData\Roaming\Typora\typora-user-images\image-20210925232634325.png)

其中step和next的区别是，当进入一个函数的时候，前者会进入到函数的第一条语句，而后者不会进入到函数体内部执行。