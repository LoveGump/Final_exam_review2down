# Lab复习

## IDE反汇编实验

就是一个函数调用的过程

函数调用的步骤：

- 参数入栈
- 返回地址入栈
- 代码区跳转
- 栈帧调整



## OLLYDBG软件破解

就是破解一个软件

- 第一个方法就是修改程序的逻辑，将`Jz short 0041364b`修改为`Jnz short 0041364b`

- 第二个方法就是将函数的实际意义进行修改

  - 对应的汇编语言代码：

    ```assembly
    Mov dword ptr [ebp-8], eax 
    Xor eax, eax
    Cmp dword ptr [ebp-8], 0 
    Sete al 
    ```

  - 我们只需要将其修改为：将al直接赋值为1，然后将后面的sete al改为nop指令就可以了

    

## 堆溢出Dword Shoot模拟实验

```c++
#include <windows.h>
main()
{
    HLOCAL h1, h2,h3,h4,h5,h6;
    HANDLE hp;
    hp = HeapCreate(0,0x1000,0x10000); //创建自主管理的堆
    h1 = HeapAlloc(hp,HEAP_ZERO_MEMORY,8);//从堆里申请空间
    h2 = HeapAlloc(hp,HEAP_ZERO_MEMORY,8);
    h3 = HeapAlloc(hp,HEAP_ZERO_MEMORY,8);
    h4 = HeapAlloc(hp,HEAP_ZERO_MEMORY,8);
    h5 = HeapAlloc(hp,HEAP_ZERO_MEMORY,8);
    h6 = HeapAlloc(hp,HEAP_ZERO_MEMORY,8);

    _asm int 3  //手动增加int3中断指令，会让调试器在此处中断
    //依次释放奇数堆块，避免堆块合并
    HeapFree(hp,0,h1); //释放堆块
    HeapFree(hp,0,h3); 
    HeapFree(hp,0,h5); //现在freelist[2]有3个元素

    h1 = HeapAlloc(hp,HEAP_ZERO_MEMORY,8); 

    return 0;
}
```

- 程序首先创建了一个大小为 0x1000 的堆区，并从其中连续申请了6个块身大小为 8 字节的堆块，加上块首实际上是6个16字节的堆块。
- 释放奇数次申请的堆块是为了防止堆块合并的发生。
- 三次释放结束后，会形成三个16个字节的空闲堆块放入空表。因为是16个字节，所以会被依次放入freelist[2]所标识的空表，它们依次是h1、h3、h5。
- 再次申请8字节的堆区内存，加上块首是16个字节，因此会从freelist[2]所标识的空表中摘取第一个空闲堆块出来，即h1。
- 如果我们手动修改h1块首中的前后向指针，能够观察到 DWORD SHOOT 的发生。



## 格式化字符串漏洞

- ％d整型输出
- ％ld长整型输出
-  ％o以八进制数形式输出整数
-  ％x以十六进制数形式输出整数
- ％u以十进制数输出unsigned型数据(无符号数)
-  ％c用来输出一个字符
-  ％s用来输出一个字符串
- ％f用来输出实数，以小数形式输出
- 格式化符号%n，它的作用是将格式化函数输出字符串的长度，写入函数参数指定的位置



## Shellcode编写及编码

不会





## API函数自搜索实验

不会



## 程序插桩及Hook实验

![image-20240619213010732](E:\学学学\本科\大二下\软件安全\复习笔记_陆皓喆\章节复习\Lab复习.assets\image-20240619213010732.png)

![image-20240619213051159](E:\学学学\本科\大二下\软件安全\复习笔记_陆皓喆\章节复习\Lab复习.assets\image-20240619213051159.png)





## Angr应用实例

![image-20240619213645875](E:\学学学\本科\大二下\软件安全\复习笔记_陆皓喆\章节复习\Lab复习.assets\image-20240619213645875.png)

![image-20240619213701419](E:\学学学\本科\大二下\软件安全\复习笔记_陆皓喆\章节复习\Lab复习.assets\image-20240619213701419.png)



## WEB开发实践

没法考



## 跨站脚本攻击

![image-20240619213957257](E:\学学学\本科\大二下\软件安全\复习笔记_陆皓喆\章节复习\Lab复习.assets\image-20240619213957257.png)



## SQL盲注