---

layout:     post
title:      Matrix_Derivation
subtitle:   矩阵及向量的微分与求导
date:       2020-06-28
author:     JL
header-img: 
catalog: true
tags:

---

<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

### 1. 矩阵求导的本质：

矩阵A对矩阵B求导，矩阵A中的每一个元素分别对矩阵B中的每一个元素进行求导。



 $\frac{d f(x)}{d x}$   

A1x1 B1x1  1个导数

Amx1 B1x1 1个导数

Amx1 Bpx1 mxp 个导数

Amxn  Bpxq  mxnxpxq个导数

#### 1. 标量函数

$[f]_{1 X 1}$

$f(x)=x$

$f(x_1, x_2) = 2x_1 +3x_2^2$

#### 2. 向量函数

$[f]_{m X n}$

$[f_1(x) = x, f_2(x) = x_2]_{1 X 2}$

$$\left[\begin{array}{l}
f_{1}\left(x_{1}, x_{2}\right) f_{2}\left(x_{1},x_{2}\right) \\
f_{3}\left(x_{1}, x_{2}\right) f_{4}\left(x_{1}, x_{2}\right)
\end{array}\right]$$



###  2. 以分母布局为例的矩阵求导的基本原则

所谓以分母布局，就是说求导后的形状和分母的形状保持一致。

$\frac {\partial{f}  \leftarrow 分子}{\partial{x} \leftarrow 分母}$

原则：

1. 分子为标量，分母为向量，求导结果同分母的形状。

   $f$ :标量函数

   $x$ ：列向量 

   

​                  $$x=\left[\begin{array}{c}x_{1} \\ x_{2} \\ \vdots \\ x_{p}\end{array}\right]$$

书写latex公式太慢了，先将笔记图片拍照放这里，后续有空了，再整理。

![image-20200721101530848](/img/in-post/matrix_derivate/1.png)

![image-20200721101612562](/img/in-post/matrix_derivate/2.png)

![image-20200721101801370](/img/in-post/matrix_derivate/3.png)

![image-20200721101824565](/img/in-post/matrix_derivate/4.png)

![image-20200721101843182](/img/in-post/matrix_derivate/5.png)

![image-20200721101905267](/img/in-post/matrix_derivate/6.png)

![image-20200721101926363](/img/in-post/matrix_derivate/7.png)

![image-20200721101951284](/img/in-post/matrix_derivate/8.png)

![image-20200721102009719](/img/in-post/matrix_derivate/9.png)

![image-20200721102042319](/img/in-post/matrix_derivate/10.png)