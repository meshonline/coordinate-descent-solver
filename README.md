# coordinate-descent-solver
Coordinate descent solver for generic linear system

# Using coordinate descent method to solve linear equations
### This is the Google translation version from the Chinese version.

In actual projects, sometimes it is necessary to solve linear equations. If you use the least squares formula to calculate directly, the speed is slow for a large coefficient matrix, and sometimes there is no solution. So I want to find a way to solve linear equations by iterative method. Firstly, it must be stable, and secondly, there must always be a solution. After searching for a long time, there was no suitable one, so I just made one myself.

There are many examples of the coordinate descent method used for regression, but the examples used to solve linear equations have not been seen yet, so let's start with this.

Maybe it’s because my level is too low. I didn’t understand the source code after I quickly read the source code on GitHub. I didn’t read it at all. Let’s start from the concept, because I just finished reading "Basics of Linear Algebra for Machine Learning", so I’m very The geometric meaning of algebra has a feeling, this is very important.

First analyze: Ax=b. For the iterative method, first give an initial x0 to start the iteration. This value can be any value, such as all zeros or random numbers. If the coefficients other than the first x component are fixed, the equation can be written as:

c1\*(x1+x)+c2\*x2+...+cn\*xn = b

Where c1...cn is the column of A, x1...xn is the initial component of x0, and x is the only unknown.

After expanding the left side, move the constant term to the right side of the equation to get:

c1\*x = b-(c1\*x1+c2\*x2+...+cn\*xn)

That is:

c1\*x = b-Ax0

The left side of the equation is equivalent to a one-dimensional line passing through the origin in the high-dimensional space, and the right side can be understood as transforming the space point b to the high-dimensional space where Ax0 is located, and becomes a space point in the high-dimensional space.

Draw an auxiliary line perpendicular to this straight line from this point. The intersection of the vertical line is the point closest to this point on the line, and the coordinates should be moved to the intersection of the vertical line.

In linear algebra, the geometric meaning of the dot product is exactly the product of the modulus length of the projection of a vector on another vector and the modulus length of another vector, so the dot product of two vectors is divided by the modulus length of the other vector. Get the modulus length of the projection of a vector on another vector.

The modulus length of the projection of a vector on another vector is the position of the vertical line. Its geometric meaning is to start from the initial x1 and how far to reach the closest position to the target. Therefore, the complete solution needs to be x1 plus one The modular length of the projection of a vector on another vector.

After processing one x1 component, update x, then process the next x2 component, update x, until all x components are processed again.

The iterative method is very slow. The above process has to be repeated many times to converge to a stable solution. For some equations, such as the case where there are more unknowns than the number of equations, it can't even converge to a stable solution, but in general, you will get a very stable solution. Good approximate optimal solution.

I used the conjugate gradient descent method to check the equations of the symmetric matrix. The number of cycles required for the convergence of the coordinate descent method is dozens of times that of the conjugate gradient descent method, but because the calculation of the coordinate descent method is particularly simple, the actual calculation speed should be poor Not so much.

For asymmetric matrices, the conjugate gradient descent method will diverge and fail to converge, while the coordinate descent method can converge to a stable solution, so the coordinate descent method can be used to solve general linear equations.

The following is the code I wrote in Python. The implementation of the function is very simple. With only fifteen lines, you can solve general linear equations. Does it feel amazing?
```python
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 7 17:07:32 2020
@author: Mingfen Wang
"""
 
import numpy as np
 
def coordinate_descent_method(A, b, guess, N, TOR):
    A_T = A.T
    x = guess.copy()
    k = 1
    while k < N:
        save_x = x.copy()
        for i in range(len(x)):
            b_in_Ax = b - np.dot(A, x)
            foot = np.dot(b_in_Ax, A_T[i]) / np.dot(A_T[i], A_T[i])
            x[i] = x[i] + foot
        diff_x = x - save_x
        if np.dot(diff_x, diff_x) < TOR:
            break
        k = k + 1
    return x, k
 
A = np.array([[1., -1., 0.],\
              [-1., 2., 1.],\
              [0., 1., 5.]])
b = np.array([3., -3., 4.])
guess = np.array([0., 0., 0.])
x, k = coordinate_descent_method(A, b, guess, 300 * len(b), 1e-6)
print("x = {} in {} iterations.".format(x.round(2), k))
```
# 坐标下降法解线性方程组

在实际项目中，有时候需要解线性方程组，如果用最小二乘法公式直接计算，对于很大的系数矩阵，速度慢，有时候无解。所以想寻找一种用迭代法解线性方程组的方法，首先要稳定，其次要总是有解。找了半天也没有合适的，于是干脆自己做一个吧。

坐标下降法用做回归的例子很多，但用于求解线性方程组的例子还没有看到，就从这个入手吧。

也许是我的水平太低，快速看了看GitHub上的那些源代码也没看懂，干脆不看了，从概念硬做吧，因为最近刚看完《机器学习线性代数基础》，所以对线性代数的几何意义有了点感觉，这个很重要。

首先分析：Ax=b，对于迭代法，首先给出一个初始x0开始迭代，这个值可以是任意值，比如全零，或者随机数。如果固定住除了第一个x分量之外的系数，方程可以写为：

c1\*(x1+x)+c2\*x2+...+cn\*xn = b

其中c1...cn为A的列，x1...xn为初始x0的分量，x是唯一的未知数。

把左边展开后，把常数项移到方程的右边，得到：

c1\*x = b - (c1\*x1+c2\*x2+...+cn\*xn)

也就是：

c1\*x = b - Ax0

方程的左边相当于高维空间中过原点的一维直线，右边可以理解为把空间点b变换到Ax0所在的高维空间，变成高维空间的一个空间点。

从这个点做垂直于这条直线的辅助线，垂线的交点处就是直线上距离这个点最近的点，坐标应该移动到垂线的交点处。

线性代数中，点积的几何意义恰好是一个矢量在另一个矢量上的投影的模长与另一个矢量的模长之积，所以把两个矢量的点积除以另一个矢量的模长就得到一个矢量在另一个矢量上的投影的模长。

一个矢量在另一个矢量上的投影的模长就是垂线的位置，其几何意义是从初始的x1出发，走多远才能到达距离目标最近的位置，因此，完整解要用x1加上把一个矢量在另一个矢量上的投影的模长。

处理完一个x1分量，更新x，再处理下一个x2分量，更新x，直到所有的x分量都处理一遍。

迭代法很慢，上面的过程要重复很多次才能收敛到稳定的解，对于有些方程，比如未知数多于方程个数的情况，甚至无法收敛到稳定的解，但一般情况下，会得到一个非常好的近似最优解。

我用共轭梯度下降法验算了一下对称矩阵的方程，坐标下降法的收敛需要的循环次数是共轭梯度下降法的几十倍，但由于坐标下降法的运算特别简单，实际运算速度应该差不了那么多。

对于非对称矩阵，共轭梯度下降法就会发散，无法收敛，而坐标下降法可以收敛到稳定的解，因此可以用坐标下降法求解一般的线性方程组。

下面是我用Python写的代码，函数的实现很简单，只有十五行，就可以解一般的线性方程组，是不是觉得很神奇呢？
```python
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 7 17:07:32 2020
@author: Mingfen Wang
"""
 
import numpy as np
 
def coordinate_descent_method(A, b, guess, N, TOR):
    A_T = A.T
    x = guess.copy()
    k = 1
    while k < N:
        save_x = x.copy()
        for i in range(len(x)):
            b_in_Ax = b - np.dot(A, x)
            foot = np.dot(b_in_Ax, A_T[i]) / np.dot(A_T[i], A_T[i])
            x[i] = x[i] + foot
        diff_x = x - save_x
        if np.dot(diff_x, diff_x) < TOR:
            break
        k = k + 1
    return x, k
 
A = np.array([[1., -1., 0.],\
              [-1., 2., 1.],\
              [0., 1., 5.]])
b = np.array([3., -3., 4.])
guess = np.array([0., 0., 0.])
x, k = coordinate_descent_method(A, b, guess, 300 * len(b), 1e-6)
print("x = {} in {} iterations.".format(x.round(2), k))
```
