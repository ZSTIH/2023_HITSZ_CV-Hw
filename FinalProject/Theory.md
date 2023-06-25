# 卷积神经网络的前向传播、反向传播算法理论推导

由题知，输入特征图$X$的尺寸为$3×3$，因此可设为$\begin{pmatrix} x_{11} & x_{12} & x_{13} \\ x_{21} & x_{22} & x_{23} \\ x_{31} & x_{32} & x_{33} \\\end{pmatrix}$；卷积核$W$的尺寸为$2×2$，因此可设为$\begin{pmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \\\end{pmatrix}$。另外，可设偏置$b$为$\begin{pmatrix} b_{11} & b_{12} \\ b_{21} & b_{22} \\\end{pmatrix}$，输出特征图$Y$为$\begin{pmatrix} y_{11} & y_{12} \\ y_{21} & y_{22} \\\end{pmatrix}$。

## 1 卷积层前向传播

根据卷积运算的定义，卷积层的前向传播可以通过滑动卷积核在输入特征图上进行**逐元素相乘、叠加求和**得到。对于每一个输出特征图的元素$y_{ij}$，可以使用以下公式计算：

$$
y_{ij} = \sum_{m=1}^{2} \sum_{n=1}^{2} w_{mn} \cdot x_{(i+m-1)(j+n-1)} + b_{ij}
$$

根据题目中给出的输入特征图$X$、卷积核$W$和偏置$b$的值，我们可以进行具体的计算。将输入特征图$X$和卷积核$W$代入上述公式，得到每个输出特征图元素的计算过程如下：

$$
\left\{
\begin{align*}
y_{11} &= w_{11} \cdot x_{11} + w_{12} \cdot x_{12} + w_{21} \cdot x_{21} + w_{22} \cdot x_{22} + b_{11} \\
y_{12} &= w_{11} \cdot x_{12} + w_{12} \cdot x_{13} + w_{21} \cdot x_{22} + w_{22} \cdot x_{23} + b_{12} \\
y_{21} &= w_{11} \cdot x_{21} + w_{12} \cdot x_{22} + w_{21} \cdot x_{31} + w_{22} \cdot x_{32} + b_{21} \\
y_{22} &= w_{11} \cdot x_{22} + w_{12} \cdot x_{23} + w_{21} \cdot x_{32} + w_{22} \cdot x_{33} + b_{22} \\
\end{align*}
\right.
$$

根据以上计算过程，可以得到输出特征图 $Y$ 的结果为：

$$
Y = \begin{pmatrix} y_{11} & y_{12} \\ y_{21} & y_{22} \\\end{pmatrix} = \begin{pmatrix} w_{11} \cdot x_{11} + w_{12} \cdot x_{12} + w_{21} \cdot x_{21} + w_{22} \cdot x_{22} + b_{11} & w_{11} \cdot x_{12} + w_{12} \cdot x_{13} + w_{21} \cdot x_{22} + w_{22} \cdot x_{23} + b_{12} \\ w_{11} \cdot x_{21} + w_{12} \cdot x_{22} + w_{21} \cdot x_{31} + w_{22} \cdot x_{32} + b_{21} & w_{11} \cdot x_{22} + w_{12} \cdot x_{23} + w_{21} \cdot x_{32} + w_{22} \cdot x_{33} + b_{22} \\\end{pmatrix}
$$

这样，我们就完成了卷积层的前向传播计算。

## 2 卷积层反向传播

设损失函数为$L$，卷积层的梯度输出为$\frac{\partial{L}}{\partial{Y}}$。

卷积层的反向传播需要计算关于输入特征图 $X$、卷积核 $W$ 和偏置 $b$ 的梯度，即 $\frac{\partial{L}}{\partial{X}}$、$\frac{\partial{L}}{\partial{W}}$ 和 $\frac{\partial{L}}{\partial{b}}$。

首先，我们来推导关于输入特征图的梯度 $\frac{\partial{L}}{\partial{X}}$。根据链式法则，可以得到以下推导过程：

$$
\frac{\partial{L}}{\partial{X}} = \frac{\partial{L}}{\partial{Y}} \cdot \frac{\partial{Y}}{\partial{X}}
$$

其中，$\frac{\partial{L}}{\partial{Y}}$ 是已知的，我们需要计算 $\frac{\partial{Y}}{\partial{X}}$。由于 $Y$ 的每个元素 $y_{ij}$ 是由 $X$ 和 $W$ 相关元素相乘求和得到的，因此可以推导出以下计算过程：

$$
\left\{
\begin{align*}
\frac{\partial{Y}}{\partial{x_{11}}} &= \frac{\partial{L}}{\partial{y_{11}}} \cdot w_{11} \\
\frac{\partial{Y}}{\partial{x_{12}}} &= \frac{\partial{L}}{\partial{y_{11}}} \cdot w_{12} + \frac{\partial{L}}{\partial{y_{12}}} \cdot w_{11} \\
\frac{\partial{Y}}{\partial{x_{13}}} &= \frac{\partial{L}}{\partial{y_{12}}} \cdot w_{12} \\
\frac{\partial{Y}}{\partial{x_{21}}} &= \frac{\partial{L}}{\partial{y_{11}}} \cdot w_{21} + \frac{\partial{L}}{\partial{y_{21}}} \cdot w_{11} \\
\frac{\partial{Y}}{\partial{x_{22}}} &= \frac{\partial{L}}{\partial{y_{11}}} \cdot w_{22} + \frac{\partial{L}}{\partial{y_{12}}} \cdot w_{21} + \frac{\partial{L}}{\partial{y_{21}}} \cdot w_{12} + \frac{\partial{L}}{\partial{y_{22}}} \cdot w_{11} \\
\frac{\partial{Y}}{\partial{x_{23}}} &= \frac{\partial{L}}{\partial{y_{12}}} \cdot w_{22} + \frac{\partial{L}}{\partial{y_{22}}} \cdot w_{12} \\
\frac{\partial{Y}}{\partial{x_{31}}} &= \frac{\partial{L}}{\partial{y_{21}}} \cdot w_{21} \\
\frac{\partial{Y}}{\partial{x_{32}}} &= \frac{\partial{L}}{\partial{y_{21}}} \cdot w_{22} + \frac{\partial{L}}{\partial{y_{22}}} \cdot w_{21} \\
\frac{\partial{Y}}{\partial{x_{33}}} &= \frac{\partial{L}}{\partial{y_{22}}} \cdot w_{22} \\
\end{align*}
\right.
$$

将上述计算结果整理成矩阵形式，即可得到关于输入特征图的梯度 $\frac{\partial{L}}{\partial{X}}$。

接下来，我们来推导关于卷积核的梯度 $\frac{\partial{L}}{\partial{W}}$。同样使用链式法则，可以得到以下推导过程：

$$
\frac{\partial{L}}{\partial{W}} = \frac{\partial{L}}{\partial{Y}} \cdot \frac{\partial{Y}}{\partial{W}}
$$

其中，$\frac{\partial{L}}{\partial{Y}}$ 是已知的，我们需要计算 $\frac{\partial{Y}}{\partial{W}}$。由于 $Y$ 的每个元素 $y_{ij}$ 是由 $X$ 和 $W$ 相关元素相乘求和得到的，可以推导出以下计算过程：

$$
\left\{
\begin{align*}
\frac{\partial{Y}}{\partial{w_{11}}} &= \frac{\partial{L}}{\partial{y_{11}}} \cdot x_{11} + \frac{\partial{L}}{\partial{y_{12}}} \cdot x_{21} \\
\frac{\partial{Y}}{\partial{w_{12}}} &= \frac{\partial{L}}{\partial{y_{11}}} \cdot x_{12} + \frac{\partial{L}}{\partial{y_{12}}} \cdot x_{22} \\
\frac{\partial{Y}}{\partial{w_{21}}} &= \frac{\partial{L}}{\partial{y_{21}}} \cdot x_{11} + \frac{\partial{L}}{\partial{y_{22}}} \cdot x_{21} \\
\frac{\partial{Y}}{\partial{w_{22}}} &= \frac{\partial{L}}{\partial{y_{21}}} \cdot x_{12} + \frac{\partial{L}}{\partial{y_{22}}} \cdot x_{22} \\
\end{align*}
\right.
$$

将上述计算结果整理成矩阵形式，即可得到关于卷积核的梯度 $\frac{\partial{L}}{\partial{W}}$。

最后，我们来推导关于偏置的梯度 $\frac{\partial{L}}{\partial{b}}$。同样使用链式法则，可以得到以下推导过程：

$$
\frac{\partial{L}}{\partial{b}} = \frac{\partial{L}}{\partial{Y}} \cdot \frac{\partial{Y}}{\partial{b}}
$$

其中，$\frac{\partial{L}}{\partial{Y}}$ 是已知的，我们需要计算 $\frac{\partial{Y}}{\partial{b}}$。由于 $Y$ 的每个元素 $y_{ij}$ 与偏置 $b_{ij}$ 相加得到，因此可以得到以下计算过程：

$$
\left\{
\begin{align*}
\frac{\partial{Y}}{\partial{b_{11}}} &= \frac{\partial{L}}{\partial{y_{11}}} \\
\frac{\partial{Y}}{\partial{b_{12}}} &= \frac{\partial{L}}{\partial{y_{12}}} \\
\frac{\partial{Y}}{\partial{b_{21}}} &= \frac{\partial{L}}{\partial{y_{21}}} \\
\frac{\partial{Y}}{\partial{b_{22}}} &= \frac{\partial{L}}{\partial{y_{22}}} \\
\end{align*}
\right.

$$

将上述计算结果整理成矩阵形式，即可得到关于偏置的梯度 $\frac{\partial{L}}{\partial{b}}$。

综上所述，我们完成了卷积层的反向传播计算。

## 3 池化层前向传播

由于使用的是最大池化层，池化核尺寸为$2\times2$，步长为1，无填充，因此池化层的前向传播可以通过滑动池化窗口的方式进行。对于最大池化层，它的作用是对输入特征图进行下采样，保留特征图中每个池化窗口中的**最大值**作为输出。

设$X_1^{'}=\begin{pmatrix} x_{11} & x_{12} \\ x_{21} & x_{22} \\\end{pmatrix}$，$X_2^{'}=\begin{pmatrix} x_{12} & x_{13} \\ x_{22} & x_{23} \\\end{pmatrix}$，$X_3^{'}=\begin{pmatrix} x_{21} & x_{22} \\ x_{31} & x_{32} \\\end{pmatrix}$，$X_4^{'}=\begin{pmatrix} x_{22} & x_{23} \\ x_{32} & x_{33} \\\end{pmatrix}$。

设函数 $max(\cdot)$ 返回的是**矩阵中的最大元素的值**，则可得到输出特征图$Y$的结果为：

$$
Y = \begin{pmatrix} y_{11} & y_{12} \\ y_{21} & y_{22} \\\end{pmatrix} = \begin{pmatrix} max(X_1^{'}) & max(X_2^{'}) \\ max(X_3^{'}) & max(X_4^{'}) \\\end{pmatrix}
$$

这样，我们就完成了池化层的前向传播计算。

## 4 池化层反向传播

已知池化层输出$Y$的梯度为$\frac{\partial L}{\partial Y}$，其中$L$是损失函数。现在我们需要计算关于输入特征图$X$的梯度$\frac{\partial L}{\partial X}$。

池化层的反向传播计算可以通过使用具有相同的池化窗口大小，但将最大值位置标记为1、其他位置标记为0的掩码矩阵来实现。将该掩码矩阵与池化层输出的梯度相乘，以便将梯度传递回池化层的输入。

假设池化层的输出特征图为$Y$，其梯度为$\frac{\partial L}{\partial Y}$，其中$L$表示损失函数。设掩码矩阵为$M$，则反向传播的计算过程如下：

- 初始化与输入特征图相同尺寸的梯度矩阵$\frac{\partial L}{\partial X}$为全零矩阵。

- 遍历池化层的输出特征图$Y$的每个元素$y_{ij}$以及对应的梯度$\frac{\partial L}{\partial y_{ij}}$：

    - 找到与$y_{ij}$对应的池化窗口$X_k^{'}$，其中$k$表示池化窗口的索引。

    - 初始化与池化窗口$X_k^{'}$相同尺寸的掩码矩阵$M_k$为全零矩阵。

    - 在池化窗口$X_k^{'}$中找到最大值的位置，并将该位置在$M_k$中标记为1，其他位置标记为0。

    - 将$\frac{\partial L}{\partial y_{ij}}$乘以掩码矩阵$M_k$，得到局部梯度$\frac{\partial L}{\partial X_k^{'}}$。

    - 将局部梯度$\frac{\partial L}{\partial X_k^{'}}$加到$\frac{\partial L}{\partial X}$的相应位置上。

- 得到最终的输入特征图的梯度$\frac{\partial L}{\partial X}$。
