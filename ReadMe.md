
# 计算机视觉中的降维方法串烧

降维方案大约分为线性降维与非线性降维，不同的降维方式之间往往是降维之后保留信息的侧重点不同，在具体的数学推导上面，体现为优化函数的不同。

## 第一部分： 线性降维理论

### PCA理论

主成分分析，或者称为PCA，是⼀种被⼴泛使⽤的技术，应⽤的领域包括维度降低、有损数
据压缩、特征抽取、数据可视化（Jolliffe, 2002）

PCA的数学定义是：一个正交化线性变换，把数据变换到一个新的坐标系统中，使得这一数据的任何投影的第一大方差在第一个坐标（称为第一主成分）上，第二大方差在第二个坐标（第二主成分）上，依次类推。

以二维空间的数据降维到一维空间为例

PCA可以被定义为数据在低维线性
空间上的正交投影，这个线性空间被称为主⼦空间（principal subspace），使得投影数据的⽅差
被最⼤化（Hotelling, 1933）。等价地，它也可以被定义为使得平均投影代价最⼩的线性投影。
平均投影代价是指数据点和它们的投影之间的平均平⽅距离（Pearson, 1901）。

![title](images/1.png)

图 1: 主成分分析寻找⼀个低维空间，被称为主⼦平⾯，⽤紫⾊的线表⽰，使得数据点（红点）在⼦空
间上的正交投影能够最⼤化投影点（绿点）的⽅差。PCA的另⼀个定义基于的是投影误差的平⽅和的最
⼩值，⽤蓝线表⽰。

#### 最大方差形式

设想一下对二维数据进行降维，我们需要做的就是将二维数据投射到一个方向上，使得平均投影代价最小。下面两个方向（大红色与紫红色）的选择，哪一种符合要求？

<img src="images/2.png" width="40%">

考虑⼀组观测数据集$x_n$，其中$n = 1, . . . , N$，因此$x_n$是⼀个D维欧⼏⾥得空间中的变量。我们的⽬标是将数据投影到维度$M < D$的空间中，同时最⼤化投影数据的⽅差。⾸先，考虑在⼀维空间$(M=1)$上的投影。我们可以使⽤D维向量$u_1$定义这个空间的⽅向。为了⽅便（并且不失⼀般性），我们假定选择⼀个单位向量，从⽽$u^T_1 u_1 = 1$（注意，我们只对$u_1$的⽅向感兴趣，⽽对$u_1$本⾝的⼤⼩不感兴趣）。这样，每个数据点$x_n$被投影到⼀个标量值$u^T_1 u_1 = 1$上。投影数据的均值是$u^T_1 \bar{x}$，其中，$\bar{x}$是样本集合的均值，形式为

$$\bar{x} = \frac{1}{N} \sum^N_{n=1} x_n$$

投影数据的⽅差为

$$\frac{1}{N} \sum^N_{n-1} { (u^T_1 x_n - u^T_1 \bar{x}) }^2= u^T_1 S u_1$$

其中$S$是数据的协⽅差矩阵，定义为

$$S = \frac{1}{N}\sum^N_{n=1}(x_n - \bar{x})(x_n - \bar{x})^T$$

为了将方差最大化，并且使得$u^T_1$满足$u^T_1 u_1 =1$，引入拉格朗日乘数，记作$\lambda_1$，有下式：

$$u^T_1 S u_1 + \lambda_1(1-u^T_1 u_1)$$

通过令它关于$u_1$的导数等于零，我们看到驻点满⾜

$$S u_1 = \lambda_1 u_1$$

这表明$u_1$一定是$S$的一个特征向量。如果我们左乘$u_1^T$，利用$u^T_1 u_1 =1$， 我们得到：

$$u^T_1 S u_1 = \lambda_1$$

因此当我们将$u_1$设置为与具有最⼤的特征值$\lambda_1$的特征向量相等时，⽅差会达到最⼤值。这个特征向量被称为第⼀主成分。

对于高维的数据，我们按照上面的方法，去除掉第一主成分后，用同样的方法得到第二主成分。

##### 注意：

1. 通常，为了确保第一主成分描述的是最大方差的方向，我们会使用平均减法进行主成分分析。如果不执行平均减法，第一主成分有可能或多或少的对应于数据的平均值。另外，为了找到近似数据的最小均方误差，我们必须选取一个零均值。
2. PCA对变量的缩放很敏感。
3. 对高维数据，PCA需要很高的计算要求。
4. 每一个特征值都与跟它们相关的方差是成正比的，而且所有特征值的总和等于所有点到它们的多维空间平均点距离的平方和。


```python
import os
import cv2
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
num_class = 40
num_in_class = 10
img_w = 92
img_h = 112
show_num = 0
images_path = './data/att_faces'
imgs_dir = os.listdir(images_path)
faces = list()
labels = list()
label_num = 0
for list_dir in imgs_dir:
    imgs = list()
    temp_path = images_path + '/' + list_dir
    if os.path.isdir(temp_path):
        label_num = label_num + 1
        imgs_list = os.listdir(temp_path)
        for img_name in imgs_list:
            img_path = temp_path + '/' + img_name
            img = cv2.imread(img_path)
            if img_name == '1.pgm':
                show_num = show_num + 1
                plt.subplot(4,10,show_num)
                plt.axis('off')
                plt.imshow(img)
            imgs.append(img)
            labels.append(label_num)
    faces.append(imgs)
plt.show()
```


![png](images/output_images/output_30_0.png)



```python
X = np.zeros((num_in_class*num_class, img_w*img_h))
num = 0
for face in faces:
    for img in face:
        X[num, :] = img[:,:,0].flatten()
        num = num + 1
face_labels = np.array(labels)
```

一般而言，图像的矩阵表示维度很高，而PCA是传统图像降维中的常用方法。

在写PCA代码之前，我们需要先讨论一下**样本矩阵**和对应的**方差矩阵**，这里涉及到一些细节问题。

值得注意的是，图像具有典型的二维空间结构。在PCA使用初期，是将图像拉伸成一个行向量，然后按列排布成一个样本矩阵。如下所示

<img src="images/3.png" width="30%">

$\phi_0,\phi_1,,,\phi_{M-1}$表示同一个样本的多个特征。
$x_1,x_2,,,x_N$表示$N$个样本。

协方差矩阵$\sum = \Phi^T * \Phi$

<font size=0.5> 

$$=\begin{bmatrix}
 \mathrm{E}[(X_1 - \mu_1)(X_1 - \mu_1)] & \mathrm{E}[(X_1 - \mu_1)(X_2 - \mu_2)] & \cdots & \mathrm{E}[(X_1 - \mu_1)(X_n - \mu_n)] \\ \\
 \mathrm{E}[(X_2 - \mu_2)(X_1 - \mu_1)] & \mathrm{E}[(X_2 - \mu_2)(X_2 - \mu_2)] & \cdots & \mathrm{E}[(X_2 - \mu_2)(X_n - \mu_n)] \\ \\
 \vdots & \vdots & \ddots & \vdots \\ \\
 \mathrm{E}[(X_n - \mu_n)(X_1 - \mu_1)] & \mathrm{E}[(X_n - \mu_n)(X_2 - \mu_2)] & \cdots & \mathrm{E}[(X_n - \mu_n)(X_n - \mu_n)]
\end{bmatrix}$$

可以看出$\sum$是一个实对称矩阵($\sum = \sum^T$)，
   满足性质： n阶实对称矩阵必可对角化。

$\sum$可以进行SVD分解$\sum = U \Lambda U^T$，其中$U$是正交矩阵

直接使用numpy.linalg.svd()函数，便可以对方差矩阵进行SVD分解，此时返回的特征值按照大小绛序排列好。

1. 当 样本数量 N >= 特征维数 M 时，随意取出特征向量前 K 列，即可达到PCA的要求。
2. 当 样本数量 N < 特征维数 M 时，提取的特征向量数量 K 不能超过 M，否则是无意义的。


```python
def norm_img(gray):
    g_max = np.max(gray)
    g_min = np.min(gray)
    gray = (gray - g_min) / (g_max - g_min) * 255
    return gray.astype('uint8')
```


```python
def show_result(X):
    show_num = 0
    for i in range(X.shape[0]):
        if i % 10 == 0:
            img = X[i,:,:]
            show_num = show_num + 1
            plt.subplot(4,10,show_num)
            plt.axis('off')
            plt.imshow(norm_img(img), cmap ='gray')
    plt.show()
    return
```


```python
def PCA(X):
    X_mean = np.mean(X, 0)
    X_std = np.std(X, 0)
    X = (X - X_mean)#/X_std
    #cov_X = X^T * X / X.shape[0]
    cov_X = np.cov(X, rowvar=0)
    U,_,_ = np.linalg.svd(cov_X)
    return U
```


```python
start = dt.datetime.now()
U = PCA(X)
end = dt.datetime.now()
print "PCA time:", (end - start)
```

    PCA time: 0:08:19.265142



```python
def reconstruct(X, U, k):
    Z = np.dot(X, U[:, 0:k])
    X_pca = np.dot(Z, U[:, 0:k].T)
    return X_pca
```


```python
k = 15
```


```python
X_pca = reconstruct(X, U, k)
X_pca = X_pca.reshape(-1, img_h, img_w)
show_result(X_pca)
```


![png](images/output_images/output_49_0.png)



```python
show_num = 0
for i in range(6):
    img = U[:,i].reshape(img_h,img_w)
    show_num = show_num + 1
    plt.subplot(1,6,show_num)
    plt.axis('off')
    plt.imshow(img, cmap ='gray')
plt.show()
```


![png](images/output_images/output_50_0.png)


在上面构建样本矩阵的过程中，是将一张图像展平为向量作为一行。假设图像的行数为$n$，列数为$n$，那么协方差矩阵的大小为$nm * nm$。这样做显然是一方面破坏了样本的二维结构，另一方面导致了巨大的计算量。为了解决这些问题，人们提出了二维PCA。

### 2DPCA

我们设数据矩阵为$A$，映射向量为$X$，映射后的矩阵为$Y$，我们如何确定一个好的映射向量$X$，使得映射后的矩阵$Y$的方差最大？

$$J(X) = tr(S_x)$$

$$S_x = E(Y - E(Y))(Y - E(Y)^T) = E[AX - E(AX)][AX - E(AX)]^T = E[(A -E(A))X][(A - E(A))X]^T$$

$$tr(S_x) = X^TE[(A - E(A))^T (A - E(A))]X$$

令$G_t$为数据$A$的方差矩阵

$$G_T = E[(A - E(A))^T (A - E(A))]$$

计算方法为：$$G_T = \frac{1}{N} \sum^N_{j=1} (A_j - \bar{A})^T(A_j - \bar{A})$$

那么： $$J(X) = X^T G_T X$$

欲使上面的方差$X^TG_TX$最大，那么由最开始处的证明可知，$X$需要取$G_T$的最大特征值对应的特征向量。

该方法是基于矩阵形式的推导，不需要将图像展平为向量形式，不会破坏图像的结构。


```python
def twoD_PCA(X):
    X_mean = np.mean(X,0)
    X = X - X_mean
    G_t = np.zeros((X.shape[2], X.shape[2]))
    for i in range(X.shape[0]):
        A = X[i,:,:]
        G_t = G_t + np.dot(A.T, A)
    G_t = G_t / X.shape[0]
    U,_,_ = np.linalg.svd(G_t)
    return U
```


```python
X = X.reshape(-1, img_h, img_w)
start = dt.datetime.now()
U = twoD_PCA(X)
end = dt.datetime.now()
print "2D PCA time:", end - start
```

    2D PCA time: 0:00:00.052070


可以看出这种方法使得方差矩阵与图片大小一致，在速度上也更有优势。


```python
X_2Dpca = reconstruct(X, U, k)
show_result(X_2Dpca)
```


![png](images/output_images/output_66_0.png)


实际上，上面的推导过程中，$G_T$的计算方法是不严谨的，我们假设$A_j$是任意一张图片，为了方便可视化说明，令$A_j$大小为$2*4$，而$A^{'} = A_j - \bar{A}$，那么$A^{'}$大小也为$2*4$，$A^{'T} A^{'}=$

<img src="images/4.png" width="30%">

我们以上图中左边矩阵中，左上角第一个像素点为例，它只和右边矩阵中红色那两个像素点相乘了。也就是说，$(A_j - \bar{A})^T(A_j - \bar{A})$只是求了行（列）方差，并没有求出全图方差。

### Two-directional two-dimensional PCA

针对这一问题，人们就提出了双向二维PCA。

$$G_T = \frac{1}{N} \sum^N_{j=1} (A_j - \bar{A})^T(A_j - \bar{A})$$

$$G_Q = \frac{1}{N} \sum^N_{j=1} (A_j - \bar{A})(A_j - \bar{A})^T$$

$G_T，G_Q$求得的特征映射矩阵分别为$Z，X$

降维映射结果：$$Y = Z^TAX$$

图片重建结果： $$\hat{A} = ZYX^T$$


```python
def twoDtwoD_PCA(X):
    X_mean = np.mean(X,0)
    X = X - X_mean
    G_q = np.zeros((X.shape[1], X.shape[1]))
    G_t = np.zeros((X.shape[2], X.shape[2]))
    for i in range(X.shape[0]):
        A = X[i,:,:]
        G_q = G_q + np.dot(A, A.T)
        G_t = G_t + np.dot(A.T, A)
    G_q = G_q / X.shape[0]
    G_t = G_t / X.shape[0]
    Z,_,_ = np.linalg.svd(G_q)
    U,_,_ = np.linalg.svd(G_t)
    return Z, U
```


```python
def reconstruct_2D(A, U, Z, k):
    Y = np.dot(Z[:,0:k].T ,np.dot(A, U[:,0:k]))
    A_ = np.dot(Z[:,0:k], np.dot(Y, U[:,0:k].T))
    return A_
```


```python
X = X.reshape(-1, img_h, img_w)
start = dt.datetime.now()
Z, U = twoDtwoD_PCA(X)
end = dt.datetime.now()
print "Two-directional two-dimensional PCA time:", end - start
```

    Two-directional two-dimensional PCA time: 0:00:00.076733



```python
X_2D2Dpca = np.zeros(X.shape)
for i in range(X.shape[0]):
    X_2D2Dpca[i,:,:] = reconstruct_2D(X[i,:,:], U, Z, 50)
show_result(X_2D2Dpca)
```


![png](images/output_images/output_80_0.png)


总结一下，可以看出上面三种方案的区别：
1. PCA是将所有信号都展成一维进行处理，处理完成之后再恢复成高维。
2. 2D PCA 提出的算法在信号的原始维度上面进行处理，但只是压缩了一个维度。
3. 2D 2D PCA 也是在信号的原始维度上面进行处理，并且压缩了两个维度。

基于PCA的这一类算法，是属于数据的无监督处理算法，优化目标是使映射后的数据方差最大。

将PCA用于图像分类时，需要记住PCA是基于一个假设： **对分类有用的信息位于降维所选择的维度上**

**当没有任何假设的信息模型时，PCA很有可能丢掉极为重要的信号**

### 一些有趣的小试验

如果我们在一个非常简单的数据集（例如手写数字）上面求得降维矩阵，然后用于其他的自然图片时，会发生什么现象呢？能降维和重建吗？


```python
X = np.loadtxt("data/mnist2500_X.txt");
T = np.zeros((2500,100,100))
for i in range(X.shape[0]):
    img = X[i,:].reshape(28,28)
    img = img[:,:,np.newaxis]
    img = cv2.resize(img,(100,100))
    T[i,:,:] = np.squeeze(img)
Z, U = twoDtwoD_PCA(T)
img = cv2.imread("./data/others/4.jpeg")
img = cv2.resize(img, (100,100))
img = img[:,:,0]
P = reconstruct_2D(img,U,Z,100)
plt.figure(figsize=(6, 6)) 
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(P)
plt.show()
```


![png](images/output_images/output_87_0.png)



```python
#Z = np.random.random((112,112))
#U = np.random.random((92,92))
num = 1
for i in range(5):
    path = "./data/others/" + str(i+1) + ".jpeg"
    t_0 = cv2.imread(path)
    t_0 = cv2.resize(t_0,(Z.shape[0],U.shape[0]))
    plt.subplot(2,5,num)
    plt.axis("off")
    plt.imshow(t_0)
    t_1 = np.zeros(t_0.shape)
    for i in range(3):
        t_1[:,:,i] = reconstruct_2D(t_0[:,:,i], U, Z, 30)
    plt.subplot(2,5,num+5)
    plt.axis("off")
    plt.imshow(norm_img(t_1))
    num = num + 1
plt.show()
```


![png](images/output_images/output_88_0.png)


可以看出，上述简单数据集求得的降维矩阵，可以挺好的用于其他的一些图片，说明了降维矩阵具有很强的普适性。

**降维矩阵的普适性说明其是过完备的，在求降维矩阵的过程中还可以施加更多的约束**，我们还有很大的操作空间，为下面的内容埋下了伏笔。

## 第二部分： 非线性降维

### 稀疏 PCA

上述的PCA方法是通过将图像数据通过线性变换，映射到其他线性空间。在这样的空间中，会造成图片直观表示的混乱和耦合。例如


```python
test_matrix = np.ones((3,3))
before_PCA = np.pad(test_matrix,((3,3),(3,3)),'constant',constant_values = (0,0))
right_matrix = np.random.rand(9,4)
left_matrix = np.random.rand(9,4)
after_PCA = left_matrix.T.dot(before_PCA.dot(right_matrix))
plt.title("PCA")
plt.subplot(1,2,1)
plt.imshow(before_PCA)
plt.subplot(1,2,2)
plt.imshow(after_PCA)
plt.show()
```


![png](images/output_images/output_94_0.png)


可以看出，图像中降维前的阶跃信号在降维后呈现高度耦合，不具备明显的可解释性，不利于降维后的后续分析，这不是我们所需要的。

**对于一张图像，我们希望的主成份分析是达到以下效果:**

<img src="images/6.png" width="50%">

映射过后的稀疏向量更加有利于图像的后续分析工作。

###### 让我们从另一个角度看待这个问题

图像的稀疏表达一般都是基于如下的最小化问题： $$E(D,Z) = \frac{1}{2}||X-DZ||^2_F + \psi(D) + \phi(Z)$$

其中$X$是我们观测到的数据特征矩阵，$D$是降维矩阵，可以称为“字典”，$Z$是$X$在字典$D$上的描述。$\psi(D),\ \phi(Z)$就是对字典和特征描述的约束。

对于稀疏PCA，就是令$$\psi(Z) = \lambda\sum^n_{i=1}||z_i||_1$$

使得基于字典的描述尽可能的稀疏，这样就可以减少耦合性，达到上述图片中的描述效果。

所以Sparse PCA是一个LASSO问题，要求求得的图像描述特征满足以下两个要求：
1. 最小重建误差（最大方差）
2. 描述尽量稀疏

LASSO问题的求解一般基于迭代方法，已经有现成工具包，此处不多做介绍。

###### 结构性稀疏模型

上述的稀疏模型可以求得图像的稀疏描述，可是没有对字典提出要求。我们在分析图像的时候，很希望字典具有可解释性，例如，我们希望模型可以这样描述一张人脸： 一个由两只耳朵，两个眼睛，一个鼻子和一个嘴巴构成的东西。如下图所示：

<img src="images/7.png" width="30%">

这个时候，我们需要对字典施加结构性约束，这一方面衍生出许多方法。

### 局部线性嵌入（Locally Linear Embedding ）

在降维的过程中，我们如果是要求重建误差最小，那么降维后就会出现上面所说的图像表示的混乱和耦合。我们如果希望相似的样本在降维后仍然相似，应该怎么做？

首先，这里需要我们先定义什么是相似？这是一个其他问题，这里不涉及。我们直接在定义好距离函数的情况下展开讨论。

我们希望，在降维前处在一个邻域的样本，降维之后仍然处在一个邻域。

<img src="images/8.png" width="50%">

LLE假设数据在较小的局部是线性的，也就是说，某一个数据可以由它邻域中的几个样本来线性表示。若$x_1$是我们的样本，我们使用k近邻算法寻找其邻域中的样本为$x_2,x_3,x_4$，然后有$$x_1 = w_{12}x_2 + w_{13}x_3 + w_{14}x_4$$

我们希望降维后对应的数据$x^*_1,x^*_2,x^*_3,x^*_4$，也有$$x^*_1 = w_{12}x^*_2 + w_{13}x^*_3 + w_{14}x^*_4$$

更广泛的，对于某个样本$x_i$的$k$个最近邻样本集合$Q_{(i)}$，我们需要先找到他们之间的线性表示$W$，我们把这转化为一个优化问题:$$J(w) = \sum^n_{i=1}||x_i - \sum_{j \in Q_{(i)}} w_{ij} x_j||^2_2$$

其中$n$为样本数量，$w_{ij}$为权重系数，我们需要进行归一化的限制：$$\sum_{j \in Q_{(i)}} w_{ij} =1$$

那么，同时有$$x_i = \sum_{j \in Q_{(i)}} w_{ij} x_i$$

对于不在样本$x_i$邻域内的样本$x_j$，我们令对应的$w^i_{j}=0$，这样可以把$w$扩展到整个数据集的维度。

那么
$$J(w) = \sum^n_{i=1}||\sum_{j \in Q_{(i)}} w_{ij} (x_i - x_j)||^2$$
$$J(w) = \sum^n_{i=1} W^T_i(x_i - x_j)(x_i - x_j)^T W_i$$

令$Z_i = (x_i -x_j)(x_i-x_j)^T$

那么利用拉格朗日加上约束，最优化目标可以变成： $$L(W) = \sum_{i=1}^n W_i^T Z_i W_i + \lambda(W^T_i 1_k -1)$$

对$W_i$进行求导，有：$$2*Z_i W_i + \lambda = 0$$

求解之后再归一化即有:$$W_i = \frac{Z_i^{-1}}{\sum Z_i^{-1}}$$

在得到高维样本的线性关系以后，我们希望在保持线性关系的前提下，得到高维样本$x_i$在低维空间$y_i$的表示，可以转化为最小化如下问题：
$$J(y) = \sum^n_{i=1}||y_i - \sum^n_{i=1}w_{ij}y_i||^2$$

并且需要加入如下约束： $$\sum^n_{i=1}y_i= 0$$ $$ \frac{1}{n} \sum^n_{i=1} y_i y_i^T = I$$

整理一下，并用矩阵表示，我们将会得到下面的结果：$$L(Y) = tr(Y^T（I-W）^T(I-W)Y) + \lambda(Y^TY - nI)$$

令$M = (I-W)^T(I-W)$,这个式子对$Y$求导并令其为零，将会得到：$$MY = \lambda Y$$

这个问题便和上面的PCA正好相反，我们取矩阵$M$最小的$d$个特征值所对应的$d$个特征向量组成的矩阵$Y=(y_1,y_2,...y_d)$即可。

注意： 由于$M = (I-W)^T(I-W)$,故$M$的特征值为非负数，且$W^T e = e$,所以$（W^T - I）e = 0$, $(I-W)^T(I-W)e = 0e$

所以$M$的最小特征值为0，不能反应数据特征，此时对应的特征向量为全1。我们通常选择$M$的第2个到第d+1个最小的特征值对应的特征向量。

局部线性嵌入有一个核心的问题就是距离函数的选取，选取不同的距离度量函数，将会得到不一样的邻域结构。这样降维形成的样本数据结构与距离度量函数的选取有直接的关系。

对于高维数据，实际上我们很难判断我们所选择的距离度量函数是否合适。

## 局部判别嵌入（Local Discriminant Embedding） 

局部判别嵌入有点类似于局部线性嵌入与线性判别分析的合体。（这一句在瞎扯）

对于有监督的图像分类问题，我们希望同类间的距离关系尽量不变的时候，也希望不同样本之间的距离尽可能的大。

由于图像是二维结构的，我们直接介绍二维形式的局部判别嵌入（LDE）。

我们有$n$张图像作为样本，表示为$\{ A_i|A_i \in \mathbb{R}^{n_1×n_2} \}^n_{i=1}$，类似于二维的PCA降维的形式，我们假设$A_i$降维后的矩阵为$B_i$，我们需要分别左乘，右乘一个矩阵： $$B_i = L^T A_i R$$

其中$L \in \mathbb{R}^{n_1×l_1},\ R \in \mathbb{R}^{n_1×l_2}, \ B \in \mathbb{R}^{l_1×l_2}$

首先，我们需要先构建两张样本数据的邻接矩阵$W，\tilde{W}$表示样本数据的距离结构，

$W$为同类间的邻接矩阵（$y_i = y_j$）

如果$A_j$是$A_i$的$k$近邻之一，$w_{ij} = e^{(-\frac{||A_i -A_j||^2}{t})}$,否则 $w_{ij} = 0$

$\tilde{W}$为异类间的邻接矩阵（$y_i \neq y_j$）

如果$A_j$是$A_i$的$k$近邻之一，$\tilde{w}_{ij} = e^{(-\frac{||A_i -A_j||^2}{t})}$,否则 $\tilde{w}_{ij} = 0$

那么优化函数可以表示为:
$$\max \ Q(L,R) = \sum_{i,j} ||L^T A_i R - L^T A_j R||^2_F \tilde{w}_{ij}$$
$$subject \ to \sum_{i,j} ||L^T A_i R - L^T A_j R||^2_F w_{ij} = 1$$

其中最大化那一项表示降维后不同类别的k近邻$B_i,B_j$之间的距离尽量大，约束项表示降维后相同类别的k近邻$B_i,B_j$之间的距离不变。

我们可以使用交叉优化来解决这个问题，也就是固定$L$的时候，对$R$求最优；固定$R$的时候，对$L$求最优。

使用拉格朗日方法，便可得到以下结果：

1. 给定$L$，求解下面式子的最大$l_2$个广义特征值对应的广义特征向量，得$R = [r_1,r_2,...,r_{l_2}]$： 

$$\big( \sum_{ij} \tilde{w}_{ij} (A_i -A_j)^T LL^T (A_i - A_j) \big)r $$ $$= \lambda_R \big( \sum_{ij} w_{ij} (A_i -A_j)^T LL^T (A_i - A_j) \big)r$$

2. 给定$R$，求解下面式子的最大$l_2$个广义特征值对应的广义特征向量，得$L = [l_1,l_2,...,l_{l_1}]$： 

$$\big( \sum_{ij} \tilde{w}_{ij} (A_i -A_j) RR^T (A_i - A_j)^T \big)l $$ $$= \lambda_L \big( \sum_{ij} w_{ij} (A_i -A_j) RR^T (A_i - A_j)^T \big)l$$

重复1,2两步直到收敛。

注意： 如果$Ax = \lambda Bx$,那么$\lambda$称作矩阵$A$相对于矩阵$B$的广义特征值，$x$称作对应的广义特征向量

然后我们可以求得降维后的矩阵$B_i = L^T A_i R$


```python
import sklearn.datasets as ds

digits_w = 8
digits_h = 8
digits_num = 500
digits = ds.load_digits()
rand_i = np.random.choice(range(digits.data.shape[0]), digits_num)
D = digits.data[rand_i, :]
D = D - np.mean(D, axis=0)
labels = digits.target[rand_i]
print D.shape,labels.shape
num = 1
for i in range(10):
    plt.subplot(2,5,num)
    plt.imshow(digits.images[i,:,:], cmap='gray')
    num = num + 1
plt.show()
```

    (500, 64) (500,)



![png](images/output_images/output_157_1.png)



```python
def calculate_knn(dis_matrix, k):
    n = dis_matrix.shape[0]
    dis_matrix[dis_matrix==0] = float('inf')
    knn_dis = np.zeros(dis_matrix.shape)
    for i in range(n):
        sorted_row = np.sort(dis_matrix[:,i])
        knn_dis[:,i] = (dis_matrix[:,i]<sorted_row[k]) * dis_matrix[:,i]
        knn_dis[np.isnan(knn_dis)] = 0
    return knn_dis
```


```python
def calculate_W(dis, k, labels):
    #Construct neighborhood graphs W and W'. 
    W = calculate_distance(labels[:,np.newaxis])
    W[W > 0] = 1
    W1 = calculate_knn(dis*(1-W),k) #同类邻接矩阵
    W2 = calculate_knn(dis* W   ,k) #异类邻接矩阵
    return W1, W2
```


```python
def cross_optimization(A, M, W1, W2, arg):
    n = A.shape[0]
    if arg['direction'] == 'R':
        shape = (A[0].shape[1], A[0].shape[1])
    elif arg['direction'] == 'L':
        shape = (A[0].shape[0], A[0].shape[0])
    else:
        raise RuntimeError('direction Error')
    A_diff = np.zeros(shape)
    A_same = np.zeros(shape)
    MMT = M.dot(M.T)
    for i in range(n):
        for j in range(n):
            A_ij = A[i,:,:] - A[j,:,:]
            if arg['direction'] == 'R':
                A_same = A_same + W1[i,j] * A_ij.T.dot( MMT.dot( A_ij))
                A_diff = A_diff + W2[i,j] * A_ij.T.dot( MMT.dot( A_ij))
            elif arg['direction'] == 'L':
                A_same = A_same + W1[i,j] * A_ij.dot( MMT.dot( A_ij.T))
                A_diff = A_diff + W2[i,j] * A_ij.dot( MMT.dot( A_ij.T))
            else:
                raise RuntimeError('direction Error')
    eigvals,eigvecs = eigh(A_diff, A_same)
    k =  arg['dims'] + 1
    return norm_eigvector(eigvecs[:, -k :-1])
```


```python
from scipy.linalg import eigh
knn = 5
dis = kernel(X, 'gauss')
W1, W2 = calculate_W(dis, knn, labels)
A = D.reshape(digits_num, digits_w, digits_h)
L = np.eye(digits_w)
R = np.eye(digits_h)
num = 0
while(num < 50):
    if num == 0:
        old_L = float('inf') 
        old_R = float('inf') 
    else:
        old_L = L
        old_R = R
    num = num + 1
    arg_R = {'direction': 'R', 'dims': 4}
    R = cross_optimization(A, L, W1, W2, arg_R)
    arg_L = {'direction': 'L', 'dims': 5}
    L = cross_optimization(A, R, W1, W2, arg_L)
    
    R_error = np.sum(np.power(old_R-R,2))
    L_error = np.sum(np.power(old_L-L,2))
    if R_error < 1e-8 and L_error < 1e-8:
        break
    print num, "     ", R_error, "     ", L_error
```

    /usr/local/lib/python2.7/dist-packages/ipykernel_launcher.py:7: RuntimeWarning: invalid value encountered in multiply
      import sys


    1       inf       inf
    2       8.143568068658011       13.95277604652349
    3       9.687227223471496       13.537627102765676
    4       10.059768257925931       10.961763149438688
    5       8.75916294328023       8.17944227232135
    6       2.5366951331427607       8.177628516609428
    7       8.337803373768917       11.18026299791158
    8       10.589067085616097       10.987552287577156
    9       6.557434757429012       8.78443355276319
    10       8.822250068157064       9.407810726134976
    11       7.65591748130174       8.578277229835585
    12       13.530509557850532       12.51740629181214
    13       10.47128227188409       6.802668904343428
    14       12.738467283492916       7.035613159090085
    15       13.595310519326942       10.240708841333333
    16       10.039021913224957       6.5491720190509515
    17       10.07450602000219       12.000902913604332
    18       13.805785449745782       8.701753201279349
    19       14.084854166499605       6.256375531034851
    20       13.438579374478671       11.953539007316714
    21       12.755735316253574       11.591308147739468
    22       10.74870553538513       11.402379086812289



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-24-64423b4238ed> in <module>()
         22     num = num + 1
         23     arg_R = {'direction': 'R', 'dims': 4}
    ---> 24     R = cross_optimization(A, L, W1, W2, arg_R)
         25     arg_L = {'direction': 'L', 'dims': 5}
         26     L = cross_optimization(A, R, W1, W2, arg_L)


    <ipython-input-23-c34fccc61d81> in cross_optimization(A, M, W1, W2, arg)
         14             A_ij = A[i,:,:] - A[j,:,:]
         15             if arg['direction'] == 'R':
    ---> 16                 A_same = A_same + W1[i,j] * A_ij.T.dot( MMT.dot( A_ij))
         17                 A_diff = A_diff + W2[i,j] * A_ij.T.dot( MMT.dot( A_ij))
         18             elif arg['direction'] == 'L':


    KeyboardInterrupt: 


## 第三部分：核方法

### kernel PCA

kernel PCA是使用核方法对PCA的非线性扩展，即将原数据通过核映射到再生核希尔伯特空间后再使用原本线性的主成分分析。

一般而言，若$N$个数据点在$d<N$维空间中是线性不可分的，但它们在$d >= N$维空间中则是几乎必然线性可分的。

故我们可以使用核方法将数据映射到高维空间。如将$x_i$通过某一函数$\Phi$映射到高维空间中成为$\Phi(x_i)$。

但是，由于我们不知道映射后$\Phi(x_i)$的维度，我们无法再像在线性PCA中那样显式地对协方差进行特征分解（我们不知道主空间的维度，也很难求出主向量）。核PCA方法不直接计算主成分，而是计算数据点在这些主成分上的投影。

关于如何计算数据点在这些主成分上的投影，有一些证明，[这里](https://www.ics.uci.edu/~welling/classnotes/papers_class/Kernel-PCA.pdf)介绍的很详细，下面我只叙述一下算法流程。

$x_i,y_i$为样本数据，
1. 计算$K$矩阵，如$k(\boldsymbol{x_i},\boldsymbol{y_i}) = e^\frac{-||\boldsymbol{x_i} - \boldsymbol{y_i}||^2}{2\sigma^2}$
2. 数据中心化，$K' = K - \mathbf{1_N} K - K \mathbf{1_N} + \mathbf{1_N} K \mathbf{1_N}$
3. 对$K'$进行特征分解，取前k个最大特征值对应的特征向量。

注意：Kernel PCA与之前的PCA方法并不相同。
1. 一般而言，Kernel PCA是用来做分类中的线性不可分的高维数据，所以没有考虑到对于原始数据的保留，也就是说没有考虑到重建。
2. 当样本量较大时，$K$计算量较大。


```python
def calculate_distance(A):
    n = A.shape[0]
    A2 = np.sum(A*A, axis=1, keepdims=True) 
    Ar_2 = np.repeat(A2, n, axis=1)
    Ac_2 = np.repeat(A2.T, n, axis=0)
    dis = Ar_2 + Ac_2 - 2*A.dot(A.T)
    dis[ dis<0 ] = 0
    return np.sqrt(dis)
```


```python
def get_sigmoid(x):
    x[x <= 0] = float('inf') 
    x = np.min(x, 1)
    sigmoid = 5 * np.mean(x)
    return sigmoid
```


```python
def norm(x):
    x_min, x_max = x.min(), x.max()
    return (x - x_min)/(x_max - x_min)
```


```python
def kernel(A, para):
    x = calculate_distance(A)
    if para == 'square':
        f = norm(x*x)
    elif para == 'gauss':
        sigmoid = get_sigmoid(x)
        f = np.exp(- x*x / (2*sigmoid*sigmoid))
    else:
        f = norm(x.dot(x))
    return f
```


```python
def centered_data(x):
    n = x.shape[0]
    a = np.ones(n)/n
    return (x - a*x -x*a + a*x*a)
```


```python
def norm_eigvector(U):
    row_sum = np.sqrt(np.sum(U*U, 0, keepdims=True))
    norm_U = U / row_sum
    return norm_U
```


```python
k = 3
K = kernel(D,'square')
K = centered_data(K)
U,_,_ = np.linalg.svd(K)
U = norm_eigvector(U)
Y = K.dot(U[:, 0:k]) 
```


```python
import matplotlib.cm as plt_cm
import matplotlib.colors as plt_col
from mpl_toolkits.mplot3d import Axes3D

def plot_dims(Y, labels):
    l = np.max(Y[:,0]) - np.min(Y[:,0])
    w = np.max(Y[:,1]) - np.min(Y[:,1])
    h = np.max(Y[:,2]) - np.min(Y[:,2])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(elev=45, azim=0)
    ax.set_xlim([np.min(Y[:,0]) - 0.1 * l, np.max(Y[:,0]) + 0.1 * l])
    ax.set_ylim([np.min(Y[:,1]) - 0.1 * w, np.max(Y[:,1]) + 0.1 * w])
    ax.set_zlim([np.min(Y[:,2]) - 0.1 * h, np.max(Y[:,2]) + 0.1 * h])
    cmap = plt_cm.jet
    norm = plt_col.Normalize(vmin=0, vmax=9)
    mapper = plt_cm.ScalarMappable(cmap=cmap, norm=norm)
    colors = mapper.to_rgba(range(10))
    for i in range(Y.shape[0]):
        ax.text(Y[i,0], Y[i,1], Y[i,2], labels[i], color=colors[int(labels[i])])
    plt.show()
    return

plot_dims(Y, labels)
```


![png](images/output_images/output_178_0.png)


## 第四部分：PCA的与其他方法的结合

### 极大似然PCA

作为概率PCA的一种，使用极大似然确定模型参数。

假设$z$的先验分布是均值为零，方差为单位方差的高斯分布。$$p(z) = N(z \ |\ 0,I)$$

观测变量$x$的条件分布也是高斯分布：$$p(x|z) = N(x|Wz + \mu, \sigma^2 I)$$

所以模型一共要估计三个参数：$W,\mu , \sigma$

$$\mu = \bar{x}$$ $$W_{ML} = U_M(L_M - \sigma^2I)^{\frac{1}{2}}R$$ $$\sigma^2_{ML} = \frac{1}{D-M}\sum^D_{i=M=1}\lambda_i$$

## PCANet

上述的PCA仅仅是一个降维过程，在2015年深度学习流行的时候，PCA也有作者推出了PCANet，结构图如下所示：

<img src="images/5.png" width="70%">

PCANet这篇文章算是CNN与PCA的一个很好的结合的例子， [这里是原本的文章](https://arxiv.org/abs/1404.3606)，[这里有中文翻译](https://blog.csdn.net/u013088062/article/details/50039573)

为了不和网络上已有的部分做重复的工作，接下来，我将会简单介绍PCANet的工作原理，然后讨论其代码实现和相关优缺点。

该网络共有两层，两层所进行的操作类似，我们只介绍第一层。

PCANet所不同于一般CNN是由于PCANet中的卷积核是通过PCA方法求取出来的特征向量，而不是CNN中那种迭代自学习产生。那么，如何通过PCA求取卷积核呢？

首先我们需要构建样本矩阵，样本的大小等于卷积核的大小，PCANet文章中是遍历一张


```python
#TODO
```

下面是乱七八糟


```python
images_path = './data/plates'
imgs_dir = os.listdir(images_path)
plates = list()
labels = list()
for list_dir in imgs_dir:
    imgs = list()
    temp_path = images_path + '/' + list_dir
    if os.path.isdir(temp_path):
        label_num = int(list_dir)
        imgs_list = os.listdir(temp_path)
        for img_name in imgs_list:
            img_path = temp_path + '/' + img_name
            img = cv2.imread(img_path)
            imgs.append(img)
            labels.append(label_num)
    plates.append(imgs)
```


```python
img_w = 100
img_h = 100
img_num = len(labels)
X = np.zeros((img_num, img_w, img_h))
num = 0
for plate in plates:
    for img in plate:
        X[num, :,:] = np.mean(cv2.resize(img, (img_w, img_h)), 2)
        num = num + 1
labels = np.array(labels)
```


```python
k = 20
P = np.zeros((img_num, k*k))
for i in range(img_num):
    temp = np.dot(Z[:,0:k].T ,np.dot(X[i,:,:], U[:,0:k]))
    P[i,:] = np.resize(temp, k*k)
```


```python
k = 3
K = kernel(P,'gauss')
K = reduce_mean(K)
U,_,_ = np.linalg.svd(K)
U = norm_eigvector(U)
Y = K.dot(U[:, 0:k]) 
```


```python
import scipy.io as scio
scio.savemat('./data/a.mat', {'X':Y, 'Y':labels})
```


```python
import random
temp_data = list(zip(Y, labels))
random.shuffle(temp_data)
Y[:,:], labels[:] = zip(*temp_data)
```


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
X_train = Y[0:60,:]
y_train = labels[0:60]
X_test = Y[60:,:]
y_test = labels[60:]
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
predict = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, predict)
print 'accuracy: %.2f%%' % (100 * accuracy)
```

    accuracy: 13.33%


### 参考文献

Kernel Principal Components Analysis

Sparse Principal Component Analysis

Structured Sparse Principal Component Analysis

2DPCANet: a deep leaning network for face recognition 

PCANet: A Simple Deep Learning Baseline for Image Classification?

Two-directional two-dimensional PCA for efficient face representation and recognition.pdf

Two-Dimensional PCA:A New Approach to Appearance-Based Face Representation and Recognition

Kernel Principal Component Analysis and its Applications in Face Recognition and Active Shape Models

https://zh.wikipedia.org/wiki/%E6%A0%B8%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90

https://zh.wikipedia.org/zh-cn/%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90#cite_note-5

https://zh.wikipedia.org/zh-cn/%E5%A5%87%E5%BC%82%E5%80%BC%E5%88%86%E8%A7%A3

https://zh.wikipedia.org/zh-cn/%E5%B0%8D%E7%A8%B1%E7%9F%A9%E9%99%A3

https://blog.csdn.net/w450468524/article/details/54895477

https://blog.csdn.net/u013088062/article/details/49934065

https://blog.csdn.net/jwh_bupt/article/details/12070273

https://www.cnblogs.com/pinard/p/6266408.html

附录：

实对称矩阵有以下的性质：

1. 实对称矩阵A的不同特征值所对应的特征向量是正交的。
2. 实对称矩阵A的特征值都是实数，特征向量都是实向量。
3. n阶实对称矩阵A必可对角化。
4. 可用正交矩阵对角化。
5. K重特征值必有K个线性无关的特征向量，或者说必有秩r(λE-A)=n-k。
