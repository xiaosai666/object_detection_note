[TOC]

# CenterNet: Object as Points

参考资料：

论文链接：[paper](https://arxiv.org/abs/1904.07850) 2019年4月发布

Github：[原作者实现](https://github.com/xingyizhou/CenterNet) ；[Detectron2实现](https://github.com/JDAI-CV/centerX)

## Motivation

传统目标检测分为单阶段和双阶段方法。单阶段通过滑动窗口和anchor去做预测，产生大量的冗余proposal，导致严重计算资源浪费

Anchor-based目标检测算法在后处理阶段需要NMS去除多余检测框，算法不是end2end训练的

1.直接预测BBox中心点和尺寸

2.不需要传统NMS操作

## 原理

输入图片尺寸为：$W \times H \times3$ ，表示为 $I \in R^{W \times H \times 3}$

输出HeatMap，尺寸为：$\frac{W}{R} \times \frac{H}{R}$，通道数$C$为类别数，heatmap中的每个值在$[0,1]$区间，表示为：
$$
Y \in[0,1]^{\frac{W}{R} \times \frac{H}{R} \times C}
$$
$R$ 的默认值为：$R=4$ ，即网络输出的heatmap的$H $和$W $为输入图像的$\frac{1}{4}$

$Y_{x, y, c}$ 表示为heatmap中 $(x, y)$处第$c $通道的值

当$Y_{x, y, z}=1$时，代表heatmap的$(x, y)$处是一个关键点，类别为$c $

当$Y_{x, y, z}=0$时，代表heatmap的$(x, y)$处是背景

需要对label中的bbox坐标进行计算，得到ground truth关键点 $p \in \mathcal{R}^{2}$ 
$$
p=\left(\frac{x_{1}+x_{2}}{2}, \frac{y_{1}+y_{2}}{2}\right)
$$
因为对原始图像进行了$R$ 倍下采样，同理对 $p$ 进行处理：$\tilde{p}=\left\lfloor\frac{p}{R}\right\rfloor$ ，最终得到低分辨率的关键点

关键点使用一个高斯核分布到heatmap中
$$
Y_{x y c}=\exp \left(-\frac{\left(x-\tilde{p}_{x}\right)^{2}+\left(y-\tilde{p}_{y}\right)^{2}}{2 \sigma_{p}^{2}}\right)
$$
其中，$\sigma_{p}$ 与目标的尺寸相关的标准差。如果$(x, y)$处可以从多个同类别目标得到多个值，取**最大值**。

## 损失函数

CenterNet的损失函数由三部分构成：**关键点损失、中心点偏置损失和目标尺寸损失**

### 关键点损失

参考Focal Loss，构造如下损失函数：
$$
L_{k}=\frac{-1}{N} \sum_{x y c}\left\{\begin{array}{cl}
\left(1-\hat{Y}_{x y c}\right)^{\alpha} \log \left(\hat{Y}_{x y c}\right) & \text { if } Y_{x y c}=1 \\
\left(1-Y_{x y c}\right)^{\beta}\left(\hat{Y}_{x y c}\right)^{\alpha} \log \left(1-\hat{Y}_{x y c}\right) & \text { otherwise }
\end{array}\right.
$$
$\alpha=2$ 和 $\beta=4$ 是Focal Loss的超参数，$N$ 是图像 $I$ 的的关键点数量，$ \hat{Y}_{x y c}$为预测值，$ Y_{x y c}$为heatmap值

若先将$\left(1-Y_{x y c}\right)^{\beta}$忽略，可将上述公式转化为：
$$
\begin{array}{c}
L_{k}=\left(1-P_{t}\right)^{\alpha} * \log P_{t} \\
P_{t}=\left\{\begin{array}{cc}
\hat{Y}_{x y c}, & \text { if } Y_{x y c}=1 \\
1-\hat{Y}_{x y c}, & \text { otherswise }
\end{array}\right.
\end{array}
$$
$\log P_{t}$为标准**交叉熵损失函数**，$\left(1-\hat{Y}_{x y c}\right)^{\alpha}$ 和 $\left(\hat{Y}_{x y c}\right)^{\alpha}$ 是Focal loss项，存在如下影响：

- 当$Y_{x y c}=1$时，当$ \hat{Y}_{x y c}$值接近1时，由于$\left(1-\hat{Y}_{x y c}\right)^{\alpha}$ 项，损失函数会乘一个很小的系数，急剧衰减；

- 当$Y_{x y c}=1$时，当$ \hat{Y}_{x y c}$值不接近1时，由于$\left(1-\hat{Y}_{x y c}\right)^{\alpha}$ 项，损失函数轻微衰减；
- 当$Y_{x y c} \neq 1$时，由于$\left(\hat{Y}_{x y c}\right)^{\alpha}$，$ \hat{Y}_{x y c}$的值越接近0，权重越小，损失函数所占比重越小
- 当$Y_{x y c} \neq 1$时，由于$\left(1-Y_{x y c}\right)^{\beta}$项，也就是当当前点离目标点越远，$\left(1-Y_{x y c}\right)^{\beta}$越大，反之，当前点离目标点越近，$\left(1-Y_{x y c}\right)^{\beta}$越小，损失函数所占比重越小（平衡正负样本：每个目标只有一个中心点正样本，其余点全是负样本）
- $\left(\hat{Y}_{x y c}\right)^{\alpha}$与$\left(1-Y_{x y c}\right)^{\beta}$协同作用

### 偏置损失

由于对图像进行了 $R=4$ 的下采样，因此在把特征图映射到原始图像上，会产生精度误差。因此，对于每个中心点，额外增加了一个偏置：$\hat{O} \in \mathcal{R}^{\frac{W}{R} \times \frac{H}{R} \times 2}$ 去补偿。这个偏置值offset用**L1 loss**来训练，公式如下：
$$
L_{o f f}=\frac{1}{N} \sum_{p}\left|\hat{O}_{\tilde{p}}-\left(\frac{p}{R}-\tilde{p}\right)\right|
$$
上式中$\hat{O}_{\tilde{p}}$表示网络预测的offset， $\left(\frac{p}{R}-\tilde{p}\right)$可以根据训练集的标注信息得到。需要特别指出的是，offset损失只针对heatmap中的关键点，对于非关键点，不存在offset损失。

### 尺寸损失

令$\left(x_{1}^{(k)}, y_{1}^{(k)}, x_{2}^{(k)}, y_{2}^{(k)}\right)$表示第$k$个目标的bbox的左上角和右下角坐标，所属类别为 $c_{k}$,中心点坐标为 $p_{k}=\left(\frac{x_{1}^{(k)}+x_{2}^{(k)}}{2}, \frac{y_{1}^{(k)}+y_{2}^{(k)}}{2}\right)$，然后对每个目标 $k$进行回归，最后回归到 $s_{k}=(x_{2}^{(k)}-x_{1}^{(k)},y_{2}^{(k)}-y_{1}^{(k)})$，这个值是在训练前提前计算得到的，是进行了下采样后的长宽值。采用**L1 loss**训练，公式如下：
$$
L_{\text {size }}=\frac{1}{N} \sum_{k=1}^{N}\left|\hat{S}_{p_{k}}-s_{k}\right|
$$
其中，$\hat{S}_{p_{k}}$为预测值

### 整体损失

$$
L_{det} = L_{k}+\lambda _{size}L_{size}+\lambda _{offset}L_{offset}
$$

在论文中，$\lambda _{size}=0.1$，$\lambda _{offset}=1$，最后有三个head layer，分别输出$[128, 128, 80], [128, 128, 2], [128, 128, 2]$

## 推理

1.先对图片进行 $R=4$ 下采样，然后对下采样后的图像进行预测。对于每个类的热点单独提取出来。提取方式如下：

- 采用 $3\times 3 $ Maxpooling的方式对每个点进行判断：当前预测$ \hat{Y}_{x y c}$是否比周围8个临近的的值都大(或者等于)，然后提取 $top=100$个这样的热点

2.生成标定框
$$
(x_{i}+ \delta_{x_{i}}-w_i/2,y_{i}+ \delta_{y_{i}}-h_i/2,x_{i}+ \delta_{x_{i}}+w_i/2,x_{i}+ \delta_{x_{i}}+w_i/2)
$$
其中， $x_i,y_i$是关键点**整型**坐标 ， $\delta_{x_i},\delta_{y_i}$为预测的偏置$offset$， $w_i,h_i$ 为预测目标的长宽

3.阈值，论文中为$0.3$，也就是预测值$ \hat{Y}_{x y c}$ ，从 $top=100$个热点中，输出 $ \hat{Y}_{x y c}>0.3$ 的关键点作为最终结果

## 缺陷

在实际训练中，如果在图像中，**同一个类别**中的某些物体的GT中心点，在**下采样**时会挤到一块，也就是两个物体在GT中的中心点重叠了，CenterNet对于这种情况也是无能为力的，也就是将这两个物体的当成一个物体来训练(因为只有一个中心点)。同理，在预测过程中，如果两个同类的物体在下采样后的中心点也重叠了，那么CenterNet也是只能检测出一个中心点，不过CenterNet对于这种情况的处理要比faster-rcnn强一些的，具体指标可以查看论文相关部分。
在我的实验中，对于大目标有概率出现大量的重复检测
