[TOC]

# FCOS：Fully Convolutional One-Stage Object Detection

论文链接：[paper](https://arxiv.org/abs/1904.01355)

代码：[code](https://github.com/tianzhi0549/FCOS/)

## Motivation

需要仔细调整anchor超参数；

长宽比、大小固定泛化能力差；

产生大量的proposal都被当成负样本；

在为每个grid分配anchor时，需要进行大量且复杂的IoU计算；

## 方案

与YOLOv1相比，FCOS利用地真边界盒中的所有点来预测边界盒，并通过提出的“中心度”分支来抑制低质量检测到的边界盒。因此，FCOS能够提供类似于我们实验中基于锚的检测器的召回。

### FCNs

对于位于BBox内的每个grid都当作正样本，否则为负样本。对于 $feature$中的每个 $grid_{x,y}$对应原始 $image$ 中的$\left(\left\lfloor\frac{s}{2}\right\rfloor+x s,\left[\frac{s}{2}\right]+y s\right)$

$s$是 $stride$ ，$BBox$ 回归目标为 $(x,y)$ 到边界的距离 $t_{x,y} =(l^*, t^*, r^*, b^*)$ ,由于回归目标都是正数，为了保证网络输出，使用 $exp()$ 

损失函数为：
$$
L_{total}= L_{Focal} + {\lambda}{\mathbb{1}_{\left\{c_{x, y}^{*}>0\right\}}}L_{IoU}
$$
$\mathbb{1}_{\left\{c_{x, y}^{*}>0\right\}}$ 表示负样本不计算 $IoU$ 损失

**trick：**$IoU$可以替换为新的$IoU$损失，例如$GIoU$, $DIoU$, $CIoU$

### Multi-level Predic with FPN

采用FPN进行多级预测，$(P_3 ,  P_4, P_5,P_6,P_7)$ 其中 $P_6，P_7$ 由 $P_5$ 使用 $s=2$ 的conv得到，head共享参数。对于每个level可预测的bbox范围进行了限制，（0，64，128，256，512，∞），使用一个可学习的参数 $s_i$ 调整学习到的 $t_{x,y}$ ，将 $exp(t)$，改为了 $exp(s_it)$

## 要点（important）

### 样本选择策略

1.限制预测范围，降低混淆样本；

2.选择面积最小的GT分配给正样本；

**trick：选择GT中心点附近的grid作为正样本去做预测，类似yolov5的跨网格预测方案**

### Center-ness分支

Performance很差：**由于产生大量 $grid_{x,y}$远离预测bbox的中心的低质pred**

**解决方法：**增加一个center-ness分支，衡量预测框中心与 $grid_{x,y}$的一个归一化的系数
$$
\text { centerness }^{*}=\sqrt{\frac{\min \left(l^{*}, r^{*}\right)}{\max \left(l^{*}, r^{*}\right)} \times \frac{\min \left(t^{*}, b^{*}\right)}{\max \left(t^{*}, b^{*}\right)}}
$$
在选择检测框top时，使用 $score \times centerness^*$ 作为最后的得分，降低了$grid_{x,y}$远离center point的pred的分数，再通过nms，剔除低质量检测框。

**trick：**与reg-head共享参数可以提高0.5 AP