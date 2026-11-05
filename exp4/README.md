# 实验四：全连接神经网络
## 一、实验目的
本实验旨在通过构建多层感知机（MLP, Multi-Layer Perceptron）模型，深入理解神经网络的基本组成与工作原理，掌握其在图像分类任务中的应用。
## 二、实验内容
### 1. 导入库
本实验使用 PyTorch 完成网络构建、损失计算和参数更新。其中，`torch.nn` 用于定义全连接层，`torch.nn.functional` 提供 ReLU 激活函数和交叉熵损失，`torch.optim` 提供优化器，`torchvision` 用于加载 MNIST 数据集和转换图像格式。NumPy 与 Matplotlib 同时导入，便于后续扩展数值处理和可视化；本次代码主要通过文字输出观察实验结果。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
```
### 2. 设置超参数
将批量大小设为 2048，表示每次参数更新最多使用 2048 张训练图片。程序根据 CUDA 是否可用选择 GPU 或 CPU，并在后续将模型与数据放到同一设备上计算。批量大小决定每轮训练的更新次数，也会影响显存占用和学习过程。

```python
batch_size = 2048
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
### 3. 加载数据
使用 MNIST 手写数字数据集，其中训练集包含 60000 张图片，测试集包含 10000 张图片，类别为数字 0～9。每张图片为 28×28 的单通道灰度图。`ToTensor()` 将图片转换为张量，并把像素值从 0～255 转为 0～1 范围内的浮点数。

通过 `DataLoader` 按批次读取数据。训练时打乱样本顺序，使模型在不同轮次以不同顺序接触训练样本；测试代码也设置了打乱顺序，但全部测试样本仍会参与准确率计算。首次运行时，训练数据加载语句会在本地缺少数据的情况下下载数据集。

```python
train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=batch_size, shuffle=True)

train_loader.dataset.data.shape,
```
```text
输出：
(torch.Size([60000, 28, 28]),)
```

上述输出表示训练数据的形状为 `[60000, 28, 28]`。外层括号和末尾逗号来自表达式 `train_loader.dataset.data.shape,`，不影响数据本身。

### 4. 创建MLP模型
构建结构为 **784 → 128 → 10** 的多层感知机。输入时将 28×28 图片展开为长度为 784 的向量，第一层全连接层将其映射为 128 个特征，再通过 ReLU 引入非线性，最后一层输出 10 个类别得分。

输出层直接返回得分，由交叉熵函数在内部完成相应的概率计算，因此无需提前添加 Softmax。预测时取最高得分对应的类别作为识别结果。优化器选用 SGD，学习率设为 0.01，用于控制每次参数调整的幅度。

```python
class mlp(nn.Module):
    def __init__(self):
        super(mlp,self).__init__()
        self.l1 = nn.Linear(784,128)
        self.l2 = nn.Linear(128,10)

    def forward(self,x):
        a1 = self.l1(x)
        x1 = F.relu(a1)
        a2 = self.l2(x1)
        return a2

model = mlp().to(device)
optimizer = optim.SGD(model.parameters(),lr=0.01)
model
```
```text
输出：
mlp(
  (l1): Linear(in_features=784, out_features=128, bias=True)
  (l2): Linear(in_features=128, out_features=10, bias=True)
)
```

### 5. 模型训练与逐轮测试
共训练 30 轮，每轮依次完成训练和测试两个阶段。训练阶段调用 `model.train()`，将图片展开后输入网络，以真实标签为依据计算交叉熵损失，再依次执行梯度清零、反向传播和参数更新。这里在前向计算前调用 `optimizer.zero_grad()`，防止上一批次的梯度累积到当前批次。

测试阶段调用 `model.eval()`，并在 `torch.no_grad()` 中计算，避免构建用于反向传播的计算图。对每张测试图片，取输出得分最高的类别，与真实标签比较，得到正确识别的样本数量。准确率计算方式为：

`测试准确率 = 正确识别的测试样本数 / 测试样本总数`

每轮结束打印一次测试损失与准确率。代码中的 `epoch` 从 0 开始，因此 `epoch:29` 对应第 30 轮。输出的 `loss` 是各测试批次平均交叉熵的算术平均，并不是训练损失。

```python
epochs = 30
for epoch in range(epochs):
    model.train()
    for batch_idx,(x,y) in enumerate(train_loader):
        x,y = x.view(x.shape[0],-1).to(device),y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output,y)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx,(x,y) in enumerate(test_loader):
            x,y = x.view(x.shape[0],-1).to(device),y.to(device)
            output = model(x)
            test_loss += F.cross_entropy(output,y)
            pred = output.max(1,keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()

    test_loss = test_loss/(batch_idx+1)
    acc = correct/len(test_loader.dataset)
    print('epoch:{}, loss:{:.4f}, acc:{:.4f}'.format(epoch,test_loss,acc))
```
```text
输出：
epoch:0, loss:2.2568, acc:0.2116
epoch:1, loss:2.1984, acc:0.4305
epoch:2, loss:2.1306, acc:0.5870
epoch:3, loss:2.0525, acc:0.6595
epoch:4, loss:1.9661, acc:0.7016
epoch:5, loss:1.8731, acc:0.7184
epoch:6, loss:1.7749, acc:0.7346
epoch:7, loss:1.6751, acc:0.7475
epoch:8, loss:1.5752, acc:0.7575
epoch:9, loss:1.4785, acc:0.7635
epoch:10, loss:1.3857, acc:0.7731
epoch:11, loss:1.2983, acc:0.7805
epoch:12, loss:1.2182, acc:0.7898
epoch:13, loss:1.1455, acc:0.7988
epoch:14, loss:1.0802, acc:0.8050
epoch:15, loss:1.0214, acc:0.8116
epoch:16, loss:0.9690, acc:0.8164
epoch:17, loss:0.9216, acc:0.8222
epoch:18, loss:0.8791, acc:0.8263
epoch:19, loss:0.8423, acc:0.8311
epoch:20, loss:0.8077, acc:0.8339
epoch:21, loss:0.7777, acc:0.8389
epoch:22, loss:0.7502, acc:0.8435
epoch:23, loss:0.7257, acc:0.8462
epoch:24, loss:0.7030, acc:0.8490
epoch:25, loss:0.6822, acc:0.8507
epoch:26, loss:0.6630, acc:0.8545
epoch:27, loss:0.6459, acc:0.8577
epoch:28, loss:0.6295, acc:0.8600
epoch:29, loss:0.6149, acc:0.8616
```

## 三、实验结果与分析
### 1. 数据与模型检查
数据加载输出为 `torch.Size([60000, 28, 28])`，与 MNIST 训练集规模及图片尺寸一致。模型输出显示第一层输入 784 维、输出 128 维，第二层输入 128 维、输出 10 维，说明图片展开后的维度与网络输入一致，输出类别数也与数字类别数一致。
### 2. 测试指标变化
根据本次 Jupyter 保存的实际输出，选取部分轮次进行比较。表中轮次按实际训练次数从 1 开始计数：

| 训练轮次 | 日志中的 epoch | 测试损失 | 测试准确率 |
| --- | --- | --- | --- |
| 第 1 轮 | 0 | 2.2568 | 21.16% |
| 第 5 轮 | 4 | 1.9661 | 70.16% |
| 第 10 轮 | 9 | 1.4785 | 76.35% |
| 第 15 轮 | 14 | 1.0802 | 80.50% |
| 第 20 轮 | 19 | 0.8423 | 83.11% |
| 第 25 轮 | 24 | 0.7030 | 84.90% |
| 第 30 轮 | 29 | 0.6149 | 86.16% |

从全部 30 轮日志可以看到，测试损失逐轮下降，准确率逐轮上升。第 1 轮的准确率为 21.16%，到第 5 轮达到 70.16%，说明模型在初期较快学到了能够区分数字的像素特征。后续准确率继续提高，但提升幅度逐渐减小：第 10 轮至第 20 轮提高了 6.76 个百分点，第 20 轮至第 30 轮提高了 3.05 个百分点。

这种变化说明当前模型的学习过程较为稳定，后期提升速度放缓。不过，最后几轮损失仍在下降，不能仅凭训练达到 30 轮就判断模型已经完全收敛。

### 3. 最终识别效果
本次实验最终测试准确率为 **86.16%**，对应在 10000 张测试图片中正确识别 8616 张，错误识别 1384 张。最终测试损失为 **0.6149**。结果表明，采用一个隐藏层的全连接神经网络，已经能够学习手写数字图像与类别标签之间的关系，完成基本的十分类任务，但仍有进一步提升的空间。

本次代码没有输出混淆矩阵或错误样本图片，因此无法直接判断哪些数字最容易混淆。仅凭整体准确率，也不能认定每个数字类别都达到了相同的识别水平。

### 4. 参数影响与实验局限
本实验采用批量大小 2048、学习率 0.01 的 SGD 优化器。60000 张训练图片每轮被分为 30 个批次，30 轮共进行 900 次参数更新。较大的批量使每轮更新次数较少，结合当前学习率，可能是指标提升较慢的原因之一；具体影响仍需通过控制其他条件的对照实验验证。

全连接网络在输入时将图片展开，模型需要从像素向量中学习分类关系，没有显式利用图像的局部空间结构。后续可以尝试调整批量大小和学习率，或者引入卷积网络，并在相同评估条件下比较效果。

此外，当前代码只记录测试指标，缺少训练集与独立验证集指标，尚不能完整判断是否存在过拟合。每轮观察测试集有助于完成本次实验展示；若进一步据此选择参数，应另设验证集，并将测试集保留用于最终评估。代码对测试批次损失采用等权平均，最后一个不足 2048 张的批次与完整批次权重相同，因此该数值并非严格的逐样本平均损失，后续可按各批次样本数加权计算。

## 四、实验总结与心得
通过本次实验，我完成了从 MNIST 数据加载、全连接神经网络构建，到训练更新和测试评估的基本流程，对图像分类任务中各步骤的作用有了更具体的认识。

1. **理解了数据形状与网络结构的对应关系。** 28×28 图片需要展开为 784 维向量，隐藏层输出与下一层输入必须匹配，最终 10 个输出对应数字 0～9。检查张量形状和模型结构，是排查网络维度问题的直接方法。
2. **掌握了模型训练的基本步骤。** 前向传播得到预测得分，损失函数衡量预测与标签之间的差异，反向传播计算梯度，优化器据此更新参数。梯度清零与参数更新都需要在每个训练批次中正确执行。
3. **认识到代码位置会改变实验输出。** 将测试代码放在外层训练循环内，才能观察每一轮的效果；放到整个循环之后，则只得到最终结果。通过逐轮日志，可以更清楚地判断模型是否持续学习。
4. **学会了结合指标分析结果。** 本次准确率由 21.16% 提升至 86.16%，测试损失由 2.2568 降至 0.6149，说明模型取得了实际学习效果。同时，准确率的提升逐渐放缓，也提示后续需要结合参数对照实验寻找改进方向。
5. **认识到实验结论需要对应实际证据。** 本次结果能够说明整体识别效果，但尚不足以分析各类别错误分布或确定是否过拟合。后续可以记录训练和验证指标、绘制损失曲线，并展示混淆矩阵与错误样本，使分析更充分。

本次实验使我熟悉了 PyTorch 图像分类程序的基本组织方式，也让我体会到，合理设置参数、保持清晰的训练与测试流程，以及依据实际输出解释结果，都是完成神经网络实验的重要环节。
