## 本地安装
```shell
conda create -n pytorch  python=3.7
activate pytorch
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes

windows/linux:conda install pytorch torchvision cpuonly
mac:conda install pytorch torchvision
```
## pytorch语法

```python
import torch
x = torch.empty(5, 3)#创建一个未初始化的Tensor
x = torch.rand(5, 3)#随机创建一个Tensor，返回值在0-1
x = torch.zeros(5, 3, dtype=torch.long)#创建一个类型全是long的全为0的Tensor
x = torch.tensor([5.5, 3])#根据数据直接创建Tensor

x.size()#返回Tensor的维度
x.shape#返回Tensor的维度

x+=y#等价于x.add_(y)，不等价于x=x+y，存在内存的复制
x=x+y#等价于x=torch.add(x,y)

#索引类似numpy

#view()是共享内存，Tensor与Tensor.data的区别是前者会被autograd记录，后者不会，只是纯数据操作，共同点是共享内存
#若不想共享内存，建议用clone()函数
#广播机制类似numpy
#自动求梯度，对所有requires_grad=True的变量进行追踪，支持非直接初始化的变量的backward()，即求导
```
## 线性回归
损失函数使用平方损失函数
Y=XW+b
用sgd做优化更新参数

## softmax回归
损失函数用交叉熵损失函数
O=XW+b
Y=softmax(O)
用sgd做优化更新参数
用accuracy评估模型好坏

## 循环神经网络

语言模型n-gram

RNN
$H_t​=ϕ(X_t​W_{xh}​+H_{t−1}​W_{hh}​+b_h​)$
$O_t​=H_t​W_{hq}​+b_q​$

建立数据集
用list建立index到char的索引

```python
idx_to_char = list(set(corpus_chars))
```
用字典建立char到index的索引
```python
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
```
对语料建立字符和index对应
```python
corpus_indices = [char_to_idx[char] for char in corpus_chars]
```
示例输出
```
chars: 想要有直升机 想要和你飞到宇宙去 想要和
indices: [250, 164, 576, 421, 674, 653, 357, 250, 164, 850, 217, 910, 1012, 261, 275, 366, 357, 250, 164, 850]
```

### 采样
随机采样
shuffle一个跟语料长度-1一样的range序列，然后根据num_step取出，对应的语料

相邻采样
把原始样本X，按照batch_size折叠，然后再按列+num_step取样，样本示例如下
```
X:  tensor([[ 0.,  1.,  2.,  3.,  4.,  5.],
        [15., 16., 17., 18., 19., 20.]]) 
Y: tensor([[ 1.,  2.,  3.,  4.,  5.,  6.],
        [16., 17., 18., 19., 20., 21.]]) 

X:  tensor([[ 6.,  7.,  8.,  9., 10., 11.],
        [21., 22., 23., 24., 25., 26.]]) 
Y: tensor([[ 7.,  8.,  9., 10., 11., 12.],
        [22., 23., 24., 25., 26., 27.]]) 
```

循环神经网络的实现
先用one-hot向量，把corpus转成one-hot向量投入循环神经网络，然后建模，优化

为了防止梯度爆炸，裁剪梯度

用困惑度(perplexity)来评估模型好坏，即交叉熵损失函数做指数运算后的值


