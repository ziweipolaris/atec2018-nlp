# ATEC2018 NLP赛题 复赛f1 = 0.7327

由于PAI平台限制，所有代码都放在一个文件里面，`pai_train.py`是获得本次比赛成绩的文件，实验共使用了4个模型，分别是自定义Siamese网络、ESIM网络、Decomposable Attention和DSSM网络。其中Siamese、ESIM和Decomposable Attention有char level和word level两个版本，DSSM网络只有char和word的合并版本。最佳记录由多个模型进行blending融合预测，遗憾没有尝试一下10fold交叉训练模型，前排貌似都用了，而且这里每个模型都只用了2个小时来训练。

模型性能比较，字符级的esim模型在这个任务中表现最佳。

| model name   | 模型输出与标签相关性r | 最优f1评分         | 取得最优f1评分的阈值 |
| ------------ | --------------------- | ------------------ | -------------------- |
| siamese char | 0.553536380131115     | 0.6971525551574581 | 0.258                |
| siamese word | 0.5308273808879237    | 0.6873517065157875 | 0.242                |
| esim char    | 0.5853469280801447    | 0.7116622491480499 | 0.233                |
| esim word    | 0.5783574742744366    | 0.7100964753080524 | 0.263                |
| decom char   | 0.5288425401105513    | 0.6825720620842572 | 0.249                |
| decom word   | 0.4943718720970039    | 0.6677430929314676 | 0.212                |
| dssm both    | 0.5638034287814917    | 0.6980098067493511 | 0.263                |


训练感受：
1. batchsize不要太大，虽然每个epoch更快完成， 但每个epoch权重更新次数变少了，收敛更慢
2. 使用[循环学习率](https://arxiv.org/abs/1506.01186)可以收敛到更好的极值点，更容易跳出局部极值，如在一个epoch中，使学习率从小变大，又逐渐变小
3. 利用[SWA](https://arxiv.org/abs/1803.05407)这种简单的模型融合方法可以获得泛化能力更好的性能，本地提升明显，但线上没有改善。


`pai_transform.py`和`pai_old.py`是两次不成功的尝试：
`pai_transform.py`试图参考fastai的ULMFiT方法，通过训练语言模型作为embedding输入，并针对当前分类任务更改网络结构以适应当前训练过程。
`pai_old.py`试图参考quora分享，使用文本特征工程进行分类。


> 模型来源siamese参考：https://blog.csdn.net/huowa9077/article/details/81082795
> ESIM网络、Decomposable Attention来自Kaggle分享：https://www.kaggle.com/lamdang/dl-models
> DSSM网络来自bird大神分享：https://openclub.alipay.com/read.php?tid=7480&fid=96
> 感谢以上！