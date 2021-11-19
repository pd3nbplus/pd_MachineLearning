# -*- coding: utf-8 -*-
"""
创建一个 alpha 向量并将其初始化为 0 向量
当迭代次数小于最大迭代次数时（外循环）：
     对数据中的每个数据向量（内循环）：
     如果该数据向量可以被优化：
             随机选择另一个数据向量
             同时优化这两个向量
             如果两个向量都不能被优化，退出内循环
     如果所有向量都没被优化，增加迭代次数，继续下一次循环
"""
import numpy as np
import random
import matplotlib.pyplot as plt


def loadData(filename):
    dataMat = []
    yMat = []
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip().split('\t')
        dataMat.append([float(line[0]), float(line[1])])
        yMat.append(float(line[2]))
    return np.array(dataMat), np.array(yMat)  # 大小 (100, 2) , (100,)


def showData(filename, line=None):
    dataArr, yArr = loadData(filename)
    data_class_1_index = np.where(yArr == -1)
    data_class_1 = dataArr[data_class_1_index, :].reshape(-1, 2)

    data_class_2_index = np.where(yArr == 1)
    data_class_2 = dataArr[data_class_2_index, :].reshape(-1, 2)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(data_class_1[:, 0], data_class_1[:, 1], c='r', label="$-1$")
    ax.scatter(data_class_2[:, 0], data_class_2[:, 1], c='g', label="$+1$")
    plt.legend()
    if line is not None:
        b, alphas = line
        x = np.linspace(1, 8, 50)
        w = np.sum(alphas * yArr[:, np.newaxis] * dataArr, axis=0)
        y = np.array([(-b - w[0] * x[i]) / w[1] for i in range(50)])
        y1 = np.array([(1 - b - w[0] * x[i]) / w[1] for i in range(50)])
        y2 = np.array([(-1 - b - w[0] * x[i]) / w[1] for i in range(50)])
        ax.plot(x, y, 'b-')
        ax.plot(x, y1, 'b--')
        ax.plot(x, y2, 'b--')
    plt.show()


def selectJrand(i, m):
    j = i  # we want to select any J not equal to i
    while j == i:
        j = int(random.uniform(0, m))
    return j


# 对 alpha 的修正函数
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


# 建立算法
def smoSimple(dataArr, yArr, C, toler, maxIter):
    """smoSimple
    Args:
        dataArr    特征集合
        yArr       类别标签
        C   松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
            控制最大化间隔和保证大部分的函数间隔小于 1.0 这两个目标的权重。
            可以通过调节该参数达到不同的结果。
        toler   容错率（是指在某个体系中能减小一些因素或选择对某个系统产生不稳定的概率。）
        maxIter 退出前最大的循环次数（alpha 不发生变化时迭代的次数）
    Returns:
        b       模型的常量值
        alphas  拉格朗日乘子
    """
    numSample, numDim = dataArr.shape  # 100 行 2 列

    # 初始化要求的参数值 b 和系数 alphas(数量等于样本数啦)
    b = 0
    alphas = np.zeros((numSample, 1))

    iterations = 0  # 记录迭代次数
    # 只有在所有数据集上遍历 maxIter 次，且不再发生任何 alpha 修改之后，才退出 while 循环
    while iterations < maxIter:
        """
        设置一个参数 alphaPairsChanged 记录 alpha 是否已经进行优化，每次循环开始
        记为 0，然后对整个集合顺序遍历, 如果没变化，则记为迭代一次 
        """
        alphaPairsChanged = 0
        for i in range(numSample):
            # 首先针对第 i 个样本给出我们预测值，这里没有加kernel变换，线性的
            fXi = np.sum(alphas * yArr[:, np.newaxis] * dataArr * dataArr[i, :]) + b
            # 计算 Ei，也相当于误差
            Ei = fXi - yArr[i]
            """
            #约束条件：KKT 条件
                yi*ui >= 1 and alpha = 0   正常分类
                yi*ui == 1 and 0 < alpha < C 边界上面
                yi*ui < 1 and alpha = C   边界之间
            # 0 <= alphas[i] <= C，由于 0 和 C 是边界值，已经在边界上的值不能够再
              减小或增大，因此无法进行优化，
            # yArr[i]*Ei = yArr[i]*fXi - 1，表示发生错误的概率，其给对值超出了 
              toler，才需要优化 
              比如，如果 (yArr[i]*Ei < -toler)，此时 alpha 应为该 C ,但是其值小于
              C，那就需要优化，同理如果 (yArr[i]*Ei > toler)，此时 alpha 应该为 0 ,
              但是其值却大于 0，也需要优化。
            """
            if (((yArr[i] * Ei < -toler) and (alphas[i] < C)) or
                    ((yArr[i] * Ei > toler) and (alphas[i] > 0))):
                # 到这儿说明满足了优化的条件，我们随机选取非 i 的一个点，进行优化比较
                j = selectJrand(i, numSample)

                # 预测样本 j 的结果
                fXj = np.sum(alphas * yArr[:, np.newaxis] * dataArr * dataArr[j, :]) + b
                Ej = fXj - yArr[j]

                # 更新 alpha 前先复制一下，作为 old
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # 计算 L 和 H, alpha 和 L,H 的关系是 0 <= L <= alpha <= H <= C
                # 异号的情况, alpha 相减, 否则同号，相加
                if yArr[i] != yArr[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                # 如果 L == H，那就没什么优化的了，continue
                if L == H:
                    print("L == H")
                    continue
                # 计算 eta，eta 是 alphas[j] 的最优修改量，如果 eta == 0，需要退出
                # for 循环迭代的过程，实际上是比较边界值，取较小，在此先不处理
                eta = np.sum(dataArr[i, :] * dataArr[i, :]) + \
                      np.sum(dataArr[j, :] * dataArr[j, :]) - \
                      2. * np.sum(dataArr[i, :] * dataArr[j, :])
                if eta <= 0:
                    print("eta <= 0")
                    continue
                # 准备好之后，就可以计算出新的 alphas[j] 值
                alphas[j] = alphaJold + yArr[j] * (Ei - Ej) / eta
                # 此时还需要对 alphas[j] 进行修正
                alphas[j] = clipAlpha(alphas[j], H, L)

                # 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j is not moving enough")
                    continue
                # 下面对 i 进行修正，修改量与 j 相同，但方向相反
                alphas[i] = alphaIold + yArr[i] * yArr[j] * (alphaJold - alphas[j])

                # 下面计算参数 b
                bi = b - Ei - yArr[i] * (alphas[i] - alphaIold) * np.sum(dataArr[i, :] * dataArr[i, :]) - \
                     yArr[j] * (alphas[j] - alphaJold) * np.sum(dataArr[i, :] * dataArr[j, :])
                bj = b - Ej - yArr[i] * (alphas[i] - alphaIold) * np.sum(dataArr[i, :] * dataArr[j, :]) - \
                     yArr[j] * (alphas[j] - alphaJold) * np.sum(dataArr[j, :] * dataArr[j, :])
                # b 的更新条件
                if 0 < alphas[i] < C:
                    b = bi
                elif 0 < alphas[j] < C:
                    b = bj
                else:
                    b = (bi + bj) / 2.
                # 到了这一步，说明 alpha , b 被更新了
                alphaPairsChanged += 1
                # 输出迭代信息
                print("iter: %d, i: %d, pairs changed %d" % (iterations, i, alphaPairsChanged))
        # 在 for 循环之外，检查 alpha 值是否做了更新，如果在更新将 iterations 设为
        # 0 后继续运行程序
        # 知道更新完毕后，iterations 次循环无变化，则退出循环
        if alphaPairsChanged == 0:
            iterations += 1
        else:
            iterations = 0
        print("iteration number: %d" % iterations)

    return b, alphas


def testSVM():
    dataArr, yArr = loadData("testSet.txt")
    C = 0.6
    toler = 0.001
    maxIter = 40
    b, alphas = smoSimple(dataArr, yArr, C, toler, maxIter)
    return b, alphas


if __name__ == "__main__":
    b, alphas = testSVM()
    print(b, alphas)

    showData("testSet.txt", line=(b, alphas))
