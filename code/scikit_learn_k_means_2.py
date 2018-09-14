# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# 生成2*10的矩阵，且值均匀分布的随机数
cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(1.5, 2.5, (2, 10))
cluster3 = np.random.uniform(1.5, 3.5, (2, 10))
cluster4 = np.random.uniform(3.5, 4.5, (2, 10))

# 顺序连接两个矩阵，形成一个新矩阵,所以生成了一个2*20的矩阵，T做转置后变成20*2的矩阵,刚好是一堆(x,y)的坐标点
X1 = np.hstack((cluster1, cluster2))
X2 = np.hstack((cluster3, cluster4))
X = np.hstack((X1, X2)).T #40 * 2 矩阵

K = range(1, 10)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    # 求kmeans的成本函数值
    meandistortions.append(sum(np.min(cdist(X,kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

plt.figure()
plt.grid(True)

#解决画图中文乱码问题
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

plt1 = plt.subplot(4,1,1)

# 画样本点
plt1.plot(X[:,0],X[:,1],'k.')
plt1.set_title('样本点')
plt1.legend()

plt2 = plt.subplot(4,1,2)
# 画成本函数值曲线
plt2.plot(K, meandistortions, 'bx-')
plt2.set_title('成本函数曲线')
plt2.legend()

# 将数据分成4簇
k = 4
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

# print(type(kmeans.cluster_centers_))
# print(kmeans.cluster_centers_)
# print(kmeans.labels_ )
# 将4簇用不同的颜色标记
plt3 = plt.subplot(4,1,3)
plt3.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
plt3.set_title("区分不同簇的点")
plt3.legend()

# 画出每一簇的中心点
plt4 = plt.subplot(4,1,4)
plt4.plot(X[:,0],X[:,1],'k.')
plt4.plot(kmeans.cluster_centers_[:,0],
         kmeans.cluster_centers_[:,1], 'ro')

plt4.set_title("各簇中心点")
# plt.legend(loc='best') #matplotlib label doesn't work if you forgot to display the legend
plt4.legend()
plt.show()
