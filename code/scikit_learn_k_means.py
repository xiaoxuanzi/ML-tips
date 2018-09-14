# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np

# 生成2*10的矩阵，且值均匀分布的随机数
cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(3.5, 4.5, (2, 10))

# 顺序连接两个矩阵，形成一个新矩阵,所以生成了一个2*20的矩阵
# T做转置后变成20*2的矩阵,刚好是一堆(x,y)的坐标点
X = np.hstack((cluster1, cluster2)).T

plt.figure()
plt.axis([0, 5, 0, 5])
plt.grid(True)
plt.plot(X[:,0],X[:,1],'k.')

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
# print(kmeans.cluster_centers_)
print("The first center: ",kmeans.cluster_centers_[0,:])
print("The second center: ", kmeans.cluster_centers_[1,:])

plt.plot(kmeans.cluster_centers_[:,0],
         kmeans.cluster_centers_[:,1], 'ro')

plt.show()
