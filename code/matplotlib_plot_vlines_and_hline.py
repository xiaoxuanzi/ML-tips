# coding:utf-8

'''
# 代码来至
链接：https://www.zhihu.com/question/21929761/answer/164975814
'''

import matplotlib.pyplot as plt
import numpy as np

def Laplacian_Prior(x, u, b):

    return 1.0/(2*b)*np.exp(-abs(x - u)/b)

if __name__ == "__main__":

    x = np.arange(-10, 10, 0.01)
    y_1 = Laplacian_Prior(x, 0, 1)
    y_2 = Laplacian_Prior(x, 0, 2)
    y_3 = Laplacian_Prior(x, 0, 4)
    y_4 = Laplacian_Prior(x, -5, 4)
    plt.plot(x, y_1, "r-")
    plt.plot(x, y_2, "k-")
    plt.plot(x, y_3, "b-")
    plt.plot(x, y_4, "g-")
    # plt.vlines(0, 0, 0.5, colors = "c", linestyles = "dashed")
    plt.hlines(0.3, -10, 10, colors = "c", linestyles = "dashed")
    plt.show()
