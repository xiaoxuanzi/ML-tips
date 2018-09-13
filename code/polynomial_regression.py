# python -version 3.5+
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from sklearn.cross_validation import train_test_split

def test_poly_fit():
    '''
    代码来至：https://carrylaw.github.io/bml/2017/10/13/ml02/
    '''
    # 读取数据集
    datasets_X = [] #建立datasets_X储存房屋尺寸数据
    datasets_Y = [] #建立datasets_Y储存房屋成交价格数据
    fr = open('../data/prices.txt', 'r', encoding='utf-8') #指定prices.txt数据集所在路径
    lines = fr.readlines() #读取一整个文件夹
    for line in lines: #逐行读取，循环遍历所有数据
        items = line.strip().split(",") #变量之间按逗号进行分隔
        datasets_X.append(int(items[0])) #读取的数据转换为int型
        datasets_Y.append(int(items[1]))

    # 数据预处理
    length = len(datasets_X)
    datasets_X = np.array(datasets_X).reshape([length, 1]) #将datasets_X转化为数组
    datasets_Y = np.array(datasets_Y)

    poly_fit(datasets_X, datasets_Y)

def poly_fit(datasets_X, datasets_Y):
    X_train, X_test, y_train, y_test = train_test_split(datasets_X, datasets_Y, test_size=0.4, random_state=0)
    # print(X_train, X_test, y_train, y_test)
    # print(len(X_train), len(X_test), len(y_train), len(y_test))

    # 数据建模
    poly_reg = PolynomialFeatures(degree=2) #degree=2表示二次多项式
    X_poly = poly_reg.fit_transform(X_train) #构造datasets_X二次多项式特征X_poly
    #X 的特征已经从 [x1, x2] 转换成 [1, x1, x2, x1^2, x1x2, x2^2] 并且能够在任意的线性模型中使用

    lin_reg_2 = linear_model.LinearRegression() #创建线性回归模型
    lin_reg_2.fit(X_poly, y_train) #使用线性回归模型学习X_poly和datasets_Y之间的映射关系

    # 查看回归系数
    print('Coefficients:',lin_reg_2.coef_)
    # 查看截距项
    print('intercept:',lin_reg_2.intercept_)
    # score
    #它可以对 Model 用 R^2 的方式进行打分，输出精确度
    print('score: %.3f', lin_reg_2.score(poly_reg.fit_transform(X_test), y_test))

    minX = min(datasets_X) #以数据datasets_X的最大值和最小值为范围，建立等差数列，方便后续画图
    maxX = max(datasets_X)
    X = np.arange(minX, maxX).reshape([-1, 1])

    # 数据可视化
    plt.scatter(datasets_X, datasets_Y, color='orange')
    plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
    plt.xlabel('Area')
    plt.ylabel('Price')
    plt.show()

if __name__ == '__main__':
    test_poly_fit()

