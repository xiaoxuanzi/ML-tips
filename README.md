# machine-learning-notes

## Linear Regression
#### Simple Linear Regression[一元线性回归](https://www.geeksforgeeks.org/linear-regression-python-implementation/)
* [code](code/simple_linear_regression.py)<br>
  <img src="images/simple_linear_regression.png" width = "300" height = "300" alt="title" align=center />
* least square regression[最小二乘法](https://zh.wikipedia.org/wiki/%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%98%E6%B3%95) [一元线性回归的细节](https://www.jianshu.com/p/fcd220697182)

#### Multiple linear regression
* [code](code/multiple_linear_regression.py)

#### Polynomial regression(多项式回归)
[扩展具有基函数的线性模型, 跟线性模型是同一类](https://www.jianshu.com/p/cf2b391a3c95)
* [code](code/polynomial_regression.py)<br>
  <img src="images/polynomial_regression.png" width = "300" height = "300" alt="title" align=center />
<pre><code>
[@CentOS ~]# python code/polynomial_regression.py
Coefficients: [  0.00000000e+00   3.26162199e-02   2.11341000e-05]
intercept: 177.607131202
score: %.3f 0.77180187392
</pre></code>

## 机器学习教程 Scikit-learn 
以下代码代码来至blog [SharEDITor](www.shareditor.com) 网址: www.shareditor.com

#### 一元线性回归
[code](code/scikit_learn_linear_model_demo.py)
<pre><code>
	import numpy as np
	from sklearn.linear_model import LinearRegression

	x = [[1],[2],[3],[4],[5],[6]]
	y = [[1],[2.1],[2.9],[4.2],[5.1],[5.8]]
	model = LinearRegression()
	model.fit(x, y)

	print("系数: ", model.coef_)
	print("截距: ", model.intercept_)
	print("回归函数：y = x * ", str(model.coef_[0][0]), " + ", str(model.intercept_[0]))
	predicted = model.predict([13])[0]
	print("x = 13 的预测值： ",predicted[0])

</pre></code>

* 画图
<pre><code>
	import numpy as np
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    x = [[1],[2],[3],[4],[5],[6]]
    y = [[1],[2.1],[2.9],[4.2],[5.1],[5.8]]
    model = LinearRegression()
    model.fit(x, y)
    x2 = [[0], [2.5], [5.3], [9.1]]
    y2 = model.predict(x2)

    plt.figure()
    plt.title('linear sample')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis([0, 10, 0, 10])
    plt.grid(True)
    plt.plot(x, y, 'k.')
    plt.plot(x2, y2, 'g-')
    plt.show()
</pre></code>
<img src="images/scikit_learn_linear_model_demo.png" width = "300" height = "300" alt="title" align=center />

#### 多元线性回归模型 [code](code/scikit_learn_multvariable_linear_model_demo.py)
* 用numpy的最小二乘函数计算
<pre><code>
	from numpy.linalg import lstsq
    #使用numpy的最小二乘函数直接计算出β
    X = [[1,1,1],[1,1,2],[1,2,1]]
    y = [[6],[9],[8]]

    print(lstsq(X, y)[0])
	
</pre></code>
* 用scikit-learn求解多元线性回归
<pre><code>
    from sklearn.linear_model import LinearRegression

    X = [[1,1,1],[1,1,2],[1,2,1]]
    y = [[6],[9],[8]]

    model = LinearRegression()
    model.fit(X, y)
    x2 = [[1,3,5]]
    y2 = model.predict(x2)
    print(y2)
</pre></code>
## 梯度下降
* [批量梯度下降法 Batch Gradient Descent, BGD](http://kissg.me/2017/07/23/gradient-descent/)

