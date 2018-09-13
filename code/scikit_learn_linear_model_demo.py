
def  test_sk_learn_linear_model():
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

def test_plot_sklearn_linear():
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

if __name__ == '__main__':
    test_plot_sklearn_linear()
    test_sk_learn_linear_model()
