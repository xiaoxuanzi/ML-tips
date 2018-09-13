
def test_np_least_square():
    from numpy.linalg import lstsq
    #使用numpy的最小二乘函数直接计算出β
    X = [[1,1,1],[1,1,2],[1,2,1]]
    y = [[6],[9],[8]]

    print(lstsq(X, y)[0])

def test_sklearn_linear_regression():

    from sklearn.linear_model import LinearRegression

    X = [[1,1,1],[1,1,2],[1,2,1]]
    y = [[6],[9],[8]]

    model = LinearRegression()
    model.fit(X, y)
    x2 = [[1,3,5]]
    y2 = model.predict(x2)
    print(y2)

if __name__ == '__main__':
    test_np_least_square()
    test_sklearn_linear_regression()
