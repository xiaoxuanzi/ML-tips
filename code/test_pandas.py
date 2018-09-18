'''
code from: http://codingpy.com/article/a-quick-intro-to-pandas/
'''
import os
import pandas as pd # This is the standard
import matplotlib.pyplot as plt

# os.chdir(os.getcwd()+'/data')
df = pd.read_csv('../data/uk_rain_2014.csv', header=0) #指定行数用来作为列名，数据开始行数。如果文件中没有列名，则默认为0，否则设置为None。
print(df.head(5))

# Getting last x rows.
#df.tail(5)

# Changing column labels.
df.columns = ['water_year','rain_octsep',
              'outflow_octsep', 'rain_decfeb',
              'outflow_decfeb', 'rain_junaug',
              'outflow_junaug']
print(df.head(5))

# Finding out how many rows dataset has.
print(len(df))

'''
你可能还想知道数据集的一些基本的统计数据，在 Pandas 中，这个操作简单到哭：
Finding out basic statistical information on your dataset.
'''

pd.options.display.float_format = '{:,.3f}'.format # Limit output to 3 decimal places.
print(df.describe())
print(type(df.describe())) #会得到一个 series ，而不是 dataframe 。


'''
有时你想提取一整列，使用列的标签可以非常简单地做到：

# Getting a column by label
'''
print(df['rain_octsep'])

'''
# Getting a column by label using .
这句代码返回的结果与前一个例子完全一样——是我们选择的那列数据。
'''

print("-----------")
print(df.rain_octsep)


'''
如果你读过这个系列关于 Numpy 的推文，你可能还记得一个叫做 布尔过滤（boolean masking）的技术，通过在一个数组上运行条件来得到一个布林数组。在 Pandas 里也可以做到。

# Creating a series of booleans based on a conditional
df.rain_octsep < 1000 # Or df['rain_octsep] < 1000
上面的代码将会返回一个由布尔值构成的 dataframe。True 表示在十月-九月降雨量小于 1000 mm，False 表示大于等于 1000 mm。
'''
print(df.rain_octsep < 1000)# Or df['rain_octsep] < 1000


'''
我们可以用这些条件表达式来过滤现有的 dataframe。

# Using a series of booleans to filter
df[df.rain_octsep < 1000]
'''

print(df[df.rain_octsep < 1000])
print(len(df[df.rain_octsep < 1000]))


'''
也可以通过复合条件表达式来进行过滤：

# Filtering by multiple conditionals
这条代码只会返回 rain_octsep 中小于 1000 的和 outflow_octsep 中小于 4000 的记录：
'''
print(df[(df.rain_octsep < 1000) & (df.outflow_octsep < 4000)]) # Can't use the keyword 'and'

'''
如果你的数据中字符串，好消息，你也可以使用字符串方法来进行过滤：

# Filtering by string methods
注意，你必须用 .str.[string method] ，而不能直接在字符串上调用字符方法。上面的代码返回所有 90 年代的记录
'''
print(df[df.water_year.str.startswith('199')])

'''
如果你的行标签是数字型的，你可以通过 iloc 来引用：

# Getting a row via a numerical index
'''

print(df.iloc[30])


# Setting a new index from an existing column
df = df.set_index(['water_year'])
print(df.head(5))

# Getting a row via a label-based index
print(df.loc['2000/01'])


# Getting a row via a label-based or numerical index
print(df.ix['1999/00']) # Label based with numerical index fallback *Not recommended


# df = df.set_index(['rain_octsep'])
# print(df.sort_index(ascending=False).head(5)) #inplace=True to apple the sorting in place

'''

当你将一列设置为索引的时候，它就不再是数据的一部分了。如果你想将索引恢复为数据，调用 set_index 相反的方法 reset_index 即可：

# Returning an index to data
'''
df = df.reset_index('water_year')
print(df.head(5))

#对数据集应用函数
# Applying a function to a column
def base_year(year):
    base_year = year[:4]
    base_year= pd.to_datetime(base_year).year
    return base_year

df['year'] = df.water_year.apply(base_year)
print(df.head(5))


#Manipulating structure (groupby, unstack, pivot)
# Grouby
print(df.groupby(df.year // 10 *10).max())
'''

你也可以按照多列进行分组：

# Grouping by multiple columns
'''

# decade_rain = df.groupby([df.year // 10 * 10, df.rain_octsep // 1000 * 1000])[['outflow_octsep', 'outflow_decfeb', 'outflow_junaug']].mean()
# print(decade_rain)


print(df.head(5))

# Using pandas to quickly plot graphs
df.plot(x='year', y=['rain_octsep', 'outflow_decfeb'])

plt.show()

# Saving your data to a csv
# df.to_csv('uk_rain.csv')