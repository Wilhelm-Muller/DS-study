import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 1. Basic operations on pandas
#s = pd.Series([1,3,6, np.nan, 44, 1])
#print(s)

#dates = pd.date_range('2025-04-30', periods=5)
#print(dates)

#df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=['a','b','c','d'])
#df1 = pd.DataFrame(np.arange(12).reshape((3,4)))
#print(df1)

#df2 = pd.DataFrame({'A': 1,
#                    'B': pd.Timestamp('2025-04-30'),
#                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
#                    'D': np.array([3] * 4, dtype='int32'),
#                    'E': pd.Categorical(['test', 'train', 'test', 'train']),
#                    'F': 'foo'})
#print(df2)
#print(df2.dtypes) #data types of the columns
#print(df2.columns) #columns of the dataframe
#print(df2.values) #index of the dataframe
#print(df2.describe()) #summary of the dataframe
#print(df2.T) #transpose of the dataframe
#print(df2.sort_index(axis=1, ascending=False)) #sort by descending order, by rows
#print(df2.sort_index(axis=0, ascending=False))


# 2. Select data from pandas dataframe
#dates = pd.date_range('2025-04-30', periods=6)
#df = pd.DataFrame(np.arange(24).reshape((6,4)), index=dates, columns=['A','B','C','D'])
#print(df)
#print(df['A'], df.A)
#print(df[0:3], df['2025-04-30':'2025-05-02'])

#select by label
#print(df.loc['2025-04-30',['A','B']]) #selecting rows by index

#select by position: iloc
#print(df.iloc[[1,3],1:3])

#mixed selection: ix
#print(df.ix[:3, ['A','C']])

#Boolean indexing
#print(df[df.A > 3]) #selecting rows where A > 3


# 3. Setting values by label
#df.iloc[2,2] = 1111
#df.loc['2025-05-01','C'] = 2222
#df[df.A > 4] = 0
#df['F'] = np.nan
#print(df)

# 4. Dealing with np.nan values
#df.iloc[0,1] = np.nan
#df.iloc[1,2] = np.nan
#print(df)
# here we see [0,1] and [1,2] are NaN values
#print(df.dropna(axis=0, how='any')) #drop rows with any NaN values
#print(df.dropna(axis=1, how='any')) #drop columns with any NaN values
#print(df.fillna(value=0)) #fill NaN values with 0
#print(df.isnull()) #check for NaN values, returrn a dataframe of boolean values of true or false
#print(np.any(df.isnull()) == True) # return boolean on whether every value is not nan


# 5. pandas data I/O
# importing data from csv file
#data = pd.read_csv('data.csv') 
#data.to_pickle('data.pickle') #save the dataframe to a pickle file
# 

# 6. Concat in Pandas
#df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
#df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
#df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])
#Concat
#res = pd.concat([df1,df2,df3],axis=0, ignore_index=True)
#print(res)

#join, [inner, outer]
#df = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
#df1 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])
#res = pd.concat([df,df1], axis=0, join='outer', ignore_index=True) #inner join
# inner join keeps common columns, outer join keeps all columns
#print(res)

#Concat with join axes
#df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
#df2 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])
#res = pd.concat([df1,df2], axis=1, join_axes=[df1.index])
#print(res)

# 7. Merge in Pandas
#left = pd.DataFrame({'key':['K0','K1','K2','K3'],
#                     'A':['A0','A1','A2','A3'],
#                     'B':['B0','B1','B2','B3']})
#right = pd.DataFrame({'key':['K0','K1','K2','K3'],
#                      'C':['C0','C1','C2','C3'],
#                      'D':['D0','D1','D2','D3']})
#res = pd.merge(left, right, on='key', how='inner')
#print(res)
#how = ['left', 'right', 'outer', 'inner']

#indicators
#df1 = pd.DataFrame({'col1':[0,1], 'col_left':['a','b']})
#df2 = pd.DataFrame({'col1':[1,2,2], 'col_right':[2,2,2]})
#res = pd.merge(df1,df2, on='col1', how='outer', indicator=True)
#Indicators are filling up the NaN values with the name of the dataframe
#print(res)


# 8. Plotting in pandas
#data = pd.DataFrame(np.random.randn(1000), index=np.arange(1000))
#data = data.cumsum()

data=pd.DataFrame(np.random.randn(1000,4), index=np.arange(1000),columns=list("ABCD"))
data = data.cumsum()

ax = data.plot.scatter(x='A', y='B', color='DarkBlue', label='Class 1')
#Plot methods
plt.show()
