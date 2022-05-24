import numpy as np
import pandas as pd

s = pd.Series([1,3,5.5,np.nan, 'a'])
# print(s)
s1 = pd.Series([10,12,8,14], index=['a','b','c','d'])
# print(s1)

dane = {'kraj': ['Belgia','Indie','Brazylia'],
        'stolica': ['Bruksela','aaaa','Brasilia'],
        'populacja': [234324234,234234,234234234]}

df = pd.DataFrame(dane)
# print(df)

# daty = pd.date_range('2022', periods = 5)
# df = pd.DataFrame(np.random.randn(5,4), index=daty, columns=list('ABCD'))
# print(df)

iris_df = pd.read_csv('iris.csv', header = 0, sep = ',', decimal = '.')
# print(iris_df)
iris_df.to_csv('nowy.csv', index=False)



# xlsx = pd.ExcelFile('wyniki.xlsx')
# df = pd.read_excel(xlsx, header= 0)
# print(df)
#
# print(s1['a'])
# print(s1.a)
#
# print(df['populacja'])
# print(df.populacja)
#
# print(df.iloc[[0],[1]])
# print(df.loc[[0],['kraj']])
# print(df.at[0,'kraj'])
#
# print('kraj: ' + df.kraj)
#
# print(df.sample(1))
# print(df.sample(frac=0.5))
#
# print('')


# print(s1[s1>10])
# print(s1.where(s1 > 10, 'element nie spelnia warunku'))
#
# seria = s1.copy()
# print(seria)
# seria.where(s1 >10,'element valid', inplace=True)
# print(seria)


# print(s1[~(s1 > 10)])
# print(s1[(s1 > 13) & (s1 > 8)])
#
# print(df[df['populacja'] > 120000000])
# print(df[(df.populacja > 120000000) & (df.index.isin([0,2]))])
#
# szukaj = ['Belgia','Brasilia']
# print(df.isin(szukaj))
#
# s1['e'] = 15
# print(s1)

df.loc[3] = 'nowy_element'
df.loc[4] = ['Polska', 'Warszawa', 234324432]

# print(df)

df.drop(3, inplace=True)
print(df)
# df.drop('kraj', axis=1, inplace=True)
# print(df)
df['kontynent']=['Europa','Azja','Ameryka Poludniowa', 'Europa']
print(df)
print(df.sort_values(by='kraj'))
grupa = df.groupby(by='kontynent')
print(grupa.get_group('Europa'))

print(df.groupby(by='kontynent').agg({'populacja' : ['sum']}))
