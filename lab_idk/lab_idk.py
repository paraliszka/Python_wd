import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sns.set(rc={'figure.figsize':(8,8)})
# sns.lineplot(x=[1,2,3,4], y=[1,4,9,16], label="linia nr 1", color="red", marker='o', linestyle=":")
# sns.lineplot(x=[1,2,3,4], y=[2,5,10,17], label="linia nr 2", color="green", marker='^', linestyle=":")
# plt.xlabel('oś x')
# plt.xlabel('oś y')
# plt.title("wykres liniowy")
# plt.show()

# s = pd.Series(np.random.randn(1000))
# s = s.cumsum()
# sns.set()
# wykres = sns.relplot(kind='line', data=s, label='linia')
# wykres.fig.set_size_inches(8,6)
# wykres.fig.suptitle('Wykres liniowy losowych ciągów')
# wykres.set_xlabels('indexy')
# wykres.set_ylabels('wartosci')
# wykres.add_legend()
# wykres.figure.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9)
# plt.show()

# sns.set()
# df = pd.read_csv('iris.csv',header=None, sep=',',decimal='.')
# print(df)
# wykres = sns.lineplot(data=df,x=df.index,y=df[0], hue=df[4])
# wykres.set_xlabel('indeksy')
# wykres.set_title('wykres liniowy danych iris dataset')
# wykres.legend(title='rodzaj kwiatu')
# plt.show()
#
# sns.set()
# data={'a':np.arange(10),
#       'c':np.random.randint(0,50,10),
#       'd':np.random.randn(10)}
# data['b']=data['a'] + 10 * np.random.randn(10)
# data['d']=np.abs(data['d']*10)
# print(data['c'])
# print(data['d'])
# df=pd.DataFrame(data)
# plot = sns.relplot(data=df, x='a',y='b', hue='c', palette='bright',size='d',legend=True)
# plot.fig.set_size_inches(6,6)
# plot.set(xticks=data['a'])
# plt.show()

# data={'Kraj':['Belgia','indie' , 'brazylia'],
#       'Stolica':['Bruksela','New Delhi', 'Brasilia'],
#       'Kontynent': ['Europa', 'Azja', 'Europa'],
#       'Populacja':[14234235,325325235,325235235]}
# df=pd.DataFrame(data)
# sns.set()
# plot = sns.catplot(data=df,x='Kontynent',y='Populacja',
#                    kind='bar',ci=None,hue='Kontynent',estimator=np.sum,
#                    dodge=False,palette=['red','green','yellow'],
#                    legend_out=False)
# plot.fig.set_size_inches(7,6)
# plot.add_legend(title='Populacja na kontynentach', loc='upper right')
# plot.fig.suptitle('Populacja na kontynentach')
# plt.show()
#
# plot = sns.barplot(data=df,x='Kontynent',y='Populacja',
#                    ci=None,hue='Kontynent',estimator=np.sum,
#                    dodge=False,palette=['red','green','yellow'])
# plot.legend(title='Populacja na kontynentach')
# plot.set(title="Wykres słupkowy")
# plt.show()


# fig= plt.figure()
# ax=fig.add_subplot(111,projection='3d')
# print(type(ax))
# t = np.linspace(0,2 * np.pi, 100)
# z = t
# x = np.sin(t)*np.cos(t)
# y = np.tan(t)
# ax.plot(x,y,z, label = 'zadanie 1')
# ax.legend()
# plt.show()

# np.random.seed(19680801)
# def randrange(n, vmin, vmax):
#       return(vmax-vmin)*np.random.rand(n) + vmin
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# n = 180
#
# for c,m,zlow,zhigh in[('r','o',-50,-25),('b','^',-30, -5)]:
#       xs=randrange(n,23,32)
#       ys=randrange(n,0,100)
#       zs=randrange(n,zlow,zhigh)
#       ax.scatter(xs,ys,zs,c=c,marker=m)
#
# # xs.set_xlabel('X Label')
# # ys.set_ylabel('Y Label')
# # zs.set_zlabel('Z Label')
#
# plt.show()

from mpl_toolkits.mplot3d.art3d import get_test_data

# fig = plt.figure(figsize=plt.figaspect(0.5))
#
# ax = fig.add_subplot(1, 2, 1, projection = '3d')
# np.random.seed(19680801)
# def randrange(n, vmin, vmax):
#       return(vmax - vmin)+np.random.rand(n) + vmin
#idk








NumPy
NumPy jest biblioteką Pythona służącą do obliczeń naukowych.

Zastosowania:

algebra liniowa
zaawansowane obliczenia matematyczne (numeryczne)
całkowania
rozwiązywanie równań
…
Import biblioteki NumPy
import numpy as np
Podstawowym bytem w bibliotece NumPy jest N-wymiarowa tablica zwana ndarray. Każdy element na tablicy traktowany jest jako typ dtype.

numpy.array(object, dtype=None, copy=True,
            order='K', subok=False, ndmin=0)
object - to co ma być wrzucone do tablicy
dtype - typ
copy - czy obiekty mają być skopiowane, domyślne True
order - sposób układania: C (rzędy), F (kolumny), A, K
subok - realizowane przez podklasy (jeśli True), domyślnie False
ndmin - minimalny rozmiar (wymiar) tablicy
import numpy as np

a = np.array([1, 2, 3])
print(a)
b = np.array([1, 2, 3.0])
print(b)
c = np.array([[1, 2], [3, 4]])
print(c)
d = np.array([1, 2, 3], ndmin=2)
print(d)
e = np.array([1, 2, 3], dtype=complex)
print(e)
f = np.array(np.mat('1 2; 3 4'))
print(f)
g = np.array(np.mat('1 2; 3 4'), subok=True)
print(g)
## [1 2 3]
## [1. 2. 3.]
## [[1 2]
##  [3 4]]
## [[1 2 3]]
## [1.+0.j 2.+0.j 3.+0.j]
## [[1 2]
##  [3 4]]
## [[1 2]
##  [3 4]]
Lista a tablica
import numpy as np
import time

start_time = time.time()
my_arr = np.arange(1000000)
my_list = list(range(1000000))
start_time = time.time()
my_arr2 = my_arr * 2
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
my_list2 = [x * 2 for x in my_list]
print("--- %s seconds ---" % (time.time() - start_time))
## --- 0.0158383846282959 seconds ---
## --- 0.08654999732971191 seconds ---
Atrybuty tablic ndarray
Atrybut	Opis
shape	krotka z informacją liczbę elementów dla każdego z wymiarów
size	liczba elementów w tablicy (łączna)
ndim	liczba wymiarów tablicy
nbytes	liczba bajtów jaką tablica zajmuje w pamięci
dtype	typ danych
https://numpy.org/doc/stable/reference/arrays.ndarray.html#array-attributes

import numpy as np

tab1 = np.array([2, -3, 4, -8, 1])
print(type(tab1))
print(tab1.shape)
print(tab1.size)
print(tab1.ndim)
print(tab1.nbytes)
print(tab1.dtype)
## <class 'numpy.ndarray'>
## (5,)
## 5
## 1
## 20
## int32
import numpy as np

tab2 = np.array([[2, -3], [4, -8]])
print(type(tab2))
print(tab2.shape)
print(tab2.size)
print(tab2.ndim)
print(tab2.nbytes)
print(tab2.dtype)
tab3 = np.array([[2, -3], [4, -8, 5], [3]])
print(type(tab3))
print(tab3.shape)
print(tab3.size)
print(tab3.ndim)
print(tab3.nbytes)
print(tab3.dtype)
## <class 'numpy.ndarray'>
## (2, 2)
## 4
## 2
## 16
## int32
## <string>:1: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
## <class 'numpy.ndarray'>
## (3,)
## 3
## 1
## 24
## object
Typy danych
https://numpy.org/doc/stable/reference/arrays.scalars.html

https://numpy.org/doc/stable/reference/arrays.dtypes.html#arrays-dtypes-constructing

Typy całkowitoliczbowe	int,int8,int16,int32,int64
Typy całkowitoliczbowe (bez znaku)	uint,uint8,uint16,uint32,uint64
Typ logiczny	bool
Typy zmiennoprzecinkowe	float, float16, float32, float64, float128
Typy zmiennoprzecinkowe zespolone	complex, complex64, complex128, complex256
Napis	str
import numpy as np

tab = np.array([[2, -3], [4, -8]])
print(tab)
tab2 = np.array([[2, -3], [4, -8]], dtype=int)
print(tab2)
tab3 = np.array([[2, -3], [4, -8]], dtype=float)
print(tab3)
tab4 = np.array([[2, -3], [4, -8]], dtype=complex)
print(tab4)
## [[ 2 -3]
##  [ 4 -8]]
## [[ 2 -3]
##  [ 4 -8]]
## [[ 2. -3.]
##  [ 4. -8.]]
## [[ 2.+0.j -3.+0.j]
##  [ 4.+0.j -8.+0.j]]
Tworzenie tablic
np.array - argumenty rzutowany na tablicę (coś po czym można iterować) - warto sprawdzić rozmiar/kształt

import numpy as np

tab = np.array([2, -3, 4])
print(tab)
tab2 = np.array((4, -3, 3, 2))
print(tab2)
tab3 = np.array({3, 3, 2, 5, 2})
print(tab3)
tab4 = np.array({"pl": 344, "en": 22})
print(tab4)
## [ 2 -3  4]
## [ 4 -3  3  2]
## {2, 3, 5}
## {'pl': 344, 'en': 22}
np.zeros - tworzy tablicę wypełnioną zerami

import numpy as np

tab = np.zeros(4)
print(tab)
print(tab.shape)
tab2 = np.zeros([2, 3])
print(tab2)
print(tab2.shape)
tab3 = np.zeros([2, 3, 4])
print(tab3)
print(tab3.shape)
## [0. 0. 0. 0.]
## (4,)
## [[0. 0. 0.]
##  [0. 0. 0.]]
## (2, 3)
## [[[0. 0. 0. 0.]
##   [0. 0. 0. 0.]
##   [0. 0. 0. 0.]]
##
##  [[0. 0. 0. 0.]
##   [0. 0. 0. 0.]
##   [0. 0. 0. 0.]]]
## (2, 3, 4)
np.ones - tworzy tablicę wypełnioną jedynkami (to nie odpowiednik macierzy jednostkowej!)

import numpy as np

tab = np.ones(4)
print(tab)
print(tab.shape)
tab2 = np.ones([2, 3])
print(tab2)
print(tab2.shape)
tab3 = np.ones([2, 3, 4])
print(tab3)
print(tab3.shape)
## [1. 1. 1. 1.]
## (4,)
## [[1. 1. 1.]
##  [1. 1. 1.]]
## (2, 3)
## [[[1. 1. 1. 1.]
##   [1. 1. 1. 1.]
##   [1. 1. 1. 1.]]
##
##  [[1. 1. 1. 1.]
##   [1. 1. 1. 1.]
##   [1. 1. 1. 1.]]]
## (2, 3, 4)
np.diag - tworzy tablicę odpowiadającą macierzy diagonalnej

import numpy as np

tab0 = np.diag([3, 4, 5])
print(tab0)
tab1 = np.array([[2, 3, 4], [3, -4, 5], [3, 4, -5]])
print(tab1)
tab2 = np.diag(tab1)
print(tab2)
tab3 = np.diag(tab1, k=1)
print(tab3)
tab4 = np.diag(tab1, k=-2)
print(tab4)
tab5 = np.diag(np.diag(tab1))
print(tab5)
## [[3 0 0]
##  [0 4 0]
##  [0 0 5]]
## [[ 2  3  4]
##  [ 3 -4  5]
##  [ 3  4 -5]]
## [ 2 -4 -5]
## [3 5]
## [3]
## [[ 2  0  0]
##  [ 0 -4  0]
##  [ 0  0 -5]]
np.arange - tablica wypełniona równomiernymi wartościami

Składnia: numpy.arange([start, ]stop, [step, ]dtype=None)

import numpy as np

a = np.arange(3)
print(a)
b = np.arange(3.0)
print(b)
c = np.arange(3, 7)
print(c)
d = np.arange(3, 11, 2)
print(d)
e = np.arange(0, 1, 0.1)
print(e)
f = np.arange(3, 11, 2, dtype=float)
print(f)
g = np.arange(3, 10, 2)
print(g)
## [0 1 2]
## [0. 1. 2.]
## [3 4 5 6]
## [3 5 7 9]
## [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]
## [3. 5. 7. 9.]
## [3 5 7 9]
np.linspace - tablica wypełniona równomiernymi wartościami wg skali liniowej

import numpy as np

a = np.linspace(2.0, 3.0, num=5)
print(a)
b = np.linspace(2.0, 3.0, num=5, endpoint=False)
print(b)
c = np.linspace(2.0, 3.0, num=5, retstep=True)
print(c)
## [2.   2.25 2.5  2.75 3.  ]
## [2.  2.2 2.4 2.6 2.8]
## (array([2.  , 2.25, 2.5 , 2.75, 3.  ]), 0.25)
np.logspace - tablica wypełniona wartościami wg skali logarytmicznej

Składnia: numpy.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0)

import numpy as np

a = np.logspace(2.0, 3.0, num=4)
print(a)
b = np.logspace(2.0, 3.0, num=4, endpoint=False)
print(b)
c = np.logspace(2.0, 3.0, num=4, base=2.0)
print(c)
## [ 100.          215.443469    464.15888336 1000.        ]
## [100.         177.827941   316.22776602 562.34132519]
## [4.         5.0396842  6.34960421 8.        ]
np.empty - pusta (niezaincjowana) tablica

import numpy as np

a = np.empty(3)
print(a)
b = np.empty(3, dtype=int)
print(b)
## [0. 1. 2.]
## [0 1 2]
np.identity - tablica przypominająca macierz jednostkową

np.eye - tablica z jedynkami na przekątnej (pozostałe zera)

import numpy as np

a = np.identity(4)
print(a)
b = np.eye(4, k=1)
print(b)
c = np.eye(4, k=2)
print(c)
d = np.eye(4, k=-1)
print(d)
## [[1. 0. 0. 0.]
##  [0. 1. 0. 0.]
##  [0. 0. 1. 0.]
##  [0. 0. 0. 1.]]
## [[0. 1. 0. 0.]
##  [0. 0. 1. 0.]
##  [0. 0. 0. 1.]
##  [0. 0. 0. 0.]]
## [[0. 0. 1. 0.]
##  [0. 0. 0. 1.]
##  [0. 0. 0. 0.]
##  [0. 0. 0. 0.]]
## [[0. 0. 0. 0.]
##  [1. 0. 0. 0.]
##  [0. 1. 0. 0.]
##  [0. 0. 1. 0.]]
Indeksowanie, “krojenie”
import numpy as np

a = np.array([2, 5, -2, 4, -7, 8, 9, 11, -23, -4, -7, 8, 1])
print(a[5])
print(a[-2])
print(a[3:6])
print(a[:])
print(a[0:-1])
print(a[:5])
print(a[4:])
print(a[4:-1])
print(a[4:10:2])
print(a[::-1])
print(a[::2])
print(a[::-2])
## 8
## 8
## [ 4 -7  8]
## [  2   5  -2   4  -7   8   9  11 -23  -4  -7   8   1]
## [  2   5  -2   4  -7   8   9  11 -23  -4  -7   8]
## [ 2  5 -2  4 -7]
## [ -7   8   9  11 -23  -4  -7   8   1]
## [ -7   8   9  11 -23  -4  -7   8]
## [ -7   9 -23]
## [  1   8  -7  -4 -23  11   9   8  -7   4  -2   5   2]
## [  2  -2  -7   9 -23  -7   1]
## [  1  -7 -23   9  -7  -2   2]
import numpy as np

a = np.array([[3, 4, 5], [-3, 4, 8], [3, 2, 9]])
b = a[:2, 1:]
print(b)
print(np.shape(b))
c = a[1]
print(c)
print(np.shape(c))
d = a[1, :]
print(d)
print(np.shape(d))
e = a[1:2, :]
print(e)
print(np.shape(e))
f = a[:, :2]
print(f)
print(np.shape(f))
g = a[1, :2]
print(g)
print(np.shape(g))
h = a[1:2, :2]
print(h)
print(np.shape(h))
## [[4 5]
##  [4 8]]
## (2, 2)
## [-3  4  8]
## (3,)
## [-3  4  8]
## (3,)
## [[-3  4  8]]
## (1, 3)
## [[ 3  4]
##  [-3  4]
##  [ 3  2]]
## (3, 2)
## [-3  4]
## (2,)
## [[-3  4]]
## (1, 2)
**Uwaga - takie “krojenie” to tzw “widok”.

import numpy as np

a = np.array([[3, 4, 5], [-3, 4, 8], [3, 2, 9]])
b = a[1:2, 1:]
print(b)
a[1][1] = 9
print(a)
print(b)
b[0][0] = -11
print(a)
print(b)
## [[4 8]]
## [[ 3  4  5]
##  [-3  9  8]
##  [ 3  2  9]]
## [[9 8]]
## [[  3   4   5]
##  [ -3 -11   8]
##  [  3   2   9]]
## [[-11   8]]
Naprawa:

import numpy as np

a = np.array([[3, 4, 5], [-3, 4, 8], [3, 2, 9]])
b = a[1:2, 1:].copy()
print(b)
a[1][1] = 9
print(a)
print(b)
b[0][0] = -11
print(a)
print(b)
## [[4 8]]
## [[ 3  4  5]
##  [-3  9  8]
##  [ 3  2  9]]
## [[4 8]]
## [[ 3  4  5]
##  [-3  9  8]
##  [ 3  2  9]]
## [[-11   8]]
Indeksowanie logiczne (fancy indexing)

import numpy as np

a = np.array([2, 5, -2, 4, -7, 8, 9, 11, -23, -4, -7, 8, 1])
b = a[np.array([1, 3, 7])]
print(b)
c = a[[1, 3, 7]]
print(c)
## [ 5  4 11]
## [ 5  4 11]
import numpy as np

a = np.array([2, 5, -2, 4, -7, 8, 9, 11, -23, -4, -7, 8, 1])
b = a > 0
print(b)
c = a[a > 0]
print(c)
## [ True  True False  True False  True  True  True False False False  True
##   True]
## [ 2  5  4  8  9 11  8  1]
import numpy as np

a = np.array([2, 5, -2, 4, -7, 8, 9, 11, -23, -4, -7, 8, 1])
b = a[a > 0]
print(b)
b[0] = -5
print(a)
print(b)
a[1] = 20
print(a)
print(b)
## [ 2  5  4  8  9 11  8  1]
## [  2   5  -2   4  -7   8   9  11 -23  -4  -7   8   1]
## [-5  5  4  8  9 11  8  1]
## [  2  20  -2   4  -7   8   9  11 -23  -4  -7   8   1]
## [-5  5  4  8  9 11  8  1]
Modyfikacja kształtu i rozmiaru
https://numpy.org/doc/stable/reference/routines.array-manipulation.html

import numpy as np

a = np.array([[3, 4, 5], [-3, 4, 8], [3, 2, 9]])
b = np.reshape(a, (1, 9))
print(b)
c = a.reshape(9)
print(c)
d = a.flatten()
print(d)
e = a.ravel()
print(e)
f = np.ravel(a)
print(f)
g = [[1, 3, 4]]
h = np.squeeze(g)
print(h)
i = a.T
print(i)
j = np.transpose(a)
print(j)
k = np.hstack((h, h, h))
print(k)
l = np.vstack((h, h, h))
print(l)
m = np.dstack((h, h, h))
print(m)
## [[ 3  4  5 -3  4  8  3  2  9]]
## [ 3  4  5 -3  4  8  3  2  9]
## [ 3  4  5 -3  4  8  3  2  9]
## [ 3  4  5 -3  4  8  3  2  9]
## [ 3  4  5 -3  4  8  3  2  9]
## [1 3 4]
## [[ 3 -3  3]
##  [ 4  4  2]
##  [ 5  8  9]]
## [[ 3 -3  3]
##  [ 4  4  2]
##  [ 5  8  9]]
## [1 3 4 1 3 4 1 3 4]
## [[1 3 4]
##  [1 3 4]
##  [1 3 4]]
## [[[1 1 1]
##   [3 3 3]
##   [4 4 4]]]
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
r1 = np.concatenate((a, b))
print(r1)
r2 = np.concatenate((a, b), axis=0)
print(r2)
r3 = np.concatenate((a, b.T), axis=1)
print(r3)
r4 = np.concatenate((a, b), axis=None)
print(r4)
## [[1 2]
##  [3 4]
##  [5 6]]
## [[1 2]
##  [3 4]
##  [5 6]]
## [[1 2 5]
##  [3 4 6]]
## [1 2 3 4 5 6]
import numpy as np

a = np.array([[1, 2], [3, 4]])
r1 = np.resize(a, (2, 3))
print(r1)
r2 = np.resize(a, (1, 4))
print(r2)
r3 = np.resize(a, (2, 4))
print(r3)
## [[1 2 3]
##  [4 1 2]]
## [[1 2 3 4]]
## [[1 2 3 4]
##  [1 2 3 4]]
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
r1 = np.append(a, b)
print(r1)
r2 = np.append(a, b, axis=0)
print(r2)
## [1 2 3 4 5 6]
## [[1 2]
##  [3 4]
##  [5 6]]
import numpy as np

a = np.array([[1, 2], [3, 7]])
r1 = np.insert(a, 1, 4)
print(r1)
r2 = np.insert(a, 2, 4)
print(r2)
r3 = np.insert(a, 1, 4, axis=0)
print(r3)
r4 = np.insert(a, 1, 4, axis=1)
print(r4)
## [1 4 2 3 7]
## [1 2 4 3 7]
## [[1 2]
##  [4 4]
##  [3 7]]
## [[1 4 2]
##  [3 4 7]]
import numpy as np

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
r1 = np.delete(a, 1, axis=1)
print(r1)
r2 = np.delete(a, 2, axis=0)
print(r2)
## [[ 1  3  4]
##  [ 5  7  8]
##  [ 9 11 12]]
## [[1 2 3 4]
##  [5 6 7 8]]
Broadcasting
Wariant 1 - skalar-tablica - wykonanie operacji na każdym elemencie tablicy

import numpy as np

a = np.array([[1, 2], [5, 6], [9, 10]])
b = a + 4
print(b)
c = 2 ** a
print(c)
## [[ 5  6]
##  [ 9 10]
##  [13 14]]
## [[   2    4]
##  [  32   64]
##  [ 512 1024]]
Wariant 2 - dwie tablice - “gdy jedna z tablic może być rozszerzona” (oba wymiary są równe lub jeden z nich jest równy 1)

https://numpy.org/doc/stable/user/basics.broadcasting.html

import numpy as np

a = np.array([[1, 2], [5, 6]])
b = np.array([9, 2])
r1 = a + b
print(r1)
r2 = a / b
print(r2)
c = np.array([[4], [-2]])
r3 = a + c
print(r3)
r4 = c / a
print(r4)
## [[10  4]
##  [14  8]]
## [[0.11111111 1.        ]
##  [0.55555556 3.        ]]
## [[5 6]
##  [3 4]]
## [[ 4.          2.        ]
##  [-0.4        -0.33333333]]
Funkcje uniwersalne
https://numpy.org/doc/stable/reference/ufuncs.html#methods

Statystyka i agregacja
Funkcja	Opis
np.mean	Średnia wszystkich wartości w tablicy.
np.std	Odchylenie standardowe.
np.var	Wariancja.
np.sum	Suma wszystkich elementów.
np.prod	Iloczyn wszystkich elementów.
np.cumsum	Skumulowana suma wszystkich elementów.
np.cumprod	Skumulowany iloczyn wszystkich elementów.
np.min,np.max	Minimalna/maksymalna wartość w tablicy.
np.argmin, np.argmax	Indeks minimalnej/maksymalnej wartości w tablicy.
np.all	Sprawdza czy wszystki elementy są różne od zera.
np.any	Sprawdza czy co najmniej jeden z elementów jest różny od zera.













Pandas
Pandas jest biblioteką Pythona służącą do analizy i manipulowania danymi

Import:

import pandas as pd
Podstawowe byty
Seria - Series



Ramka danych - DataFrame



import pandas as pd
import numpy as np

s = pd.Series([3, -5, 7, 4])
print(s)
print(s.values)
print(type(s.values))
t = np.sort(s.values)
print(t)
print(s.index)
print(type(s.index))
## 0    3
## 1   -5
## 2    7
## 3    4
## dtype: int64
## [ 3 -5  7  4]
## <class 'numpy.ndarray'>
## [-5  3  4  7]
## RangeIndex(start=0, stop=4, step=1)
## <class 'pandas.core.indexes.range.RangeIndex'>
import pandas as pd
import numpy as np

s = pd.Series([3, -5, 7, 4], index=['a', 'b', 'c', 'd'])
print(s)
print(s['b'])
s['b'] = 8
print(s)
print(s[s > 5])
print(s * 2)
print(np.sin(s))
## a    3
## b   -5
## c    7
## d    4
## dtype: int64
## -5
## a    3
## b    8
## c    7
## d    4
## dtype: int64
## b    8
## c    7
## dtype: int64
## a     6
## b    16
## c    14
## d     8
## dtype: int64
## a    0.141120
## b    0.989358
## c    0.656987
## d   -0.756802
## dtype: float64
import pandas as pd

d = {'key1': 350, 'key2': 700, 'key3': 70}
s = pd.Series(d)
print(s)
## key1    350
## key2    700
## key3     70
## dtype: int64
import pandas as pd

d = {'key1': 350, 'key2': 700, 'key3': 70}
k = ['key0', 'key2', 'key3', 'key1']
s = pd.Series(d, index=k)
print(s)
pd.isnull(s)
pd.notnull(s)
s.isnull()
s.notnull()
s.name = "Wartosc"
s.index.name = "Klucz"
print(s)
## key0      NaN
## key2    700.0
## key3     70.0
## key1    350.0
## dtype: float64
## key0     True
## key2    False
## key3    False
## key1    False
## dtype: bool
## key0    False
## key2     True
## key3     True
## key1     True
## dtype: bool
## key0     True
## key2    False
## key3    False
## key1    False
## dtype: bool
## key0    False
## key2     True
## key3     True
## key1     True
## dtype: bool
## Klucz
## key0      NaN
## key2    700.0
## key3     70.0
## key1    350.0
## Name: Wartosc, dtype: float64
import pandas as pd

data = {'Country': ['Belgium', 'India', 'Brazil'],
        'Capital': ['Brussels', 'New Delhi', 'Brasília'],
        'Population': [11190846, 1303171035, 207847528]}
frame = pd.DataFrame(data)
print(frame)
df = pd.DataFrame(data, columns=['Country', 'Capital',
                                 'Population'])
print(df)
print(df.iloc[[0], [0]])
print(df.loc[[0], ['Country']])
print(df.loc[2])
print(df.loc[:, 'Capital'])
print(df.loc[1, 'Capital'])
print(df[df['Population'] > 1200000000])
print(df.drop('Country', axis=1))
print(df.shape)
print(df.index)
print(df.columns)
print(df.info())
print(df.count())
##    Country    Capital  Population
## 0  Belgium   Brussels    11190846
## 1    India  New Delhi  1303171035
## 2   Brazil   Brasília   207847528
##    Country    Capital  Population
## 0  Belgium   Brussels    11190846
## 1    India  New Delhi  1303171035
## 2   Brazil   Brasília   207847528
##    Country
## 0  Belgium
##    Country
## 0  Belgium
## Country          Brazil
## Capital        Brasília
## Population    207847528
## Name: 2, dtype: object
## 0     Brussels
## 1    New Delhi
## 2     Brasília
## Name: Capital, dtype: object
## New Delhi
##   Country    Capital  Population
## 1   India  New Delhi  1303171035
##      Capital  Population
## 0   Brussels    11190846
## 1  New Delhi  1303171035
## 2   Brasília   207847528
## (3, 3)
## RangeIndex(start=0, stop=3, step=1)
## Index(['Country', 'Capital', 'Population'], dtype='object')
## <class 'pandas.core.frame.DataFrame'>
## RangeIndex: 3 entries, 0 to 2
## Data columns (total 3 columns):
##  #   Column      Non-Null Count  Dtype
## ---  ------      --------------  -----
##  0   Country     3 non-null      object
##  1   Capital     3 non-null      object
##  2   Population  3 non-null      int64
## dtypes: int64(1), object(2)
## memory usage: 200.0+ bytes
## None
## Country       3
## Capital       3
## Population    3
## dtype: int64
Uzupełnianie braków
import pandas as pd

s = pd.Series([3, -5, 7, 4], index=['a', 'b', 'c', 'd'])
s2 = pd.Series([7, -2, 3], index=['a', 'c', 'd'])
print(s + s2)
print(s.add(s2, fill_value=0))
print(s.mul(s2, fill_value=2))
## a    10.0
## b     NaN
## c     5.0
## d     7.0
## dtype: float64
## a    10.0
## b    -5.0
## c     5.0
## d     7.0
## dtype: float64
## a    21.0
## b   -10.0
## c   -14.0
## d    12.0
## dtype: float64
Obsługa plików csv
Funkcja pandas.read_csv

Dokumentacja: link

Zapis pandas.DataFrame.to_csv

Dokumentacja: link

Obsługa plików z Excela
Funkcja pandas.read_excel

https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html

** Ważne: trzeba zainstalować bibliotekę openpyxl do importu .xlsx oraz xlrd do importu .xls (nie trzeba ich importować w kodzie jawnie w większości wypadków)

Operacje manipulacyjne
merge
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html?highlight=merge#pandas.DataFrame.merge

join
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.join.html?highlight=join#pandas.DataFrame.join

concat
https://pandas.pydata.org/docs/reference/api/pandas.concat.html?highlight=concat#pandas.concat

pivot


https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot.html?highlight=pivot#pandas.DataFrame.pivot

melt


https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.melt.html?highlight=melt#pandas.DataFrame.melt

“Tidy data”
Imię	Wiek	Wzrost	Kolor oczu
Adam	26	167	Brązowe
Sylwia	34	164	Piwne
Tomasz	42	183	Niebieskie
jedna obserwacja (jednostka statystyczna) = jeden wiersz w tabeli/macierzy/ramce danych
wartosci danej cechy znajduja sie w kolumnach
jeden typ/rodzaj obserwacji w jednej tabeli/macierzy/ramce danych
Obsługa brakujących danych
import numpy as np
import pandas as pd

string_data = pd.Series(['aardvark', 'artichoke', np.nan, 'avocado'])
print(string_data)
print(string_data.isnull())
print(string_data.dropna())
## 0     aardvark
## 1    artichoke
## 2          NaN
## 3      avocado
## dtype: object
## 0    False
## 1    False
## 2     True
## 3    False
## dtype: bool
## 0     aardvark
## 1    artichoke
## 3      avocado
## dtype: object
from numpy import nan as NA
import pandas as pd

data = pd.DataFrame([[1., 6.5, 3.], [1., NA, NA],
                     [NA, NA, NA], [NA, 6.5, 3.]])
cleaned = data.dropna()
print(cleaned)
print(data.dropna(how='all'))
data[4] = NA
print(data.dropna(how='all', axis=1))
print(data)
print(data.fillna(0))
print(data.fillna({1: 0.5, 2: 0}))
##      0    1    2
## 0  1.0  6.5  3.0
##      0    1    2
## 0  1.0  6.5  3.0
## 1  1.0  NaN  NaN
## 3  NaN  6.5  3.0
##      0    1    2
## 0  1.0  6.5  3.0
## 1  1.0  NaN  NaN
## 2  NaN  NaN  NaN
## 3  NaN  6.5  3.0
##      0    1    2   4
## 0  1.0  6.5  3.0 NaN
## 1  1.0  NaN  NaN NaN
## 2  NaN  NaN  NaN NaN
## 3  NaN  6.5  3.0 NaN
##      0    1    2    4
## 0  1.0  6.5  3.0  0.0
## 1  1.0  0.0  0.0  0.0
## 2  0.0  0.0  0.0  0.0
## 3  0.0  6.5  3.0  0.0
##      0    1    2   4
## 0  1.0  6.5  3.0 NaN
## 1  1.0  0.5  0.0 NaN
## 2  NaN  0.5  0.0 NaN
## 3  NaN  6.5  3.0 NaN
Usuwanie duplikatów
import pandas as pd

data = pd.DataFrame({'k1': ['one', 'two'] * 3 + ['two'],
                     'k2': [1, 1, 2, 3, 3, 4, 4]})
print(data)
print(data.duplicated())
print(data.drop_duplicates())
##     k1  k2
## 0  one   1
## 1  two   1
## 2  one   2
## 3  two   3
## 4  one   3
## 5  two   4
## 6  two   4
## 0    False
## 1    False
## 2    False
## 3    False
## 4    False
## 5    False
## 6     True
## dtype: bool
##     k1  k2
## 0  one   1
## 1  two   1
## 2  one   2
## 3  two   3
## 4  one   3
## 5  two   4
Zastępowanie wartościami
import pandas as pd
import numpy as np

data = pd.Series([1., -999., 2., -999., -1000., 3.])
print(data)
print(data.replace(-999, np.nan))
print(data.replace([-999, -1000], np.nan))
print(data.replace([-999, -1000], [np.nan, 0]))
print(data.replace({-999: np.nan, -1000: 0}))
## 0       1.0
## 1    -999.0
## 2       2.0
## 3    -999.0
## 4   -1000.0
## 5       3.0
## dtype: float64
## 0       1.0
## 1       NaN
## 2       2.0
## 3       NaN
## 4   -1000.0
## 5       3.0
## dtype: float64
## 0    1.0
## 1    NaN
## 2    2.0
## 3    NaN
## 4    NaN
## 5    3.0
## dtype: float64
## 0    1.0
## 1    NaN
## 2    2.0
## 3    NaN
## 4    0.0
## 5    3.0
## dtype: float64
## 0    1.0
## 1    NaN
## 2    2.0
## 3    NaN
## 4    0.0
## 5    3.0
## dtype: float64
Dyskretyzacja i podział na koszyki
import pandas as pd

ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)
print(cats)
print(cats.codes)
print(cats.categories)
print(pd.value_counts(cats))
## [(18, 25], (18, 25], (18, 25], (25, 35], (18, 25], ..., (25, 35], (60, 100], (35, 60], (35, 60], (25, 35]]
## Length: 12
## Categories (4, interval[int64, right]): [(18, 25] < (25, 35] < (35, 60] < (60, 100]]
## [0 0 0 1 0 0 2 1 3 2 2 1]
## IntervalIndex([(18, 25], (25, 35], (35, 60], (60, 100]], dtype='interval[int64, right]')
## (18, 25]     5
## (25, 35]     3
## (35, 60]     3
## (60, 100]    1
## dtype: int64
import pandas as pd

ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
cats2 = pd.cut(ages, [18, 26, 36, 61, 100], right=False)
print(cats2)
group_names = ['Youth', 'YoungAdult',
               'MiddleAged', 'Senior']
print(pd.cut(ages, bins, labels=group_names))
## [[18, 26), [18, 26), [18, 26), [26, 36), [18, 26), ..., [26, 36), [61, 100), [36, 61), [36, 61), [26, 36)]
## Length: 12
## Categories (4, interval[int64, left]): [[18, 26) < [26, 36) < [36, 61) < [61, 100)]
## ['Youth', 'Youth', 'Youth', 'YoungAdult', 'Youth', ..., 'YoungAdult', 'Senior', 'MiddleAged', 'MiddleAged', 'YoungAdult']
## Length: 12
## Categories (4, object): ['Youth' < 'YoungAdult' < 'MiddleAged' < 'Senior']
import pandas as pd
import numpy as np

data = np.random.rand(20)
print(pd.cut(data, 4, precision=2))
## [(0.081, 0.29], (0.29, 0.5], (0.29, 0.5], (0.29, 0.5], (0.081, 0.29], ..., (0.5, 0.7], (0.5, 0.7], (0.5, 0.7], (0.081, 0.29], (0.7, 0.91]]
## Length: 20
## Categories (4, interval[float64, right]): [(0.081, 0.29] < (0.29, 0.5] < (0.5, 0.7] < (0.7, 0.91]]
import pandas as pd
import numpy as np

data = np.random.randn(1000)
cats = pd.qcut(data, 4)
print(cats)
print(pd.value_counts(cats))
## [(-0.606, -0.0364], (0.601, 2.906], (0.601, 2.906], (-0.606, -0.0364], (0.601, 2.906], ..., (-0.606, -0.0364], (0.601, 2.906], (-0.0364, 0.601], (-0.606, -0.0364], (0.601, 2.906]]
## Length: 1000
## Categories (4, interval[float64, right]): [(-3.3489999999999998, -0.606] < (-0.606, -0.0364] < (-0.0364, 0.601] <
##                                            (0.601, 2.906]]
## (-3.3489999999999998, -0.606]    250
## (-0.606, -0.0364]                250
## (-0.0364, 0.601]                 250
## (0.601, 2.906]                   250
## dtype: int64
Wykrywanie i filtrowanie elementów odstających
import pandas as pd
import numpy as np

data = pd.DataFrame(np.random.randn(1000, 4))
print(data.describe())
col = data[2]
print(col[np.abs(col) > 3])
print(data[(np.abs(data) > 3).any(1)])
##                  0            1            2            3
## count  1000.000000  1000.000000  1000.000000  1000.000000
## mean     -0.057109     0.012727     0.045559    -0.064776
## std       0.985574     1.006186     0.992972     1.008053
## min      -2.871763    -3.826926    -3.477464    -3.300333
## 25%      -0.728968    -0.661943    -0.597260    -0.727630
## 50%      -0.054641     0.040948     0.054566    -0.045455
## 75%       0.617440     0.633324     0.706807     0.587376
## max       3.007700     2.994164     3.011782     2.917422
## 127    3.011782
## 229   -3.368881
## 963   -3.218372
## 997   -3.477464
## Name: 2, dtype: float64
##             0         1         2         3
## 127 -0.175062  1.782662  3.011782 -2.371048
## 229 -0.253300 -0.120071 -3.368881 -1.770955
## 502  3.006265 -0.696815  0.010320  0.239408
## 669  0.816250 -3.826926 -0.177378 -0.525864
## 811  1.048825  0.556837  1.602003 -3.300333
## 877  3.007700  0.343596 -1.207129  0.541541
## 963  1.575457 -0.156850 -3.218372  0.252540
## 997  0.384228 -0.597243 -3.477464 -0.037356





Import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
Galerie wykresów
https://matplotlib.org/gallery/index.html

https://python-graph-gallery.com/

https://github.com/rasbt/matplotlib-gallery

https://seaborn.pydata.org/examples/index.html



import matplotlib.pyplot as plt

x = [0, 7, 4, 5,8,-9]

plt.plot(x)

plt.show()
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2, 100)

plt.plot(x, x, label='linear')
plt.plot(x, x**2, label='quadratic')
plt.plot(x, x**3, label='cubic')

plt.xlabel('x label')
plt.ylabel('y label')

plt.title("Simple Plot")

plt.legend()

plt.show()


import numpy as np
import matplotlib.pyplot as plt

x = np.arange(14)
y = np.cos(5 * x)

plt.plot(x, y + 2, 'blue', linestyle="-", label="niebieski")

plt.plot(x, y + 1, 'red', linestyle=":", label="czerwony")

plt.plot(x, y, 'green', linestyle="--", label="zielony")

plt.legend(title='Legenda:')
plt.show()






import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10, 0.2)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([1, 2, 3, 4], [10, 20, 25, 30], color='lightblue', linewidth=3)
ax.scatter([0.3, 3.8, 1.2, 2.5], [11, 25, 9, 26], color='darkgreen', marker='^')
ax.set_xlim(0.5, 4.5)
plt.show()
Kolory

https://matplotlib.org/stable/gallery/color/named_colors.html
https://pl.wikipedia.org/wiki/Lista_kolor%C3%B3w
Markery https://matplotlib.org/stable/api/markers_api.html

import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4], [10, 20, 25, 30], color='lightblue', linewidth=3)
plt.scatter([0.3, 3.8, 1.2, 2.5], [11, 25, 9, 26], color='darkgreen', marker='^')
plt.xlim(0.5, 4.5)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10)
y = x ^ 2
# Labeling the Axes and Title
plt.title("Graph Drawing")
plt.xlabel("Time")
plt.ylabel("Distance")

# Formatting the line colors
plt.plot(x, y, 'r')

# Formatting the line type
plt.plot(x, y, '>')

# save in pdf formats
plt.savefig('timevsdist.pdf', format='pdf')


import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
x1 = x[x < 0]
y1 = 1 / x1
plt.plot(x1, y1)
x2 = x[x > 0]
y2 = 1 / x2
plt.plot(x2, y2)
plt.ylim(-10, 10)
plt.axhline(y=0, linestyle="--")
plt.axvline(x=0, linestyle=":")
plt.show()
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-3, 3, 0.1)
y = x ** 2 + 2 * x
plt.plot(x, y)s
plt.annotate(xy=[-1, 5], text="cos tam")
plt.xticks([-2, 1, 2], color="red")
plt.ylabel("abc", color="green")
plt.show()
import numpy as np
from matplotlib import pyplot as plt

x = np.arange(0, 10)
y = x ^ 2
z = x ^ 3
t = x ^ 4
# Labeling the Axes and Title
plt.title("Graph Drawing")
plt.xlabel("Time")
plt.ylabel("Distance")
plt.plot(x, y)

# Annotate
plt.annotate(xy=[2, 1], text='Second Entry')
plt.annotate(xy=[4, 6], text='Third Entry')
# Adding Legends
plt.plot(x, z)
plt.plot(x, t)
plt.legend(['Race1', 'Race2', 'Race3'], loc=4)
plt.show()
from matplotlib import pyplot as plt

x = [1, -3, 4, 5, 6]
y = [2, 6, -4, 1, 2]
area = [70, 60, 1, 50, 2]
plt.scatter(x, y, marker=">", color="brown", alpha=0.5, s=area)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)

N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = (30 * np.random.rand(N)) ** 2  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()
import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = ['Frogs', 'Hogs', 'Dogs', 'Logs']
sizes = [15, 30, 45, 10]
explode = [0, 0.1, 0, 0]  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
Wersja prostsza:

import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = ['Frogs', 'Hogs', 'Dogs', 'Logs']
sizes = [15, 30, 45, 10]
explode = [0, 0.1, 0, 0]  # only "explode" the 2nd slice (i.e. 'Hogs')

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.axis('equal')

plt.show()


import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,3,0.1)
plt.subplot(2, 2, 1)
plt.plot(x, x)
plt.subplot(2, 2, 2)
plt.plot(x, x * 2)
plt.subplot(2, 2, 3)
plt.plot(x, x * x)
plt.subplot(2, 2, 4)
plt.plot(x, x ** 3)
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, np.pi * 2, 100)
plt.subplot(3, 1, 1)
plt.plot(x, np.sin(x), 'r')
plt.grid(True)
plt.xlim(0, np.pi * 2)
plt.subplot(3, 1, 2)
plt.plot(x, np.cos(x), 'g')
plt.grid(True)
plt.xlim(0, np.pi * 2)
plt.subplot(3, 1, 3)
plt.plot(x, np.sin(x), 'r', x, np.cos(x), 'g')
plt.grid(True)
plt.xlim(0, np.pi * 2)
plt.tight_layout()
plt.savefig("fig3.png", dpi=72)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
x = np.arange(0.01, 10.0, 0.01)
y = x ** 2
ax1.plot(x, y, 'r')
ax2 = ax1.twinx()
y2 = np.sin(x)
ax2.plot(x, y2)
fig.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
t = np.arange(0.01, 10.0, 0.01)
s1 = np.exp(t)
ax1.plot(t, s1, 'b-')
ax1.set_xlabel('time (s)')

ax1.set_ylabel('exp', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
s2 = np.sin(2 * np.pi * t)
ax2.plot(t, s2, 'r.')
ax2.set_ylabel('sin', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

wys = [10, 15, 18, 22, 27]
x = np.arange(0, len(wys))
k = ["black", "red", "green", "yellow", "pink"]
plt.bar(x, wys, color=k, width=0.75)
etyk = ["Kategoria A", "Kategoria B", "Kategoria C", "Kategoria D", "Kategoria E"]
plt.xticks(x, etyk, rotation=45)
y2 = [20, 30, 40, 50, 60]
plt.plot(x, y2)
plt.title("tytulik")
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

height = [3, 12, 5, 18, 45]
bars = ('A', 'B', 'C', 'D', 'E')
y_pos = np.arange(len(bars))
plt.bar(y_pos, height, color=['black', 'red', 'green', 'blue', 'cyan'])
plt.xticks(y_pos, bars)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

data = [[30, 25, 50, 20],
        [40, 23, 51, 17],
        [35, 22, 45, 19]]
X = np.arange(4)

plt.bar(X + 0.00, data[0], color='b', width=0.25, label="A")
plt.bar(X + 0.25, data[1], color='g', width=0.25, label="B")
plt.bar(X + 0.50, data[2], color='r', width=0.25, label="C")
labelsbar = np.arange(2015,2019)
plt.xticks(X+0.25,labelsbar)
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

N = 5

boys = (20, 35, 30, 35, 27)
girls = (25, 32, 34, 20, 25)
ind = np.arange(N)
width = 0.35

plt.bar(ind, boys, width, label="boys")
plt.bar(ind, girls, width,bottom=boys, label="girls")

plt.ylabel('Contribution')
plt.title('Contribution by the teams')
plt.xticks(ind, ('T1', 'T2', 'T3', 'T4', 'T5'))
plt.yticks(np.arange(0, 81, 10))
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

width = [3, 12, 5, 18, 45]
bars = ('A', 'B', 'C', 'D', 'E')
x_pos = np.arange(len(bars))
plt.barh(x_pos, width, color=['black', 'red', 'green', 'blue', 'cyan'])
plt.yticks(x_pos, bars)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

data = [[30, 25, 50, 20],
        [40, 23, 51, 17],
        [35, 22, 45, 19]]
Y = np.arange(4)

plt.barh(Y + 0.00, data[0], color='b', height=0.25, label="A")
plt.barh(Y + 0.25, data[1], color='g', height=0.25, label="B")
plt.barh(Y + 0.50, data[2], color='r', height=0.25, label="C")
labelsbar = np.arange(2015,2019)
plt.yticks(Y + 0.25, labelsbar)
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

N = 5

boys = (20, 35, 30, 35, 27)
girls = (25, 32, 34, 20, 25)
ind = np.arange(N)
height = 0.35

plt.barh(ind, boys, height, label="boys")
plt.barh(ind, girls, height, left=boys, label="girls")

plt.xlabel('Contribution')
plt.title('Contribution by the teams')
plt.yticks(ind, ('T1', 'T2', 'T3', 'T4', 'T5'))
plt.xticks(np.arange(0, 81, 10))
plt.legend()
plt.show()


import matplotlib.pyplot as plt

dane = [1, 4, 5, 6, 3, 9, 7, 20]
plt.boxplot(dane)
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Creating dataset
np.random.seed(10)
data = np.random.normal(100, 20, 200)

# Creating plot
plt.boxplot(data)

# show plot
plt.show()
import matplotlib.pyplot as plt

x = [1, 1, 2, 3, 3, 5, 7, 8, 9, 10,
     10, 11, 11, 13, 13, 15, 16, 17, 18, 18]

plt.hist(x, bins=4)
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
cats2 = pd.cut(ages, [18, 26, 36, 61, 100], right=False)
print(cats2)
group_names = ['Youth', 'YoungAdult',
               'MiddleAged', 'Senior']
data=pd.cut(ages, bins, labels=group_names)
plt.hist(data)
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
plt.hist(ages, bins=bins)
plt.show()
import matplotlib.pyplot as plt

x = [1, 1, 2, 3, 3, 5, 7, 8, 9, 10,
     10, 11, 11, 13, 13, 15, 14, 12, 18, 18]

plt.hist(x, bins=[0,5,10,15,20])
plt.xticks([0,5,10,15,20])
plt.show()
import matplotlib.pyplot as plt

x = [1, 1, 2, 3, 3, 5, 7, 8, 9, 10,
     10, 11, 11, 13, 13, 15, 14, 12, 18, 18]

plt.hist(x, bins=[0,5,10,15,20], cumulative=True)
plt.xticks([0,5,10,15,20])
plt.show()
import matplotlib.pyplot as plt

x = [1, 1, 2, 3, 3, 5, 7, 8, 9, 10,
     10, 11, 11, 13, 13, 15, 14, 12, 18, 18]

plt.hist(x, bins=[0,5,10,15,20], density=True)
plt.xticks([0,5,10,15,20])
plt.show()
Helisa:

⎧⎩⎨x=acos(t)y=asin(t)z=at

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')
t = np.linspace(0, 15, 1000)
a = 3
xline = a * np.sin(t)
yline = a * np.cos(t)
zline = a * t
ax.plot3D(xline, yline, zline)
plt.show()
Torus

p(α, β)=((R+rcosα)cosβ, (R+rcosα)sinβ, rsinα)

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')
r = 1
R = 5
alpha = np.arange(0, 2 * np.pi, 0.1)
beta = np.arange(0, 2 * np.pi, 0.1)
alpha, beta = np.meshgrid(alpha, beta)
x = (R + r * np.cos(alpha)) * np.cos(beta)
y = (R + r * np.cos(alpha)) * np.sin(beta)
z = r * np.sin(alpha)
ax.plot_wireframe(x, y, z)
plt.show()