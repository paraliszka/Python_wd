import math
#zadanie 1
a = "a"
b = "b"
print(a+b)
a1 = 1
b1 = 2
print(a1+b1)
a2 = 5.2
b2 = 4.9
print(a2+b2)
a3 = 2+3j
b3 = 2+4j
print(a3*b3)
#zadanie 2
x = 2
y = 3
print((x / y), (x * y), (x + y), (x - y), (x // y), (x // y), (x % y), (x ** y), sep=' ')
#zadanie 3
x += y
x -= y
x *= y
x /= y
x **= y
x %= y
print( x , y )
#zadanie 4
print(math.e ** 10)
print(math.log(5 + math.sin(8) ** 2, math.e) ** (1/6))
print(math.floor(3.55), math.ceil(4.80), sep=', ')
#zadanie 5
name = "ANDRZEJ"
surname = "PLISZKA"
name = name.capitalize()
surname = surname.capitalize()
print(name, surname, sep=' ')
#zadanie 6
xd = "bbab a a a a a a a bbbbcccaaaaaa a a a"
print("ilosc slow 'a': ", xd.count(' a'))
#zadanie 7
ab420 = "X1DDDDDX"
print(ab420[1], ab420[len(ab420)-1], sep='   ')
#zadanie 8
print(xd.split())
#zadanie 9
zmienna_string = "XD xd xD"
zmienna_float = 420.42
zmienna_szestastkowy = 0xFFFF
print(zmienna_string)
print(zmienna_float)
print('{0:x}'.format(zmienna_szestastkowy))