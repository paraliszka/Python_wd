import math
# lista = []
# for element in sekwencja
#     if warunek_na_element
#         lista append(akcja na element)
# lista=(akcja_na_element for element in sekwencja if warunek_na_element]

#a = {x^2: x w <0,9>} b={1,3,9,27,3^5}
#c =
# a = []
# for x in range(0,10):
#     a.append(x**2)
# print(a)
# a2=[]
# a2 = [x**2 for x in range(0,10)]
# print(a2)
# b=[]
# for x in range(0, 6):
#     b.append(3**x)
# print(b)
# b2 = [3**x for x in range(0,6)]
# print(b2)
# c = []
#
#
# for x in a:
#     if(x%2==1):
#         c.append(x)
# print(c)
# c2=[x for x in a if(x**2%2==1)]
# print(c2)

# lista=[]
# for a in [1,2,3]:
#     for b in [4,5,6]:
#         lista.append((a,b))
# print(lista)
# lista2 = [(a,b) for a in [1,2,3] for b in [1,2,3]]
# print(lista2)

# slownik = {'PZU': 'Państwowy zakład ubezpieczeń',
#            'ZUS': 'Zaklad ubezoieczeń społecznych',
#            'PKO': 'Państwowa kasa oszczędnościowa'}
#
# slownik_odwrocowny= {}
# for key, value in slownik.items():
#     slownik_odwrocowny[value] = key
# print(slownik_odwrocowny)
# slownik2 = {value: key for key, value in slownik.items()}
# print(slownik2)
# def nazwa_funkcji(arg pozycyjne, domyślna wartość,
#                 * argument, **argument)
#     instrukcje
#     return
# def row_kwadratowe(a, b, c):
#     delta = b**2 - 4 * a * c
#
#     if delta < 0:
#         print("brak")
#         return -1
#     elif delta == 0:
#         print("jedno")
#         x = (-b)/(2*a)
#         return x
#     else:
#         print('dwa rozwiazania')
#         x1 = ((-b) - math.sqrt(delta))/(2*a)
#         x2 = ((-b) + math.sqrt(delta))/(2*a)
#         return x1, x2
#
# print(row_kwadratowe(6, 1, 3))
# print(row_kwadratowe(1, 2, 1))
# print(row_kwadratowe(1, 4, 1))


# def sprawdz(x):
#     if x%2==0:
#         print("parzysta:" , x)
#     else:
#         print("nie parzysta: " , x)
#
# sprawdz(5)
# sprawdz(2)
# sprawdz(0)

# def dlugosc_odcnika(x1=1, y1=2, x2 = 3,y2=4):
#     return math.sqrt(((x2-x1)**2)**2 + (y2-y1)**2)
#
# #argumenty domyslne
# print(dlugosc_odcnika())
# #argumenty pozycyjne
# print(dlugosc_odcnika(4,5,6,9))
#
# print(dlugosc_odcnika(1, 1 , y2=8, x2 = 7))
#
# print(dlugosc_odcnika(y2=3, x1 = 5, y1 = 6, x2 = 0))
#
# print(dlugosc_odcnika(y2=1, x2=6))

# def ciag(* liczba):
#     if len(liczba) == 0:
#         return 0
#     else:
#         suma = 0
#         for i in liczba:
#             suma += 1
#         return suma
#
# print(ciag())
# print(ciag(1,2,3,4,5,6,7,8))

def oo_lubie(** rzeczy):
    for cos in rzeczy:
        print('lubie')
        print(cos)
        print('co lubie')
        print(rzeczy[cos])

oo_lubie(gry=['planszowe', 'komputerowe'])
