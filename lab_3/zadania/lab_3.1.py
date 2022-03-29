import random

#zadanie 1
A = [1-x for x in range(1,10)]
B = [4 ** x for x in range(7)]
C = [x for x in B if x%2==0]
print(C)
#zadanie 2

lista1 = [random.randint(0,100) for x in range(10)]
print(lista1)
lista2 = [x for x in lista1 if x%2==0]
print(lista2)
#zadanie 3
slownik = {'jablko': 'kg.', 'jajka':'szt.', 'woda': 'l.', 'crystal':'szt.'}
slownik_nowa = [key for key in slownik.keys() if slownik[key] == 'szt.']
print(slownik)
print(slownik_nowa)
#zadanie 4
def czy_kwadratowy(a,b,c):
    if((a**2+b**2)==(c**2)):
        print('tak, jest to trojkat prostokatny')
    else:
        print('to nie jest trojkat prostokatny')
czy_kwadratowy(3,4,5)
#zadanie 5
def trapez(a = 1,b = 3,h = 2):
    return [(a+b)*h]/2
#zadanie 6
def licz_iloczyn(a=1,b=4,ile=10):
    ciag = [a*(b**x) for x in range(ile)]
    i = 0
    for x in ciag:
        x *= ciag[i]
        i+=1
    return x
print(licz_iloczyn(1,4,10))

#zadenie 7
def idk_zad7(* liczby):
    if len(liczby)==0:
        return 0
    else:
        a = 1
        for x in liczby:
            a = a * x
        return a

print(idk_zad7(1,3,4,5,6,7,1,2))
#zadanie 8

def zakupy(** cos):
    ilosc = 0
    koszt = 0
    for key,value in cos:
        ilosc+=1
        koszt += value

    print('produkty : ', ilosc)
    return koszt
#zadanie 9
from ciagi import *
print(arytmetyczny.n_ty_wyraz(6,6,2))
print(arytmetyczny.suma_ciagu(6,16,6))
print(arytmetyczny.n_ty_wyraz(1,4,11))

