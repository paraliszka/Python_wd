import random as r

#zadanie 1
lista = [r.randint(0,30) for x in range(10)]
nowa_lista = [x*2 for x in lista]
# plik = open('zad1.txt', 'w')
# for x in nowa_lista:
#     plik.write(str(x) + '\n')
# plik.close()
# print(nowa_lista)

#zadanie 2
# plik = open('zad1.txt', 'r')
# lista = [x for x in plik]
# print(lista)
#zadanie 3
# with open('zad3.txt', 'w') as plik:
#     for b in lista:
#         plik.write(str(b) + '  ')
# with open('zad3.txt', 'r') as plik:
#     for a in plik:
#         print(a)
#zadanie 4
# class NaZakupy:
#     def __init__(self, nazwa_produktu, ilosc, jednostka_miasry, cena_jed):
#         self.nazwa = nazwa_produktu
#         self.ilosc = ilosc
#         self.jednostka = jednostka_miasry
#         self.cena = cena_jed
#     def wyswietl_produkt(self):
#         print("{0}, {1:.2f} {2} w cenie {3}".format(self.nazwa, self.ilosc, self.jednostka, self.cena))
#
#     def ile_produktu(self):
#         print("{} {}".format(self.ilosc, self.jednostka))
#
#     def ile_kosztuje(self):
#         return self.ilosc * self.cena
#
# mleko = NaZakupy('mleko', 2, 'sztuki', 2.50)
# mleko.wyswietl_produkt()
# mleko.ile_produktu()
# print(mleko.ile_kosztuje())
#zadanie 5
class ciagi:
    def __init__(self):
        self.a1 = None
        self.r = None
        self.n = None
        self.lista = []
    def print(self):
        if len(self.lista) == 0:
            self.lista = [self.a1+(self.r*n) for n in range(self.n)]

        print(self.lista)
    def get_parameters(self):
        self.a1 = int(input('podaj wyraz a1: '))
        self.r = int(input('podaj wyraz r: '))
        self.n = int(input('podaj wyraz n: '))
    def get_elements(self):
        XD = input('podaj elementy ciagu odzielone spacja :').split(' ')
        for x in XD:
            self.lista.append(int(x))
    def sum(self, od=0, do=0):
        if do == 0:
            suma = sum(self.lista)
            print('sum: {}'.format(suma))
        else:
            suma = 0
            for x in range(od-1,do):
                suma += self.lista[x]
            print('sum: {}'.format(suma))



XD = ciagi()
XD.get_elements()
XD.print()
XD.sum(4,5)

