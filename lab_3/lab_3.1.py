import random

#zadanie 1
A = [1-x for x in range(1,10)]
B = [4 ** x for x in range(7)]
C = [x for x in B if x%2==0]
print(C)
#zadanie 2

lista1 = [random.randint(0,100) for x in range(11)]
print(lista1)
lista2 = [x for x in lista1 if x%2==0]
print(lista2)
#zadanie 3