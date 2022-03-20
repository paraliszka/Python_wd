# #zadanie 1
sport = ['koszykowka','e-sport','workout']
sport.reverse()
sport.append('mniej')
sport.append('lubiane')
sport.append('sporty')
# #zadanie 2
slownik = {'idk': 'I don t know', 'btw': 'by the way'}
# #zadanie 3
gry = {1:'Leauge of Legends', 2: 'Leauge of Legends', 3: 'Leauge of Legends'}
print(len(gry))
#zadanie 4
a = input("aaa? aaa?: ")
print("aaa:  " + str(len(a)))
# #zadanie 5
import sys
x = int(sys.stdin.readline())
y = int(sys.stdin.readline())
z = int(sys.stdin.readline())
aaaa = (x ** y) + z
sys.stdout.write(str(aaaa))
#zadanie 6
a1, b1, c1 = input("podaj 3 liczby: ").split()

liczby = [a1, b1, c1]
max = a1
for x in liczby:
    if x > max:
        max = x
print("największa liczba to: ", max)
#zadanie 7
listaa = [ 1, 2, 3.5, 5, 21.37]

i=0
for x in listaa:
   listaa[i] = x**2
   i+=1

print(listaa)
#zadanie 8
parzyste = []
x=0
while x < 3:
    m = int(input())
    if m%2==0:
        parzyste.append(m)
    x += 1
print(parzyste)
#zadanie 9
coo = int(input())
import math
try:
    print(math.sqrt(coo))
except ValueError:
    print("nie wyciągamy pierwiastka z liczby ujemnej")
