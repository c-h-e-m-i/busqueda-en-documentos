from distancias_parte1 import *

test = [
    ("camarero","caramelos"),
    ("ejemplo","campos"),
    ("algoritmo","algortimo"),
    ("algoritmo","algortximo"),
    ("algoritmo","lagortimo"),
    ("algoritmo","agaloritom"),
    ("algoritmo","algormio"),
    ("acb","ba")]

lon_primera = max(len(x) for x,y in test)
lon_segunda = max(len(y) for x,y in test)

lontotal = lon_primera+1+lon_segunda+sum(1+len(name) for name in opcionesSpell.keys())

print(' '*(2+lon_primera+lon_segunda),end="")
for name in opcionesSpell.keys():
    print(name,end=" ")
print()

for x,y in test:
    print(f"{x:{lon_primera}} {y:{lon_segunda}}",end="")
    for name,function in opcionesSpell.items():
        print(f" {function(x,y):{len(name)}}",end="")
    print()
    
                 
"""
Salida del programa:

                     levenshtein_m damerau_rm damerau_im 
camarero  caramelos              4          4          4
ejemplo   campos                 5          5          5
algoritmo algortimo              2          1          1
algoritmo algortximo             3          3          2
algoritmo lagortimo              4          2          2
algoritmo agaloritom             5          4          3
algoritmo algormio               3          3          2
acb       ba                     3          3          2

"""         
