######################################################################
#
# INTEGRANTES DEL EQUIPO/GRUPO:
#
# - José Miguel Alonso Fontenele
# - Álvaro Sanjuán Piqueras
# - José María Rodríguez Charco
#
######################################################################


import numpy as np

def levenshtein_matriz(x, y):
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int32)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
            )
    return D[lenX, lenY]

def levenshtein_edicion(x, y):
    # Calculamos la matriz de distancias reutilizando el código de 'levenshtein_matriz':
    lenX, lenY = len(x), len(y)

    # > Creamos una matriz de ceros y resolvemos parcialmente el problema de contorno rellenando la primera columna:
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int32)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    
    # > A continuación, vamos columna por columna resolviendo el problema de contorno para el primer elemento de cada una y 
    #    eligiendo el movimiento óptimo para el resto:
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
            )
    
    # Una vez rellena nuestra matriz, partimos de la entrada (lenX, lenY) y vamos calculando inversamente los movimientos
    # que hemos usado para llegar hasta ella comparando su valor con el resultado de aplicar cada operación de edición:
    i, j = lenX, lenY
    res = []
    while D[i][j] != 0:
        aux = D[i][j]

        # > Inserción:
        if D[i - 1][j] + 1 == aux:
            i -= 1
            res.append((x[i], ''))
        
        # > Borrado:
        elif D[i][j - 1] + 1 == aux:
            j -= 1
            res.append(('', y[j]))
        
        # > Sustitución:
        else:
            i -= 1
            j -= 1
            res.append((x[i], y[j]))
    
    # Si hemos salido del while, significa que estamos en una entrada de valor 0, lo que puede significar que, o bien hemos
    # llegado a la entrada (0, 0), o bien solo nos queda realizar sustituciones (estamos en una entrada (k, k) donde k > 0):
    while i > 0 and j > 0:
        i -= 1
        j -= 1
        res.append((x[i], y[j]))
    
    # Como hemos seguido el camino inverso, nuestra secuencia de movimientos está al revés, por lo que le damos la vuelta y la
    # devolvemos:
    res.reverse()
    return D[lenX, lenY], res


def damerau_restricted_matriz(x, y):
    # La implementación restringida de Damerau-Levenshtein es igual a la de 'levenshtein_matriz', pero añadiendo una nueva
    # condición para aquellos casos donde podamos transponer 2 letras contiguas:
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int32)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
                D[i - 2][j - 2] + 1 if i > 1 and j > 1 and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2] else float('inf') # > Condición nueva
            )

    return D[lenX, lenY]

def damerau_restricted_edicion(x, y):
    # Igual que con 'damerau_restricted_matriz', este método también es muy similar a su homólogo de Levenshtein: 'levenshtein_edicion'.
    # Nos bastará nuevamente con añadir una condición adicional para contemplar los casos donde hayamos realizado una transposición:
    
    # Código de damerau_restricted_matriz:
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int32)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
                D[i - 2][j - 2] + 1 if i > 1 and j > 1 and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2] else float('inf')
            )

    # Cálculo inverso del camino:
    i, j = lenX, lenY
    res = []
    while D[i][j] != 0:
        aux = D[i][j]

        # > Inserción:
        if D[i - 1][j] + 1 == aux:
            i -= 1
            res.append((x[i], ''))
        
        # > Borrado:
        elif D[i][j - 1] + 1 == aux:
            j -= 1
            res.append(('', y[j]))
        
        # > Sustitución:
        elif D[i - 1][j - 1] + (x[i - 1] != y[j - 1]) == aux:
            i -= 1
            j -= 1
            res.append((x[i], y[j]))
        
        # > Transposición:
        else:
            i -= 2
            j -= 2
            res.append((x[i : i + 2], y[j : j + 2]))

    while i > 0 and j > 0:
        i -= 1
        j -= 1
        res.append((x[i], y[j]))

    res.reverse()
    return D[lenX, lenY], res

def damerau_intermediate_matriz(x, y):
    # Partiremos de la implementación de 'damerau_restricted_matriz' y le añadiremos las 2 nuevas condiciones de la ecuación recursiva
    # de la versión intermedia:
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int32)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
                D[i - 2][j - 2] + 1 if i > 1 and j > 1 and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2] else float('inf'),
                D[i - 2][j - 3] + 2 if i > 1 and j > 2 and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 3] else float('inf'), # > Condición nueva
                D[i - 3][j - 2] + 2 if i > 2 and j > 1 and x[i - 3] == y[j - 1] and x[i - 1] == y[j - 2] else float('inf')  # > Condición nueva
            )

    return D[lenX, lenY]

def damerau_intermediate_edicion(x, y):
    # Partiremos de la implementación de 'damerau_restricted_edicion' y le añadiremos las 2 nuevas condiciones de la ecuación recursiva
    # de la versión intermedia:

    # Código de 'damerau_intermediate_matriz':
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int32)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
                D[i - 2][j - 2] + 1 if i > 1 and j > 1 and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2] else float('inf'),
                D[i - 2][j - 3] + 2 if i > 1 and j > 2 and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 3] else float('inf'),
                D[i - 3][j - 2] + 2 if i > 2 and j > 1 and x[i - 3] == y[j - 1] and x[i - 1] == y[j - 2] else float('inf')
            )
    
    # Cálculo inverso del camino:
    i, j = lenX, lenY
    res = []
    while D[i][j] != 0:
        aux = D[i][j]

        # > Inserción:
        if D[i - 1][j] + 1 == aux:
            i -= 1
            res.append((x[i], ''))
        
        # > Borrado:
        elif D[i][j - 1] + 1 == aux:
            j -= 1
            res.append(('', y[j]))
        
        # > Sustitución:
        elif D[i - 1][j - 1] + (x[i - 1] != y[j - 1]) == aux:
            i -= 1
            j -= 1
            res.append((x[i], y[j]))
        
        # > Transposición (ba -> ab):
        elif D[i - 2][j - 2] + 1 == aux and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2]:
            i -= 2
            j -= 2
            res.append((x[i : i + 2], y[j : j + 2]))

        # > Transposición (bca -> ab):
        elif j > 2 and D[i - 2][j - 3] + 2 == aux and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 3]:
            i -= 2
            j -= 3
            res.append((x[i : i + 2], y[j : j + 3]))

        # > Transposición (ab -> bca):
        else:
            i -= 3
            j -= 2
            res.append((x[i : i + 3], y[j : j + 2]))

    while i > 0 and j > 0:
        i -= 1
        j -= 1
        res.append((x[i], y[j]))

    res.reverse()
    return D[lenX, lenY], res
    
opcionesSpell = {
    'levenshtein_m': levenshtein_matriz,
    'damerau_rm':    damerau_restricted_matriz,
    'damerau_im':    damerau_intermediate_matriz,
}

opcionesEdicion = {
    'levenshtein': levenshtein_edicion,
    'damerau_r':   damerau_restricted_edicion,
    'damerau_i':   damerau_intermediate_edicion
}

if __name__ == "__main__":
    print(damerau_intermediate_edicion("algoritmo","algortximo"))