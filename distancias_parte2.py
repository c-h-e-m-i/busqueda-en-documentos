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
from collections import Counter

INF = np.iinfo(np.int32).max

def levenshtein_matriz(x, y, threshold=None):
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

def levenshtein_reduccion(x, y, threshold=None):
    # Nos guardamos la longitud de cada palabra, y creamos dos vectores: uno para la columna actual y otro para la anterior:
    lenX, lenY = len(x), len(y)
    V0 = [0] * (lenX + 1)
    V1 = [0] * (lenX + 1)

    # Resolvemos el problema de contorno rellenando la primera columna con valores entre 0 y 'lenX':
    for i in range(1, lenX + 1):
        V0[i] = V0[i - 1] + 1

    # En cada columna, rellenamos la primera posición con un valor 1 unidad mayor que el de la columna anterior en cada iteración:
    for j in range(1, lenY + 1):
        V1[0] = V0[0] + 1

        # Para el resto de posiciones de la columna, elegimos un valor u otro según la operación de edición que menos nos cueste:
        for i in range(1, lenX + 1):
            V1[i] = min(
                V1[i - 1] + 1,
                V0[i] + 1,
                V0[i - 1] + (x[i - 1] != y[j - 1]),
            )
        
        # Intercambiamos las columnas para que la actual pase a ser la anterior en la siguiente iteración:
        V0, V1 = V1, V0
    
    # Devolvemos el último elemento de la columna actual (intercambiada para guardarse en V0) tras la última iteración:
    return V0[lenX]

def levenshtein(x, y, threshold):
    # El código de esta función es similar al de 'levenshtein_reduccion', pero añadiendo un umbral:
    lenX, lenY = len(x), len(y)
    V0 = [0] * (lenX + 1)
    V1 = [0] * (lenX + 1)

    for i in range(1, lenX + 1):
        V0[i] = V0[i - 1] + 1
    for j in range(1, lenY + 1):
        V1[0] = V0[0] + 1
        for i in range(1, lenX + 1):
            V1[i] = min(
                V1[i - 1] + 1,
                V0[i] + 1,
                V0[i - 1] + (x[i - 1] != y[j - 1]),
            )
        V0, V1 = V1, V0

        # > Si incluso el camino de menor coste se pasa del umbral ('threshold') en la última columna procesada, devolvemos 'threshold' + 1:
        if min(V0) > threshold: return threshold + 1

    # Ya que 'V0[lenX]' puede no ser el mínimo de su columna, lo comparamos con el umbral (+ 1) y devolvemos el menor de los dos:
    return min(threshold + 1, V0[lenX])

def levenshtein_cota_optimista(x, y, threshold):
    # La función 'Counter' cuenta el número de apariciones de cada letra en una cadena y lo guarda en un objeto de tipo 'Counter' (subclase de diccionario):
    count_x = Counter(x)
    count_y = Counter(y)

    # Puedes realizar sumas y restas sobre objetos 'Counter':
    # > Calculamos el mínimo de borrados/sustituciones a realizar sobre la cadena 'x':
    pos = sum((count_x - count_y).values()) # 'Counter' quita automáticamente los negativos al restar.

    # > Calculamos el mínimo de inserciones/sustituciones a realizar sobre la cadena 'y':
    neg = sum((count_y - count_x).values()) # Cambiamos el orden y así nos da la suma de los "negativos" en valor absoluto.
    
    # Si el máximo de esos mínimos es mayor que el umbral, devolvemos 'threshold' + 1. Si no, llamo a la función 'levenshtein' pasándole el umbral
    # (esto es porque aún no tenemos asegurado que no vayamos a sobrepasarlo, solo que la cota inferior no lo hace):
    if max(pos, neg) > threshold: return threshold + 1
    return levenshtein(x, y, threshold)

def damerau_restricted(x, y, threshold):
    # El código de esta función es similar a 'levenshtein' (con umbral), pero añadiendo la expansión de Damerau (restringida).
    # Empezamos, como siempre, guardando las longitudes de las cadenas 'x' e 'y', pero ahora creamos vectores para 3 columnas en vez de 2,
    # pues para la distancia de Damerau-Levenshtein restringida es posible que tengamos que consultar 2 columnas atrás desde la actual:
    lenX, lenY = len(x), len(y)
    V0 = [0] * (lenX + 1)
    V1 = [0] * (lenX + 1)
    V2 = [0] * (lenX + 1)

    # Resolvemos el problema de contorno rellenando la columna 'V1' con valores entre 1 y 'lenX':
    for i in range(1, lenX + 1):
        V1[i] = V1[i - 1] + 1

    # A continuación, rellenamos la columna actual 'V2' con valores incrementales en su primera posición, y con el mínimo entre aplicar
    # cada operación de edición para el resto (nótese que, en la primera iteración, 'V0' no se utiliza):
    for j in range(1, lenY + 1):
        V2[0] = V1[0] + 1
        for i in range(1, lenX + 1):
            V2[i] = min(
                V2[i - 1] + 1,
                V1[i] + 1,
                V1[i - 1] + (x[i - 1] != y[j - 1]),
                V0[i - 2] + 1 if i > 1 and j > 1 and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2] else INF
            )
        
        # Intercambiamos las columnas para que la anterior pase a estar 2 columnas atrás, y la actual pase a ser la anterior: 
        V0, V1, V2 = V1, V2, V0

        # Si en el camino de menor coste de la columna actual nos pasamos del umbral, lo devolvemos sumándole 1:
        if min(V1) > threshold: return threshold + 1
    
    # Devolvemos el mínimo entre el umbral (+ 1) y el último elemento de la columna 'V1' (que era la actual en la última iteración):
    return min(threshold + 1, V1[lenX])

def damerau_intermediate(x, y, threshold):
    # La lógica de esta función es similar a las anteriores, pero ahora necesitaremos 4 columnas,
    # pues en la versión intermedia de Damerau-Levenshtein podemos llegar a consultar hasta 3 columnas más atrás:
    lenX, lenY = len(x), len(y)
    V0 = [0] * (lenX + 1)
    V1 = [0] * (lenX + 1)
    V2 = [0] * (lenX + 1)
    V3 = [0] * (lenX + 1)

    # Resolvemos problema de contorno:
    for i in range(1, lenX + 1):
        V2[i] = V2[i - 1] + 1

    # Rellenamos incrementalmente la primera posición de la columna actual, y el resto las rellenamos añadiendo las restricciones de
    # la versión intermedia:
    for j in range(1, lenY + 1):
        V3[0] = V2[0] + 1
        for i in range(1, lenX + 1):
            V3[i] = min(
                V3[i - 1] + 1,
                V2[i] + 1,
                V2[i - 1] + (x[i - 1] != y[j - 1]),
                V1[i - 2] + 1 if i > 1 and j > 1 and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2] else INF,
                V0[i - 2] + 2 if i > 1 and j > 2 and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 3] else INF,
                V1[i - 3] + 2 if i > 2 and j > 1 and x[i - 3] == y[j - 1] and x[i - 1] == y[j - 2] else INF
            )
        V0, V1, V2, V3 = V1, V2, V3, V0
        if min(V2) > threshold: return threshold + 1
        
    return min(threshold + 1, V2[lenX])

opcionesSpell = {
    'levenshtein_m': levenshtein_matriz,
    'levenshtein_r': levenshtein_reduccion,
    'levenshtein':   levenshtein,
    'levenshtein_o': levenshtein_cota_optimista,
    'damerau_r':     damerau_restricted,
    'damerau_i':     damerau_intermediate
}

if __name__ == "__main__":
    print(levenshtein("acb", "ba", 1))
