# -*- coding: utf-8 -*-
import re

class SpellSuggester:

    """
    Clase que implementa el método suggest para la búsqueda de términos.
    """

    def __init__(self,
                 dist_functions,
                 vocab = [],
                 default_distance = None,
                 default_threshold = None):
        
        """Método constructor de la clase SpellSuggester

        Construye una lista de términos únicos (vocabulario),

        Args:
           dist_functions es un diccionario nombre->funcion_distancia
           vocab es una lista de palabras o la ruta de un fichero
           default_distance debe ser una clave de dist_functions
           default_threshold un entero positivo

        """
        self.distance_functions = dist_functions
        self.set_vocabulary(vocab)
        if default_distance is None:
            default_distance = 'levenshtein'
        if default_threshold is None:
            default_threshold = 3
        self.default_distance = default_distance
        self.default_threshold = default_threshold

    def build_vocabulary(self, vocab_file_path):
        """Método auxiliar para crear el vocabulario.

        Se tokeniza por palabras el fichero de texto,
        se eliminan palabras duplicadas y se ordena
        lexicográficamente.

        Args:
            vocab_file (str): ruta del fichero de texto para cargar el vocabulario.
            tokenizer (re.Pattern): expresión regular para la tokenización.
        """
        tokenizer=re.compile(r"\W+")
        with open(vocab_file_path, "r", encoding="utf-8") as fr:
            vocab = set(tokenizer.split(fr.read().lower()))
            vocab.discard("")  # por si acaso
            return sorted(vocab)

    def set_vocabulary(self, vocabulary):
        if isinstance(vocabulary,list):
            self.vocabulary = vocabulary # atención! nos quedamos una referencia, a tener en cuenta
        elif isinstance(vocabulary,str):
            self.vocabulary = self.build_vocabulary(vocabulary)
        else:
            raise Exception("SpellSuggester incorrect vocabulary value")

    def suggest(self, term, distance=None, threshold=None, flatten=True):
        """

        Args:
            term (str): término de búsqueda.
            distance (str): nombre del algoritmo de búsqueda a utilizar
            threshold (int): threshold para limitar la búsqueda
        """
        # Si hemos especificado una función de distancia o un umbral en los parámetros, lo usamos.
        # Si no, empleamos sus valores por defecto ('levenshtein' y 3, respectivamente):
        if distance is None:
            distance = self.default_distance
        dist_func = self.distance_functions[distance]
        if threshold is None:
            threshold = self.default_threshold

        # Creamos una lista de listas para cada posible distancia entre 0 y 'threshold':
        resul = [[] for _ in range(threshold + 1)]

        # Calculamos la distancia de cada palabra de nuestro vocabulario al término 'term' pasado como parámetro.
        # Si dicha distancia es menor o igual que el umbral (ponemos '< threshold + 1' para marcar la relación con la talla de 'resul'),
        # añadimos la palabra a la lista asociada a dicha distancia dentro de 'resul':
        for word in self.vocabulary:
            dist = dist_func(word, term, threshold)
            if dist < threshold + 1: resul[dist].append(word)
        
        # Si queremos una sola lista de palabras en vez de separarlas en sublistas según su distancia, lo especificamos en el parámetro 'flatten':
        if flatten:
            resul = [word for wlist in resul for word in wlist]
            
        return resul

