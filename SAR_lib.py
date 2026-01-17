import json
import os
import re
import sys
import math
from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle
import numpy as np
import nltk
from scipy.spatial import KDTree
from SAR_semantics import SpacyStaticModel, BetoEmbeddingModel, BetoEmbeddingCLSModel, SentenceBertEmbeddingModel
from nltk.tokenize import sent_tokenize
from spellsuggester import SpellSuggester
from distancias_parte2 import opcionesSpell 

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt_tab')

# INICIO CAMBIO EN v1.1
## UTILIZAR PARA LA AMPLIACION
# Selecciona un modelo semántico
SEMANTIC_MODEL = "SBERT"
#SEMANTIC_MODEL = "BetoCLS"
#SEMANTIC_MODEL = "Beto"
#SEMANTIC_MODEL = "Spacy"
#SEMANTIC_MODEL = "Spacy_noSW_noA"

def create_semantic_model(modelname):
    assert modelname in ("SBERT", "BetoCLS", "Beto", "Spacy", "Spacy_noSW_noA")
    
    if modelname == "SBERT": return SentenceBertEmbeddingModel()    
    elif modelname == "BetoCLS": return BetoEmbeddingCLSModel()
    elif modelname == "Beto": return BetoEmbeddingModel()
    elif modelname == "Spacy": SpacyStaticModel(remove_stopwords=False, remove_noalpha=False)
    return SpacyStaticModel()
# FIN CAMBIO EN v1.1

class SAR_Indexer:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de artículos de Wikipedia
        
        Preparada para todas las ampliaciones:
          posicionales + busqueda semántica + ranking semántico

    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria
    """


    
    # campo que se indexa
    DEFAULT_FIELD = 'all'
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10


    all_atribs = ['urls', 'index', 'docs', 'articles', 'tokenizer', 'show_all', 'positional',
                  'semantic', 'chuncks', 'embeddings', 'chunck_index', 'kdtree', 'artid_to_emb']


    def __init__(self):
        """
        Constructor de la clase SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesarias pero
        puedes añadir más variables si las necesitas. 
            
        """
        self.urls = set() # hash para las urls procesadas,
        self.index = {} # hash para el indice invertido de terminos --> clave: termino, valor: posting list
        self.docs = {} # diccionario de terminos --> clave: entero(docid),  valor: ruta del fichero.
        self.articles = {} # hash de articulos --> clave entero (artid), valor: la info necesaria para diferencia los artículos dentro de su fichero
        self.tokenizer = re.compile(r"\W+") # expresion regular para hacer la tokenizacion
        self.show_all = False # valor por defecto, se cambia con self.set_showall()
        self.positional = False # AÑADIDO POR NOSOTROS: Guarda si hemos hecho un índice posicional o no.

        # PARA LA AMPLIACION:
        self.semantic = None
        self.chuncks = []
        self.embeddings = []
        self.chunck_index = {}
        self.artid_to_emb = {}
        self.kdtree = None
        self.semantic_threshold = None
        self.semantic_ranking = None # ¿¿ ranking de consultas binarias ??
        self.model = None
        self.MAX_EMBEDDINGS = 200 # número máximo de embedding que se extraen del kdtree en una consulta
        
        # ALT:
        self.use_spelling = False
        self.speller = None
        
        
    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################


    def set_spelling(self, use_spelling:bool, distance:str=None, threshold:int=None):
        """
        self.use_spelling a True activa la corrección ortográfica EN LAS PALABRAS NO ENCONTRADAS,
        en caso contrario NO utilizará corrección ortográfica.

        input: "use_spell" booleano, determina el uso del corrector.
                "distance" cadena, nombre de la función de distancia.
                "threshold" entero, umbral del corrector.

        """
        # Hacemos casting a tipo booleano del argumento 'use_spelling' (por si acaso nos lo pasan como número,
        # pues Python no impone las anotaciones de tipo especificadas en la cabecera de la función):
        self.use_spelling = bool(use_spelling)

        # Si no se usa el corrector, lo desactivamos y salimos:
        if not self.use_spelling:
            self.speller = None
            return
        
        # Guardamos como vocabulario las claves de la postings list almacenada en 'self.index':
        vocab = list(self.index.keys())

        # Creamos el corrector:
        self.speller = SpellSuggester(
            opcionesSpell,
            vocab=vocab,
            default_distance=distance,
            default_threshold=threshold
        )

 #----------------------------------------------------------------------   

    def set_showall(self, v:bool):
        """

        Cambia el modo de mostrar los resultados.

        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C

        """
        self.show_all = v


    def set_semantic_threshold(self, v:float):
        """

        Cambia el umbral para la búsqueda semántica.

        input: "v" booleano.

        UTIL PARA LA AMPLIACIÓN

        si self.semantic es False el umbral no tendrá efecto.

        """
        self.semantic_threshold = v

    def set_semantic_ranking(self, v:bool):
        """

        Cambia el valor de semantic_ranking.

        input: "v" booleano.

        UTIL PARA LA AMPLIACIÓN

        si self.semantic_ranking es True se hará una consulta binaria y los resultados se rankearán por similitud semántica.

        """
        self.semantic_ranking = v


    #############################################
    ###                                       ###
    ###      CARGA Y GUARDADO DEL INDICE      ###
    ###                                       ###
    #############################################


    def save_info(self, filename:str):
        """

        Guarda la información del índice en un fichero en formato binario

        """
        info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'wb') as fh:
            pickle.dump(info, fh)

    def load_info(self, filename:str):
        """

        Carga la información del índice desde un fichero en formato binario

        """
        #info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'rb') as fh:
            info = pickle.load(fh)
        atrs = info[0]
        for name, val in zip(atrs, info[1:]):
            setattr(self, name, val)


    ###############################
    ###                         ###
    ###   SIMILITUD SEMANTICA   ###
    ###                         ###
    ###############################

            
    def load_semantic_model(self, modelname:str=SEMANTIC_MODEL):
        """
    
        Carga el modelo de embeddings para la búsqueda semántica.
        Solo se debe cargar una vez
        
        """
        if self.model is None:
            # INICIO CAMBIO EN v1.1
            print(f"loading {modelname} model ... ",end="", file=sys.stderr)             
            self.model = create_semantic_model(modelname)
            print("done!", file=sys.stderr)
            # FIN CAMBIO EN v1.1

            if self.embeddings is not None:
                self.model.kdtree = self.kdtree
                self.model.embeddings = self.embeddings
            else:
                raise ValueError("No se encontraron embeddings en el modelo cargado.")

            

    # INICIO CAMBIO EN v1.2

    # IMPLEMENTADA POR NOSOTROS
    def update_chuncks(self, txt:str, artid:int):
        """
        
        Añade los chuncks (frases en nuestro caso) del texto "txt" correspondiente al articulo "artid" en la lista de chuncks
        Pasos:
            1 - extraer los chuncks de txt, en nuestro caso son las frases. Se debe utilizar "sent_tokenize" de la librerÃ­a "nltk"
            2 - actualizar los atributos que consideres necesarios: self.chuncks, self.embeddings, self.chunck_index y self.artid_to_emb.
        
        """

        # Tokenizamos el string txt y a partir de los chuncks de resultado calculamos los embeddings:
        chuncks = sent_tokenize(txt)
        new_embeddings = self.model.get_embeddings(chuncks)

        # Ya que los índices de los chuncks los obtenemos de forma incremental, calculamos cuántos chuncks nuevos tenemos y creamos una lista con los índices que les corresponderán:
        start_idx = len(self.chuncks)
        end_idx = start_idx + len(chuncks)
        indices = list(range(start_idx, end_idx))

        # Guardamos los chuncks, los embeddings y los índices correspondientes a los chuncks en la entrada correspondiente a este artículo dentro de self.artid_to_emb:
        self.chuncks.extend(chuncks)
        self.embeddings.extend(new_embeddings)
        self.artid_to_emb.setdefault(artid, []).extend(indices)

        # Dentro del índice de chuncks, asociamos que el artículo al que pertenecen es el identificado por artid:
        for idx in indices:
            self.chunck_index[idx] = artid


    # IMPLEMENTADA POR NOSOTROS
    def create_kdtree(self):
        """
        
        Crea el tktree utilizando un objeto de la librerÃ­a SAR_semantics
        Solo se debe crear una vez despues de indexar todos los documentos
        
        # 1: Se debe llamar al mÃ©todo fit del modelo semÃ¡ntico
        # 2: Opcionalmente se puede guardar informaciÃ³n del modelo semÃ¡ntico (kdtree y/o embeddings) en el SAR_Indexer
        
        """

        # Creamos el KD-Tree con el método fit del modelo elegido:
        print(f"Creating kdtree ...", end="")
        self.model.fit(self.chuncks)
        self.kdtree = self.model.kdtree
        self.embeddings = self.model.embeddings  

        print("done!")


    # IMPLEMENTADA POR NOSOTROS
    def solve_semantic_query(self, query:str):
        """

        Resuelve una consulta utilizando el modelo semÃ¡ntico.
        Pasos:
            1 - utiliza el mÃ©todo query del modelo sÃ©mantico
            2 - devuelve top_k resultados, inicialmente top_k puede ser MAX_EMBEDDINGS
            3 - si el Ãºltimo resultado tiene una distancia <= self.semantic_threshold 
                  ==> no se han recuperado todos los resultado: vuelve a 2 aumentando top_k
            4 - tambiÃ©n se puede salir si recuperamos todos los embeddings
            5 - tenemos una lista de chuncks que se debe pasar a artÃ­culos
        """

        # Cargamos el modelo semántico y calculamos los top_k resultados más semejantes:
        self.load_semantic_model() # Carga por defecto el modelo guardado en la constante SEMANTIC_MODEL. En nuestro caso, SBERT.
        top_k = self.MAX_EMBEDDINGS
        top_k_results = self.model.query(query,top_k)

        # Si el último resultado cargado sigue por debajo del umbral semántico (semantic_threshold), aumentamos el valor de top_k y repetimos la consulta:
        while top_k_results and top_k_results[-1][0] <= self.semantic_threshold:
            top_k*=2 # MODIFICACIÓN: Ahora aumentamos top_k duplicándolo en cada iteración
            if top_k > len(self.embeddings):
                top_k_results = self.model.query(query, len(self.embeddings)) # MODIFICACIÓN: Si top_k se pasa del número total de embeddings, hacemos una última consulta con él y salimos del bucle.
                break
            top_k_results = self.model.query(query, top_k)

        # De los resultados obtenidos, nos quedamos solo con los que se encuentren por debajo del umbral semántico:
        retrieved_indices = []
        for d, index in top_k_results:
            if d > self.semantic_threshold:
                break
            retrieved_indices.append(index)
        
        # A partir de cada chunck del resultado, sacamos el artículo al que pertenece y lo añadimos a la lista result_articles:
        result_articles = []
        for idx in retrieved_indices:
            artId = self.chunck_index[idx]
            if artId not in result_articles:
                result_articles.append(artId)

        return result_articles


    # IMPLEMENTADA POR NOSOTROS
    def semantic_reranking(self, query:str, articles: List[int]):
        """

        Ordena los articulos en la lista 'article' por similitud a la consulta 'query'.
        Pasos:
            1 - utiliza el mÃ©todo query del modelo sÃ©mantico
            2 - devuelve top_k resultado, inicialmente top_k puede ser MAX_EMBEDDINGS
            3 - a partir de los chuncks se deben obtener los artÃ­culos
            3 - si entre los artÃ­culos recuperados NO estan todos los obtenidos por la RI binaria
                  ==> no se han recuperado todos los resultado: vuelve a 2 aumentando top_k
            4 - se utiliza la lista ordenada del kdtree para ordenar la lista "articles"
        """

        # Cargamos el modelo semántico y eliminamos repeticiones en la lista de artículos de entrada, si las hubiera:
        self.load_semantic_model() # Carga por defecto el modelo guardado en la constante SEMANTIC_MODEL, que en este caso es SBERT. 
        needed_articles = set(articles)

        # Calculamos los top_k resultados más semejantes:
        top_k = self.MAX_EMBEDDINGS
        top_k_results = self.model.query(query, top_k)

        # Mientras dentro de la lista de artículos recuperados (article_list) siga habiendo artículos de needed_articles, vamos aumentando el valor de top_k:
        article_list = []

        while self.minus_posting(list(needed_articles), article_list):
            top_k += self.MAX_EMBEDDINGS
            if top_k >= len(self.embeddings):
                break
            top_k_results = self.model.query(query, top_k)
            retrieved_indices = [index for _, index in top_k_results]

            # A partir de los índices de los chuncks recuperados, sacamos los artículos a los que pertenecen, y los añadimos a article_list:
            for idx in retrieved_indices:
                artid = self.chunck_index[idx]
                if artid not in article_list:
                    article_list.append(artid)

        # Creamos un diccionario con el orden en el que vamos a reordenar los artículos de la lista articles y lo aplicamos:
        order_dict = {artid: i for i, artid in enumerate(article_list)}
        articles.sort(key=lambda artid: order_dict.get(artid, float('inf')))

        return articles
    
    # FIN CAMBIO EN v1.2
        
     

    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################

    def already_in_index(self, article:Dict) -> bool:
        """

        Args:
            article (Dict): diccionario con la información de un artículo

        Returns:
            bool: True si el artículo ya está indexado, False en caso contrario
        """
        return article['url'] in self.urls


    # AÑADIDA UNA FUNCIÓN ADICIONAL
    def index_dir(self, root:str, **args):
        """

        Recorre recursivamente el directorio o fichero "root"
        NECESARIO PARA TODAS LAS VERSIONES

        Recorre recursivamente el directorio "root"  y indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """
        self.positional = args['positional']
        self.semantic = args['semantic']
        if self.semantic is True:      # MODIFICACIÓN: Si activamos el índice semántico, el posicional se activará también, para evitar poner "-P -S" al indexar y evitar problemas con el reranking.
            self.positional = True
            self.load_semantic_model()

        file_or_dir = Path(root)

        if file_or_dir.is_file():
            # is a file
            self.index_file(root)
        elif file_or_dir.is_dir():
            # is a directory
            for d, _, files in os.walk(root):
                for filename in sorted(files):
                    if filename.endswith('.json'):
                        fullname = os.path.join(d, filename)
                        self.index_file(fullname)
        else:
            print(f"ERROR:{root} is not a file nor directory!", file=sys.stderr)
            sys.exit(-1)

        # FUNCIÓN ADICIONAL: Si tenemos la búsqueda semántica activada, creamos el KD-Tree una vez indexados todos los documentos:
        if self.semantic is True:
            self.create_kdtree()

        
        
    def parse_article(self, raw_line:str) -> Dict[str, str]:
        """
        Crea un diccionario a partir de una linea que representa un artículo del crawler

        Args:
            raw_line: una linea del fichero generado por el crawler

        Returns:
            Dict[str, str]: claves: 'url', 'title', 'summary', 'all', 'section-name'
        """
        
        article = json.loads(raw_line)
        sec_names = []
        txt_secs = ''
        for sec in article['sections']:
            txt_secs += sec['name'] + '\n' + sec['text'] + '\n'
            txt_secs += '\n'.join(subsec['name'] + '\n' + subsec['text'] + '\n' for subsec in sec['subsections']) + '\n\n'
            sec_names.append(sec['name'])
            sec_names.extend(subsec['name'] for subsec in sec['subsections'])
        article.pop('sections') # no la necesitamos
        article['all'] = article['title'] + '\n\n' + article['summary'] + '\n\n' + txt_secs
        article['section-name'] = '\n'.join(sec_names)

        return article


    # IMPLEMENTADA POR NOSOTROS
    def index_file(self, filename:str):
        """

        Indexa el contenido de un fichero.

        input: "filename" es el nombre de un fichero generado por el Crawler cada línea es un objeto json
            con la información de un artículo de la Wikipedia

        NECESARIO PARA TODAS LAS VERSIONES

        dependiendo del valor de self.positional se debe ampliar el indexado

        """
        
        # Ya que calculamos los ids de los documentos de forma incremental, el id de este será la cantidad que llevábamos indexada:
        docId = len(self.docs)
        self.docs[docId] = filename

        # Procesamos un artículo por cada línea del documento:
        for index, line in enumerate(open(filename)):
            article = self.parse_article(line)

            # Añadimos su URL a la lista de enlaces:            
            if self.already_in_index(article): # MODIFICACIÓN: Antes repetíamos el contenido de already_in_index() aquí en vez de llamar a dicha función.
                continue
            
            self.urls.add(article['url'])

            # Calculamos su id del mismo modo que hicimos con el documento:
            artId = len(self.articles)

            # Creamos una entrada en el diccionario de artículos cuya clave será el id, y su valor será la siguiente tupla:
            # (índice de documento, posición relativa dentro del documento, título, URL)
            self.articles[artId] = (docId, index, article['title'], article['url'])
            
            # Sacamos el contenido de self.DEFAULT_FIELD y lo tokenizamos:
            content = article[self.DEFAULT_FIELD]
            contentToIndex = self.tokenize(content)

            # Si tenemos habilitada la búsqueda semántica, procesamos los tokens (chuncks) del artículo:
            if self.semantic:
                self.update_chuncks(content, artId)

            # Si tenemos habilitada la búsqueda posicional, guardamos en qué artículos y qué posiciones dentro de ellos se encuentra cada token:
            for pos, token in enumerate(contentToIndex):
                if self.positional:
                    self.index.setdefault(token,{})
                    self.index[token].setdefault(artId,[]).append(pos)
                # En caso contrario, simplemente guardamos el id del artículo dentro de la posting list del token:
                else:
                    posting_list = self.index.setdefault(token,[])
                    if artId not in posting_list:
                        posting_list.append(artId)



    def tokenize(self, text:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(' ', text.lower()).split()


    # IMPLEMENTADA POR NOSOTROS
    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Muestra estadisticas de los indices

        """ 

        # Mostramos las estadísticas de los índices según el patrón de las referencias:
        print(f"========================================\n"
              f"Number of indexed files: {len(self.docs)}\n"
              "----------------------------------------\n"
              f"Number of indexed articles: {len(self.articles)}\n"
              "----------------------------------------\n"
              "TOKENS:\n"
              f"\t# of tokens in 'all': {len(self.index)}\n"
              "----------------------------------------\n"
              f"Positional queries are {'NOT ' if not self.positional else ''}allowed\n"
              "========================================")



    #################################
    ###                           ###
    ###   PARTE 2: RECUPERACION   ###
    ###                           ###
    #################################

    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################


    # IMPLEMENTADA POR NOSOTROS
    def solve_query(self, query:str, prev:Dict={}):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una query.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen


        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva. No es necesario utilizarlo.


        return: posting list con el resultado de la query

        """

        # Si la query es nula o está vacía, devolvemos una lista vacía:
        if query is None or len(query) == 0:
            return []

        # Si estamos en el modo de búsqueda semántica sin reranking, llamamos a self.solve_semantic_query():
        if self.semantic and not self.semantic_ranking:
            return self.solve_semantic_query(query) # MODIFICACIÓN: Le pasamos la query sin limpiar.
        
        # COMENTARIO: Como ahora le pasamos a la búsqueda semántica la query tal cual (sin ponerla en minúsculas ni limpiarla),
        #             el test "Reunión informe y presentación" nos da mal, porque en la referencia subida a PoliformaT se calculó
        #             a partir de la query en minúscula (esto lo sabemos porque al pasarla a minúsculas no da bien el test).

        # Partimos de que la query devuelve todos los artículos:
        res = list(self.articles.keys())

        # MODIFICACIÓN DE ALT: Creamos una variable 'or_flag' que se active cuando hagamos un OR en la query:
        or_flag = False

        # Creamos una variable 'neg' que nos sirva para identificar cuándo se ha escrito un NOT en la query:
        neg = False

        query_original = query # MODIFICACIÓN: Nos guardamos la query original sin limpiar para pasársela al reranking.

        # MODIFICACIÓN: Si estamos en búsqueda posicional (o reranking con posicionales), limpiamos toda la query menos las comillas:
        if self.positional:
            query = re.compile(r"[^\w\"]+").sub(' ', query)
        else:
            query = re.compile(r"[^\w]+").sub(' ', query)

        # MODIFICACIÓN DE ALT: Sustituimos los OR por el símbolo '@', los AND por '%' y los NOT por '#' para que no se pongan en minúscula:
        query = query.replace('OR', '@').replace('AND', '%').replace('NOT', '#')
        query = query.lower() # MODIFICACIÓN: Ahora ponemos la query en minúscula después de haberla limpiado para posibilitar la sustitución de los OR, AND y NOT de la línea anterior.

        # Separamos la query en sub-queries, de modo que las frases entre comillas estén contenidas dentro de una misma sub-query.
        # Si no estamos en búsqueda posicional, no tendremos comillas, así que estas dos líneas no harán nada:
        positional_queries = iter(re.findall('"([^"]*)"', query))
        normal_queries = re.sub('"([^"]*)"', '$', query).split()

        # MODIFICACIÓN DE ALT: Cambiamos los símbolos '@', '%' y '#' por 'OR', 'AND' y 'NOT', respectivamente:
        symbol_map = {
            '@': 'OR',
            '%': 'AND',
            '#': 'NOT',
        }
        queries = [
            next(positional_queries) if x == '$' else symbol_map.get(x, x)
            for x in normal_queries
        ]
 
        # Analizamos cada subconsulta e intersecamos sus listas de artículos de resultado:
        for subq in queries:

            # MODIFICACIÓN DE ALT: Si nos encontramos con un OR, activamos el booleano 'or_flag' y pasamos a la siguiente iteración:
            if subq == 'OR':
                or_flag = True
                continue

            # MODIFICACIÓN DE ALT: Si nos encontramos con un AND, pasamos a la siguiente iteración:
            if subq == 'AND':
                continue

            # Si nos encontramos con un NOT, cambiamos el valor de neg y pasamos a la siguiente iteración.
            if subq == 'NOT':
                neg = not neg # MODIFICACIÓN: Ahora, si aparecen dos NOT seguidos, sus efectos se cancelan. 
                continue
            
            # Si nuestra subconsulta contiene más de un término, devolvemos los artículos que los contengan de manera consecutiva (get_positionals):
            if len(subq.split()) > 1:
                aux = self.get_positionals(subq)
            # Si no, devolvemos todos los artículos que contengan al término de la consulta:
            else:
                aux = sorted(self.get_posting(subq))

            # MODIFICACIÓN ALT: Si la 'or_flag' está activada, unimos la lista de artículos que llevábamos
            #                   con la resultante de la última subconsulta (o su complementaria si neg == True):
            if or_flag:
                res = self.or_posting(res, aux if not neg else self.reverse_posting(aux))
                or_flag = False
            else:
                # Si la 'or_flag' está desactivada, realizamos la intersección en vez de la unión:
                res = self.and_posting(res, aux if not neg else self.reverse_posting(aux))
            neg = False

        # Una vez resuelta la búsqueda binaria, si teníamos el ranking semántico habilitado, ordenamos los artículos de resultado según su similitud con la query sin limpiar:
        if self.semantic_ranking:
            res = self.semantic_reranking(query_original, res) # MODIFICACIÓN: Le pasamos la query sin limpiar.

        return res


    # IMPLEMENTADA POR NOSOTROS
    def get_posting(self, term:str):
        # Si el término está en 'self.index':
        if term in self.index:   
            val = self.index[term]
            if self.positional:
                return sorted(val.keys())
            return val
        
        # Si el término no está en 'self.index', pero tenemos el corrector:
        if self.use_spelling and self.speller is not None:
            # > Pedimos sugerencias:
            sugerencia = self.speller.suggest(term, flatten=True)

            # > Si ha encontrado alguna (lista no vacía), unimos sus postings lists:
            if sugerencia:
                acepta = []
                for w in sugerencia:
                    # if w in self.index: # Por si acaso.
                    if self.positional:
                        acepta = self.or_posting(acepta, sorted(self.index[w].keys()))
                    else:
                        acepta = self.or_posting(acepta, self.index[w])
                            
                return acepta

        return []


    # FUNCIÓN AUXILIAR CREADA POR NOSOTROS
    def pos_intersect(self, positions1, positions2):
        """

        Algoritmo de intersección para pares de términos consecutivos y ordenados.

        param:  "positions1": posting list posicional del término 1
                "positions2": posting list posicional del término 2
        
        """

        # Implementación del algoritmo de intersección posicional vista en el tema 2 de seminario:
        positions1 = list(positions1.items())
        positions2 = list(positions2.items())

        counter1 = 0
        counter2 = 0
        ans = {}

        while counter1 < len(positions1) and counter2 < len(positions2):

            artsID1 = positions1[counter1][0]
            artsID2 = positions2[counter2][0]
            
            # Si ambos artículos coinciden comparamos sus listas de posiciones:
            if artsID1 == artsID2:
                # Sumamos 1 a cada elemento de la lista de positions1, pues, para que las palabras aparezcan de forma consecutiva,
                # la posición del primer término debe ser una unidad menor que la del segundo:
                pos1 = [i + 1 for i in positions1[counter1][1]]

                # Llamamos al algoritmo de intersección con pos1 y la lista de posiciones del artículo 2:
                and12 = self.and_posting(pos1, positions2[counter2][1])

                # Si and_posting devuelve una lista no vacía, la guardamos en el diccionario ans, dándole como clave el ID del artículo que estábamos analizando:
                if and12:
                    ans[artsID1] = and12
                
                counter1 += 1
                counter2 += 1
            elif artsID1 < artsID2:
                counter1+=1
            else:
                counter2+=1
        
        return ans


    # IMPLEMENTADA POR NOSOTROS
    def get_positionals(self, terms:str):
        """
        Devuelve la posting list asociada a una secuencia de terminos consecutivos.
        NECESARIO PARA LAS BÚSQUEDAS POSICIONALES

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.

        return: posting list

        """

        # Tokenizamos la consulta:
        separatedTerms = self.tokenize(terms)
        postings = None

        for siguientes in separatedTerms:
            
            # Si no hay coincidencias posibles, devolvemos la lista vacía:
            if postings == {}:
                return []

            # En currentPostList guardamos los postings asociados a cada token. Si el token no estaba en el índice invertido, definimos currentPostList como un diccionario vacío: 
            currentPostList = self.index.get(siguientes, {})

            # Si hemos buscado un término que no aparecía en el índice invertido, sabemos que la consulta fallará:
            if not currentPostList:
                return []

            # Si estamos al inicio de la consulta, los postings válidos serán todos los de currentPostList:
            if postings is None:
                postings = currentPostList.copy()
            
            # En caso contrario, los postings válidos serán la intersección entre los que llevábamos de antes y currentPostList:
            else:
                postings = self.pos_intersect(postings, currentPostList)

        # Devolvemos la lista de postings válidos:
        return list(postings.keys())             


    # IMPLEMENTADA POR NOSOTROS
    def reverse_posting(self, p:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los artid exceptos los contenidos en p

        """

        # Implementación del algoritmo NOT vista en el tema 1 de seminario:
        counter1 = 0
        counter2 = 0

        artIds1 = sorted(self.articles.keys())
        artIds2 = sorted(set(p))
        aux = artIds1.copy()
        while counter1 < len(artIds1) and counter2 < len(artIds2):
            if artIds1[counter1] == artIds2[counter2]:
                aux.remove(artIds1[counter1])
                counter1+=1
                counter2+=1
            elif artIds1[counter1] < artIds2[counter2]:
                counter1+=1
            else:
                counter2+=1  

        return aux


    # IMPLEMENTADA POR NOSOTROS
    def and_posting(self, p1:list, p2:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular

        return: posting list con los artid incluidos en p1 y p2

        """

        # Implementación del algoritmo de intersección vista en el tema 1 de seminario:
        pRes = []
        i = 0
        j = 0

        while i < len(p1) and j < len(p2):
            if p1[i] > p2[j]:
                j += 1
            elif p1[i] < p2[j]:
                i += 1
            else:
                pRes.append(p1[i])
                i+=1
                j+=1       

        return pRes
    

    # IMPLEMENTADA POR NOSOTROS
    def or_posting(self, p1:list, p2:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el OR de dos posting lists de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular

        return: posting list con los artid incluidos en p1 y p2

        """
        pRes = []
        i = 0
        j = 0
        
        while i < len(p1) and j < len(p2):
            if p1[i] > p2[j]:
                pRes.append(p2[j])
                j += 1
            elif p1[i] < p2[j]:
                pRes.append(p1[i])
                i += 1
            else:
                pRes.append(p1[i])
                i += 1
                j += 1

        while i < len(p1):
            pRes.append(p1[i])
            i += 1
        
        while j < len(p2):
            pRes.append(p2[j])
            j += 1

        return pRes


    # IMPLEMENTADA POR NOSOTROS
    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se incluye por si es util, no es necesario utilizarla.

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 y no en p2

        """

        # Hacer la diferencia entre dos postings list es equivalente a hacer la intersección entre la primera y la complementaria de la segunda:
        return self.and_posting(p1, self.reverse_posting(p2))


    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################

    def solve_and_count(self, ql:List[str], verbose:bool=True) -> List:
        results = []
        for query in ql:
            if len(query) > 0 and query[0] != '#':
                r = self.solve_query(query)
                results.append(len(r))
                if verbose:
                    print(f'{query}\t{len(r)}')
            else:
                results.append(0)
                if verbose:
                    print(query)
        return results


    def solve_and_test(self, ql:List[str]) -> bool:
        errors = False
        for line in ql:
            if len(line) > 0 and line[0] != '#':
                query, ref = line.split('\t')
                reference = int(ref)
                result = len(self.solve_query(query))
                if reference == result:
                    print(f'{query}\t{result}')
                else:
                    print(f'>>>>{query}\t{reference} != {result}<<<<')
                    errors = True
            else:
                print(line)

        return not errors


    # IMPLEMENTADA POR NOSOTROS
    def solve_and_show(self, query:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra junto al numero de resultados

        param:  "query": query que se debe resolver.

        return: el numero de artículo recuperadas, para la opcion -T

        """

        # Resolvemos la query de entrada y mostramos por pantalla cuántos artículos hemos recibido de resultado:
        ans = self.solve_query(query)
        print(f"{query} \t{len(ans)} artículos encontrados:")

        # Si hemos recibido algún artículo, mostramos todos si está habilitada la opción "show_all" o los 10 primeros (todos si son 10 o menos) si no lo está:
        if len(ans) > 0:
            if self.show_all:
                num = len(ans)
            else:
                num = min(10,len(ans))

            # Para cada artículo, mostramos por pantalla su número de orden, su identificador, su título y su URL:
            for index, article in enumerate(ans[:num]):
                print(f"{index+1}. \t{article} \t{self.articles[article][2]} \t{self.articles[article][3]}")
