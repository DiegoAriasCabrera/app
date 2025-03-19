import os
import json
import random
import osmnx as ox
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import LineString
from typing import List, Dict, Tuple, Union, Optional, Any

# ============================================================================
# Excepciones Personalizadas
# ============================================================================
class ErrorShapefile(Exception):
    """
    Excepción personalizada para errores relacionados con los shapefiles.
    
    :param mensaje: Mensaje descriptivo del error ocurrido.
    :type mensaje: str
    """
    def __init__(self, mensaje: str) -> None:
        super().__init__(mensaje)


class ErrorRedVial(Exception):
    """
    Excepción personalizada para errores relacionados con la construcción o el manejo de la red vial
    (gráfica de nodos y aristas).
    
    :param mensaje: Mensaje descriptivo del error ocurrido.
    :type mensaje: str
    """
    def __init__(self, mensaje: str) -> None:
        super().__init__(mensaje)


# ============================================================================
# Clase Camion
# ============================================================================

class Camion:
    """
    Representa un camión con información de capacidad, factor de reserva, cantidad disponible y un nombre identificador.
    
    :param capacidad: Capacidad máxima del camión (peso máximo de carga).
    :type capacidad: float
    :param factor_reserva: Factor multiplicador para otorgar margen adicional a la capacidad.
    :type factor_reserva: float
    :param cantidad_camiones: Número de camiones de este tipo disponibles.
    :type cantidad_camiones: int
    :param nombre: Identificador o nombre del camión.
    :type nombre: str
    """

    def __init__(
        self,
        capacidad: float,
        factor_reserva: float,
        cantidad_camiones: int,
        nombre: str
    ) -> None:
        """
        Inicializa una nueva instancia de la clase Camion con los valores provistos.
        """
        # Se asigna la capacidad nominal sin considerar el factor de reserva.
        self.capacidad: float = capacidad

        # Se asigna el factor de reserva para considerar margen adicional.
        self.factor_reserva: float = factor_reserva

        # Se asigna el número de camiones disponibles de este tipo.
        self.cantidad_camiones: int = cantidad_camiones

        # Se asigna el identificador o nombre del camión.
        self.nombre: str = nombre

    def __repr__(self) -> str:
        """
        Regresa una representación en cadena del objeto Camion.
        
        :return: Representación del camión en el formato Camion(nombre, Capacidad=..., Cantidad=...).
        :rtype: str
        """
        return f"Camion({self.nombre}, Capacidad={self.capacidad}, Cantidad={self.cantidad_camiones})"


# ============================================================================
# Clase AgrupamientoAGEB
# ============================================================================

class AgrupamientoAGEB:
    """
    Gestiona el proceso de agrupamiento de nodos (AGEBs) mediante un algoritmo genético elitista.
    Además, facilita la asignación de camiones para cada grupo resultante y permite el posterior
    ajuste de la asignación si es necesario subdividir grupos con sobrepeso.
    
    :param ruta_shp: Ruta al archivo Shapefile que contiene la información de las AGEBs.
    :type ruta_shp: str
    :param tamano_poblacion: Tamaño de la población inicial del algoritmo genético.
    :type tamano_poblacion: int
    :param num_generaciones: Número de iteraciones (generaciones) que ejecutará el algoritmo genético.
    :type num_generaciones: int
    :param tasa_mutacion: Probabilidad de mutación para cada gen (nodo) en un individuo.
    :type tasa_mutacion: float
    :param factor_basura: Factor que multiplica la población de cada nodo para estimar el peso en basura.
    :type factor_basura: float
    :param camiones: Lista de objetos Camion disponibles para atender los grupos resultantes.
    :type camiones: Optional[List[Camion]]
    :param reconectar_grupos: Indica si se deben reconectar aristas entre los nodos que pertenezcan
                              al mismo grupo, una vez finalizado el agrupamiento.
    :type reconectar_grupos: bool
    :param semilla_random: Semilla para reproducibilidad de resultados aleatorios.
    :type semilla_random: Optional[int]
    """

    def __init__(
        self,
        ruta_shp: str,
        tamano_poblacion: int = 750,
        num_generaciones: int = 500,
        tasa_mutacion: float = 0.01,
        factor_basura: float = 1.071,
        camiones: Optional[List[Camion]] = None,
        reconectar_grupos: bool = True,
        semilla_random: Optional[int] = None
    ) -> None:
        """
        Configura la instancia cargando la información geográfica desde el Shapefile especificado.
        Posteriormente, genera la gráfica de nodos y, si existen camiones, ajusta los parámetros de
        límite de peso de acuerdo con sus capacidades.
        """
        # Si se proporciona una semilla aleatoria, se fija para garantizar reproducibilidad.
        if semilla_random is not None:
            random.seed(semilla_random)

        # Almacena la ruta del archivo Shapefile.
        self.ruta_shapefile: str = ruta_shp

        # Define el tamaño de la población inicial para el algoritmo genético.
        self.tamano_poblacion: int = tamano_poblacion

        # Número total de generaciones que se ejecutarán.
        self.num_generaciones: int = num_generaciones

        # Probabilidad de que ocurra mutación en cada gen (nodo).
        self.tasa_mutacion: float = tasa_mutacion

        # Factor para convertir la población de cada nodo en peso de basura.
        self.factor_basura: float = factor_basura

        # Lista de camiones disponibles para la asignación.
        self.camiones: Optional[List[Camion]] = camiones

        # Indica si se reconectarán los nodos dentro de cada grupo tras el agrupamiento.
        self.reconectar_grupos: bool = reconectar_grupos

        # Tamaño de la élite utilizada en la fase de selección elitista del algoritmo genético.
        self.tamano_elite: int = tamano_poblacion // 10

        # Se intenta leer el archivo Shapefile para generar el GeoDataFrame.
        try:
            self.gdf: gpd.GeoDataFrame = gpd.read_file(ruta_shp)
        except Exception as error:
            raise ErrorShapefile(f"Error al leer el shapefile: {error}")

        # Se construye la gráfica principal.
        self.gráfica: nx.Graph = nx.Graph()
        self._crear_gráfica()
    
    # -------------------------------------------------------------------------
    # Métodos privados
    # -------------------------------------------------------------------------

    def _obtener_limite_peso(self) -> float:
        """
        Determina el límite de peso a utilizar para agrupar. Si existen camiones disponibles, se toma el
        máximo de sus capacidades efectivas (capacidad * factor_reserva).
        
        :return: Límite de peso para el agrupamiento.
        :rtype: float
        :raises ValueError: Si no se especifica un límite de peso y no se proporcionan camiones.
        """
        if self.camiones and len(self.camiones) > 0:
            return max(camion.capacidad * camion.factor_reserva for camion in self.camiones)
        else:
            # Si no hay camiones y no hay límite de peso, se lanza excepción.
            raise ValueError("Debe especificarse un límite de peso o proporcionar camiones.")

    def _crear_gráfica(self) -> None:
        """
        Construye la gráfica a partir de la información contenida en el GeoDataFrame.
        Cada fila del GeoDataFrame se convierte en un nodo con atributos, y se añaden aristas
        entre nodos cuyas geometrías se tocan (adyacencia espacial).
        
        :raises ErrorShapefile: Si la columna 'pob' no se encuentra en el shapefile.
        :raises ErrorRedVial: Si ocurre algún error durante la construcción de la gráfica.
        """
        try:
            for indice, fila in self.gdf.iterrows():
                if 'pob' not in fila:
                    raise ErrorShapefile(
                        f"No se encontró la columna 'pob' en el shapefile para el nodo {indice}."
                    )
                # Se calcula el peso de los nodos tomando en cuenta a la población y a la generación per cápita de basura por persona.
                peso: float = fila['pob'] * self.factor_basura
                self.gráfica.add_node(
                    indice,
                    geometria=fila.geometry,
                    posicion=(fila.geometry.centroid.x, fila.geometry.centroid.y),
                    peso=peso
                )

            for i, fila1 in self.gdf.iterrows():
                for j, fila2 in self.gdf.iterrows():
                    if i < j and fila1.geometry.touches(fila2.geometry):
                        self.gráfica.add_edge(i, j)

        except Exception as error:
            raise ErrorRedVial(f"Error creando la gráfica: {error}")

    def _crear_poblacion_inicial(self, grafica: nx.Graph, limite: float) -> List[List[int]]:
        """
        Crea la población inicial para el algoritmo genético sobre la gráfica especificada.
        Cada individuo es una lista en la que cada posición indica el grupo asignado al nodo correspondiente.
        
        El método recorre los nodos y los agrupa de manera voraz mientras no se exceda el límite de peso,
        generando múltiples soluciones aleatorias que conformarán la población inicial.
        
        :param grafica: Gráfica sobre la cual se generarán los individuos.
        :type grafica: nx.Graph
        :param limite: Límite de peso que no deben exceder los grupos.
        :type limite: float
        :return: Lista de individuos representados como listas de asignaciones de grupos.
        :rtype: List[List[int]]
        """
        poblacion: List[List[int]] = []
        num_nodos: int = grafica.number_of_nodes()
        nodos_ordenados = list(grafica.nodes())

        for _ in range(self.tamano_poblacion):
            individuo: List[int] = [-1] * num_nodos
            nodos_visitados: set = set()
            grupo_actual: int = 0

            for nodo in nodos_ordenados:
                if nodo not in nodos_visitados:
                    peso_acumulado: float = grafica.nodes[nodo]['peso']
                    nodos_grupo: List[int] = [nodo]

                    while nodos_grupo:
                        nodo_actual: int = nodos_grupo.pop()
                        pos_nodo = nodos_ordenados.index(nodo_actual)

                        if individuo[pos_nodo] == -1:
                            individuo[pos_nodo] = grupo_actual
                            nodos_visitados.add(nodo_actual)

                            for vecino in grafica.neighbors(nodo_actual):
                                if vecino not in nodos_visitados:
                                    peso_vecino = grafica.nodes[vecino]['peso']
                                    if peso_acumulado + peso_vecino <= limite:
                                        nodos_grupo.append(vecino)
                                        peso_acumulado += peso_vecino

                    grupo_actual += 1

            poblacion.append(individuo)

        return poblacion

    def _calcular_fitness(
        self,
        grafica: nx.Graph,
        individuo: List[int],
        limite: float
    ) -> float:
        """
        Calcula el valor de fitness de un individuo sobre la gráfica indicada, basándose en criterios:
        
        1) Penaliza grupos cuyo peso excede el límite.
        2) Penaliza las aristas que conectan nodos de grupos distintos y premia las aristas internas.
        3) Si hay camiones definidos, penaliza fuertemente los grupos cuyo peso supere la capacidad
           de todos los camiones disponibles.
        
        :param grafica: Gráfica a la cual corresponde el individuo (puede ser la principal o una subgráfica).
        :type grafica: nx.Graph
        :param individuo: Lista que asigna un grupo a cada nodo (por su posición en la lista).
        :type individuo: List[int]
        :param limite: Límite de peso para los grupos del individuo.
        :type limite: float
        :return: Puntaje de aptitud obtenido para el individuo.
        :rtype: float
        """
        puntaje: float = 0.0
        pesos_por_grupo: Dict[int, float] = {}
        nodos_ordenados = list(grafica.nodes())

        for i, nodo in enumerate(nodos_ordenados):
            grupo: int = individuo[i]
            pesos_por_grupo[grupo] = pesos_por_grupo.get(grupo, 0.0) + grafica.nodes[nodo]['peso']

        for peso_grupo in pesos_por_grupo.values():
            if peso_grupo > limite:
                puntaje -= (peso_grupo - limite)

        for u, v in grafica.edges():
            idx_u = nodos_ordenados.index(u)
            idx_v = nodos_ordenados.index(v)
            if individuo[idx_u] == individuo[idx_v]:
                puntaje += 2.0
            else:
                puntaje -= 1.0

        if self.camiones:
            capacidades_efectivas: List[float] = []
            for camion in self.camiones:
                capacidades_efectivas.extend(
                    [camion.capacidad * camion.factor_reserva] * camion.cantidad_camiones
                )
            capacidades_efectivas.sort(reverse=True)

            for peso_grupo in pesos_por_grupo.values():
                if not any(cap >= peso_grupo for cap in capacidades_efectivas):
                    puntaje -= (peso_grupo * 5)

        return puntaje

    def _seleccion_elitista(
        self,
        poblacion: List[List[int]],
        puntajes: List[float],
        num_seleccionar: int
    ) -> List[List[int]]:
        """
        Realiza un proceso de selección elitista para elegir los individuos que pasarán a la siguiente generación.
        Se ordena la población de mayor a menor según su fitness. Luego:
        
        - Se conservan los mejores individuos (élite).
        - Se completa la selección tomando individuos de la mitad superior.
        
        :param poblacion: Conjunto de individuos de la generación actual.
        :type poblacion: List[List[int]]
        :param puntajes: Lista con los valores de fitness respectivos.
        :type puntajes: List[float]
        :param num_seleccionar: Número total de individuos a conservar.
        :type num_seleccionar: int
        :return: Lista de individuos seleccionados para la siguiente generación.
        :rtype: List[List[int]]
        """
        combinados = list(zip(poblacion, puntajes))
        combinados.sort(key=lambda par: par[1], reverse=True)

        elites = combinados[:self.tamano_elite]
        mitad_superior = combinados[:len(combinados)//2]
        faltan_por_seleccionar = num_seleccionar - self.tamano_elite
        seleccion_restante = mitad_superior[self.tamano_elite:self.tamano_elite + faltan_por_seleccionar]

        nueva_seleccion = elites + seleccion_restante
        return [ind for (ind, _) in nueva_seleccion]

    def _cruzar(self, padre1: List[int], padre2: List[int]) -> List[int]:
        """
        Realiza un cruce simple entre dos individuos, eligiendo un punto de cruce aleatorio y combinando
        la parte inicial del primer padre con la parte final del segundo.
        
        :param padre1: Primer individuo.
        :type padre1: List[int]
        :param padre2: Segundo individuo.
        :type padre2: List[int]
        :return: Hijo resultante del cruce.
        :rtype: List[int]
        """
        if len(padre1) <= 2:
            return padre1
        
        punto_cruce = random.randint(1, len(padre1) - 2)
        return padre1[:punto_cruce] + padre2[punto_cruce:]

    def _mutar(self, individuo: List[int]) -> List[int]:
        """
        Aplica la operación de mutación con una probabilidad igual a la tasa de mutación.
        Para cada gen, se puede reasignar el grupo a un nuevo valor (max_grupo + 1).
        
        :param individuo: Individuo a mutar.
        :type individuo: List[int]
        :return: Individuo luego de aplicar la mutación.
        :rtype: List[int]
        """
        max_grupo = max(individuo)
        for i in range(len(individuo)):
            if random.random() < self.tasa_mutacion:
                individuo[i] = max_grupo + 1
        return individuo

    def _evolucion_poblacion(
        self,
        grafica: nx.Graph,
        poblacion: List[List[int]],
        limite: float
    ) -> Tuple[List[List[int]], List[int], float]:
        """
        Ejecuta una generación de evolución genética (cálculo de fitness, selección, cruce y mutación) y
        devuelve la nueva población, el mejor individuo y su fitness.
        
        :param grafica: Gráfica sobre la cual se evalúan los individuos.
        :type grafica: nx.Graph
        :param poblacion: Población de individuos en la generación actual.
        :type poblacion: List[List[int]]
        :param limite: Límite de peso de los grupos.
        :type limite: float
        :return: Tupla con (nueva_población, mejor_individuo, mejor_fitness).
        :rtype: Tuple[List[List[int]], List[int], float]
        """
        puntajes = [self._calcular_fitness(grafica, ind, limite) for ind in poblacion]
        seleccionados = self._seleccion_elitista(poblacion, puntajes, num_seleccionar=len(poblacion) // 2)

        nueva_poblacion: List[List[int]] = []
        while len(nueva_poblacion) < len(poblacion):
            padre1, padre2 = random.sample(seleccionados, 2)
            hijo = self._cruzar(padre1, padre2)
            hijo_mutado = self._mutar(hijo)
            nueva_poblacion.append(hijo_mutado)

        indice_mejor = puntajes.index(max(puntajes))
        mejor_individuo = poblacion[indice_mejor]
        mejor_fitness = puntajes[indice_mejor]

        return nueva_poblacion, mejor_individuo, mejor_fitness

    def _ejecutar_agrupamiento_genetico(
        self,
        grafica: nx.Graph,
        limite: float,
        reconectar: bool
    ) -> Tuple[List[int], Dict[int, List[int]], Dict[int, float]]:
        """
        Ejecuta el algoritmo genético elitista sobre la gráfica indicada (puede ser la principal o una subgráfica).
        Devuelve la asignación de grupos (mejor individuo), así como diccionarios de grupos y pesos.
        
        :param grafica: Gráfica de la cual se generará el agrupamiento.
        :type grafica: nx.Graph
        :param limite: Límite de peso para los grupos.
        :type limite: float
        :param reconectar: Indica si se deben reconectar aristas dentro de cada grupo al finalizar.
        :type reconectar: bool
        :return: Tupla (mejor_individuo, grupos, pesos_grupos).
        :rtype: Tuple[List[int], Dict[int, List[int]], Dict[int, float]]
        :raises RuntimeError: Si no se logra encontrar un individuo válido tras las generaciones.
        """
        poblacion = self._crear_poblacion_inicial(grafica, limite)
        mejor_fitness = float('-inf')
        mejor_individuo: Optional[List[int]] = None

        for _ in tqdm(range(self.num_generaciones), desc="Generaciones", ncols=80):
            poblacion, ind, fitness = self._evolucion_poblacion(grafica, poblacion, limite)
            if fitness > mejor_fitness:
                mejor_fitness = fitness
                mejor_individuo = ind

        if mejor_individuo is None:
            raise RuntimeError("No se encontró un individuo válido.")

        grupos: Dict[int, List[int]] = {}
        pesos_grupos: Dict[int, float] = {}
        nodos_ordenados = list(grafica.nodes())

        for i, nodo in enumerate(nodos_ordenados):
            id_grupo = mejor_individuo[i]
            if id_grupo not in grupos:
                grupos[id_grupo] = []
                pesos_grupos[id_grupo] = 0.0
            grupos[id_grupo].append(nodo)
            pesos_grupos[id_grupo] += grafica.nodes[nodo]['peso']

        aristas_eliminar = []
        for u, v in grafica.edges():
            idx_u = nodos_ordenados.index(u)
            idx_v = nodos_ordenados.index(v)
            if mejor_individuo[idx_u] != mejor_individuo[idx_v]:
                aristas_eliminar.append((u, v))
        grafica.remove_edges_from(aristas_eliminar)

        if reconectar:
            for grupo, lista_nodos in grupos.items():
                for i in range(len(lista_nodos)):
                    for j in range(i + 1, len(lista_nodos)):
                        nodo_u = lista_nodos[i]
                        nodo_v = lista_nodos[j]
                        if not grafica.has_edge(nodo_u, nodo_v):
                            grafica.add_edge(nodo_u, nodo_v)

        return mejor_individuo, grupos, pesos_grupos

    # -------------------------------------------------------------------------
    # Métodos públicos
    # -------------------------------------------------------------------------

    def ejecutar_agrupamiento(self) -> Tuple[List[int], Dict[int, List[int]], Dict[int, float]]:
        """
        Ejecuta el algoritmo genético elitista sobre la gráfica principal para encontrar
        la mejor partición en grupos, respetando el límite de peso establecido.
        
        :return: Tupla con (mejor_individuo, grupos, pesos_grupos).
        :rtype: Tuple[List[int], Dict[int, List[int]], Dict[int, float]]
        """
        # Se obtiene el límite a partir de los camiones (o se genera excepción si no existen).
        limite_peso = self._obtener_limite_peso()
        return self._ejecutar_agrupamiento_genetico(
            grafica=self.gráfica,
            limite=limite_peso,
            reconectar=self.reconectar_grupos
        )

    def asignar_camiones(
        self,
        grupos: Dict[int, List[int]],
        pesos_grupos: Dict[int, float]
    ) -> Tuple[Dict[int, Optional[Camion]], List[Camion]]:
        """
        Asigna camiones a los grupos generados según la capacidad efectiva de cada camión (capacidad * factor_reserva).
        
        - Identifica "grupos prioritarios": aquellos de un solo nodo con peso que excede la capacidad mínima.
        - Asigna dichos grupos prioritarios con camiones de mayor capacidad.
        - Para el resto, se usa el camión más pequeño que sea suficiente.
        
        :param grupos: Diccionario que mapea el id de grupo a la lista de nodos que contiene.
        :type grupos: Dict[int, List[int]]
        :param pesos_grupos: Diccionario que mapea el id de grupo a su peso total.
        :type pesos_grupos: Dict[int, float]
        :return: Tupla (asignaciones, camiones_restantes).
                 - asignaciones: Diccionario que mapea cada grupo al camión asignado (o None).
                 - camiones_restantes: Lista de camiones no utilizados.
        :rtype: Tuple[Dict[int, Optional[Camion]], List[Camion]]
        """
        if not self.camiones:
            return {g: None for g in grupos}, []

        camiones_disponibles: List[Camion] = []
        for camion in self.camiones:
            for _ in range(camion.cantidad_camiones):
                camiones_disponibles.append(camion)

        if not camiones_disponibles:
            return {g: None for g in grupos}, []

        min_cap = min(c.capacidad * c.factor_reserva for c in camiones_disponibles)

        grupos_prioritarios: Dict[int, List[int]] = {}
        grupos_restantes: Dict[int, List[int]] = {}

        for g, nodos in grupos.items():
            if len(nodos) == 1 and pesos_grupos[g] > min_cap:
                grupos_prioritarios[g] = nodos
            else:
                grupos_restantes[g] = nodos

        asignaciones: Dict[int, Optional[Camion]] = {}

        camiones_disponibles.sort(key=lambda c: c.capacidad * c.factor_reserva, reverse=True)
        grupos_prioritarios_ordenados = sorted(
            grupos_prioritarios.items(),
            key=lambda item: pesos_grupos[item[0]],
            reverse=True
        )

        for g, _ in grupos_prioritarios_ordenados:
            peso_g = pesos_grupos[g]
            asignado = False
            for i, cam in enumerate(camiones_disponibles):
                if cam.capacidad * cam.factor_reserva >= peso_g:
                    asignaciones[g] = cam
                    del camiones_disponibles[i]
                    asignado = True
                    break
            if not asignado:
                asignaciones[g] = None

        camiones_disponibles.sort(key=lambda c: c.capacidad * c.factor_reserva)
        grupos_restantes_ordenados = sorted(
            grupos_restantes.items(),
            key=lambda item: pesos_grupos[item[0]],
            reverse=True
        )

        for g, _ in grupos_restantes_ordenados:
            peso_g = pesos_grupos[g]
            asignado = False
            for i, cam in enumerate(camiones_disponibles):
                if cam.capacidad * cam.factor_reserva >= peso_g:
                    asignaciones[g] = cam
                    del camiones_disponibles[i]
                    asignado = True
                    break
            if not asignado:
                asignaciones[g] = None

        return asignaciones, camiones_disponibles

    def ejecutar_agrupamiento_subgrupo(
        self,
        nodos: List[int],
        limite_peso_sub: float
    ) -> Tuple[List[int], Dict[int, List[int]], Dict[int, float]]:
        """
        Ejecuta el algoritmo genético elitista sobre la subgráfica inducida por los nodos indicados,
        con un límite de peso particular. Se utiliza comúnmente para subdividir grupos que exceden
        la capacidad de los camiones.
        
        :param nodos: Lista de nodos que conforman la subgráfica a agrupar.
        :type nodos: List[int]
        :param limite_peso_sub: Límite de peso para los grupos en la subgráfica.
        :type limite_peso_sub: float
        :return: Tupla (mejor_individuo, grupos_sub, pesos_sub) con la mejor solución encontrada.
        :rtype: Tuple[List[int], Dict[int, List[int]], Dict[int, float]]
        """
        subgrafica = self.gráfica.subgraph(nodos).copy()
        mejor_individuo, grupos_sub, pesos_sub = self._ejecutar_agrupamiento_genetico(
            grafica=subgrafica,
            limite=limite_peso_sub,
            reconectar=False
        )
        return mejor_individuo, grupos_sub, pesos_sub

    def post_procesar_asignacion(
        self,
        grupos: Dict[int, List[int]],
        pesos_grupos: Dict[int, float],
        asignaciones: Dict[int, Optional[Camion]],
        camiones_restantes: List[Camion]
    ) -> Tuple[Dict[int, List[int]], Dict[int, float], Dict[int, Optional[Camion]]]:
        """
        Refina la asignación de camiones tras el agrupamiento inicial de la siguiente forma:
        
        1) Subdivide los grupos que no cuentan con camión asignado (porque su peso excede la capacidad
           de todos los camiones disponibles) mediante un nuevo agrupamiento (subgráfica).
        2) Vuelve a asignar camiones a la estructura resultante.
        3) Intenta reagrupar los subgrupos que provengan del mismo grupo original, si existe un camión
           capaz de cubrirlos en conjunto.
        
        :param grupos: Diccionario original (grupo -> lista de nodos).
        :type grupos: Dict[int, List[int]]
        :param pesos_grupos: Diccionario (grupo -> peso total del grupo).
        :type pesos_grupos: Dict[int, float]
        :param asignaciones: Diccionario (grupo -> camión asignado o None) tras la asignación inicial.
        :type asignaciones: Dict[int, Optional[Camion]]
        :param camiones_restantes: Lista de camiones que no se utilizaron en la asignación previa.
        :type camiones_restantes: List[Camion]
        :return: Tupla (nuevos_grupos, nuevos_pesos, nuevas_asignaciones).
        :rtype: Tuple[Dict[int, List[int]], Dict[int, float], Dict[int, Optional[Camion]]]
        """
        if not camiones_restantes:
            print("No hay camiones restantes para intentar dividir ni reagrupar grupos.")
            return grupos, pesos_grupos, asignaciones

        capacidad_maxima = max(c.capacidad * c.factor_reserva for c in camiones_restantes)

        nuevos_grupos: Dict[int, List[int]] = {}
        nuevos_pesos: Dict[int, float] = {}
        nuevas_asignaciones: Dict[int, Optional[Camion]] = {}
        grupo_padre: Dict[int, int] = {}
        id_nuevo: int = 0

        for id_grupo, nodos_grupo in grupos.items():
            peso_g = pesos_grupos[id_grupo]
            camion_asignado = asignaciones[id_grupo]

            if camion_asignado is not None or peso_g <= capacidad_maxima:
                nuevos_grupos[id_nuevo] = nodos_grupo
                nuevos_pesos[id_nuevo] = peso_g
                grupo_padre[id_nuevo] = id_grupo
                id_nuevo += 1
            else:
                _, subgrupos, pesos_sub = self.ejecutar_agrupamiento_subgrupo(nodos_grupo, capacidad_maxima)

                for sub in subgrupos.values():
                    peso_sub = sum(self.gráfica.nodes[n]['peso'] for n in sub)
                    nuevos_grupos[id_nuevo] = sub
                    nuevos_pesos[id_nuevo] = peso_sub
                    grupo_padre[id_nuevo] = id_grupo
                    id_nuevo += 1

        nuevas_asignaciones, camiones_sobrantes = self.asignar_camiones(nuevos_grupos, nuevos_pesos)

        padre_a_subgrupos: Dict[int, List[int]] = {}
        for sub_id, p_id in grupo_padre.items():
            padre_a_subgrupos.setdefault(p_id, []).append(sub_id)

        ids_a_unir = []
        for p_id, lista_sub_ids in padre_a_subgrupos.items():
            if len(lista_sub_ids) <= 1:
                continue
            peso_total = sum(nuevos_pesos[sub_id] for sub_id in lista_sub_ids)
            if not camiones_sobrantes:
                continue
            max_cap_sobrante = max(c.capacidad * c.factor_reserva for c in camiones_sobrantes)
            if peso_total <= max_cap_sobrante:
                ids_a_unir.append((p_id, lista_sub_ids, peso_total))

        for (p_id, lista_sub_ids, peso_total) in ids_a_unir:
            todos_nodos = []
            for sid in lista_sub_ids:
                todos_nodos.extend(nuevos_grupos[sid])

            nuevo_id = min(lista_sub_ids)
            nuevos_grupos[nuevo_id] = todos_nodos
            nuevos_pesos[nuevo_id] = peso_total

            for sid in lista_sub_ids:
                if sid != nuevo_id:
                    if sid in nuevos_grupos:
                        del nuevos_grupos[sid]
                    if sid in nuevos_pesos:
                        del nuevos_pesos[sid]
                    if sid in nuevas_asignaciones:
                        del nuevas_asignaciones[sid]

            camion_elegido = None
            for i, c in enumerate(camiones_sobrantes):
                if c.capacidad * c.factor_reserva >= peso_total:
                    camion_elegido = c
                    camiones_sobrantes.pop(i)
                    break
            nuevas_asignaciones[nuevo_id] = camion_elegido

        return nuevos_grupos, nuevos_pesos, nuevas_asignaciones

    def graficar_con_camiones(
        self,
        grupos: Dict[int, List[int]],
        asignaciones: Dict[int, Optional[Camion]],
        output_path: Optional[str] = None
    ) -> None:
        """
        Genera una visualización que muestra los nodos y sus conexiones, coloreando cada grupo
        de acuerdo con el camión asignado. Los grupos sin camión se representan en rojo. 
        Si `output_path` está definido, se guarda la figura en esa ruta en lugar de mostrarla.
        
        :param grupos: Diccionario (grupo -> lista de nodos).
        :type grupos: Dict[int, List[int]]
        :param asignaciones: Diccionario (grupo -> camión o None).
        :type asignaciones: Dict[int, Optional[Camion]]
        """
        posiciones: Dict[Any, Tuple[float, float]] = {
            indice: (fila.geometry.centroid.x, fila.geometry.centroid.y)
            for indice, fila in self.gdf.iterrows()
        }

        colores_base: List[str] = [
            'blue', 'green', 'orange', 'purple',
            'brown', 'pink', 'gray', 'olive',
            'cyan', 'magenta'
        ]
        colores_por_camion: Dict[str, str] = {}
        nombres_asignados: List[str] = []

        if self.camiones:
            for indice, camion in enumerate(self.camiones):
                color_camion = colores_base[indice % len(colores_base)]
                if camion.nombre not in nombres_asignados:
                    colores_por_camion[camion.nombre] = color_camion
                    nombres_asignados.append(camion.nombre)

        mapa_colores: Dict[int, str] = {}
        for id_grupo, lista_nodos in grupos.items():
            camion = asignaciones.get(id_grupo, None)
            if camion is not None:
                color_grupo = colores_por_camion.get(camion.nombre, 'blue')
            else:
                color_grupo = 'red'
            for nodo in lista_nodos:
                mapa_colores[nodo] = color_grupo

        plt.figure(figsize=(16, 10))
        plt.title("Agrupamiento de AGEB con asignación de camiones")
        nx.draw(
            self.gráfica,
            posiciones,
            node_color=[mapa_colores[nodo] for nodo in self.gráfica.nodes()],
            with_labels=True,
            node_size=[(self.gráfica.nodes[nodo]['peso'] * self.factor_basura) / 50
                       for nodo in self.gráfica.nodes()],
            edge_color="gray"
        )

        leyenda: List[Patch] = []
        camiones_agregados: set = set()
        for id_grupo, camion in asignaciones.items():
            if camion is not None and camion.nombre not in camiones_agregados:
                color_camion = colores_por_camion[camion.nombre]
                leyenda.append(
                    Patch(facecolor=color_camion, edgecolor='black', label=camion.nombre)
                )
                camiones_agregados.add(camion.nombre)

        if any(camion is None for camion in asignaciones.values()):
            leyenda.append(Patch(facecolor='red', edgecolor='black', label='Sin asignación'))

        if leyenda:
            plt.legend(handles=leyenda, loc='best', title="Camiones")

        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def guardar_resultados(self, mejor_individuo: List[int], archivo: str) -> None:
        """
        Guarda la asignación de grupos del mejor individuo (solución resultante) en un archivo JSON.
        
        :param mejor_individuo: Lista que asigna a cada nodo el grupo correspondiente.
        :type mejor_individuo: List[int]
        :param archivo: Ruta o nombre del archivo en el que se almacenarán los resultados.
        :type archivo: str
        """
        try:
            with open(archivo, 'w', encoding='utf-8') as f:
                json.dump(mejor_individuo, f, ensure_ascii=False, indent=2)
            print(f"Resultados guardados en {archivo}")
        except Exception as error:
            print(f"Error al guardar los resultados: {error}")


# ============================================================================
# Clase ProcesadorCalles
# ============================================================================
class ProcesadorCalles:
    """
    Clase encargada de procesar y manipular la red vial de la Ciudad de México (CDMX).
    Proporciona funcionalidades para recortar la red vial según sectores específicos y corregir la
    conectividad de las redes recortadas asegurando su integridad.
    
    .. Nota::
        Este ejemplo asume que la red global está compuesta por los shapefiles
        `aristas_cdmx_shp` y `nodos_cdmx_shp`. No se requiere una red adicional.
    """

    def __init__(
        self,
        aristas_cdmx_shp: str,
        nodos_cdmx_shp: str,
        carpeta_sectores: str,
        carpeta_salida_calles: str,
        carpeta_salida_nodos: str,
        carpeta_salida_final: str = "./salida_final",
        crs_metrico: str = "EPSG:32614",
        crs_original: str = "EPSG:4326"
    ) -> None:
        """
        Constructor de la clase ProcesadorCalles.

        Inicializa las rutas a los shapefiles de la red vial completa de CDMX, así como las carpetas
        de entrada y salida para los procesos de recorte y corrección de la red.

        :param aristas_cdmx_shp: Ruta al shapefile que contiene las aristas (calles) de la red vial de CDMX.
        :type aristas_cdmx_shp: str
        :param nodos_cdmx_shp: Ruta al shapefile que contiene los nodos (intersecciones) de la red vial de CDMX.
        :type nodos_cdmx_shp: str
        :param carpeta_sectores: Directorio que contiene los shapefiles de sectores (polígonos) utilizados para recortar la red vial.
        :type carpeta_sectores: str
        :param carpeta_salida_calles: Directorio de salida donde se almacenarán las calles recortadas por sector.
        :type carpeta_salida_calles: str
        :param carpeta_salida_nodos: Directorio de salida donde se almacenarán los nodos recortados por sector.
        :type carpeta_salida_nodos: str
        :param carpeta_salida_final: Directorio de salida donde se guardarán los shapefiles finales corregidos.
                                      Por defecto, se establece en "./salida_final".
        :type carpeta_salida_final: str
        :param crs_metrico: Sistema de Referencia de Coordenadas (CRS) métrico utilizado para cálculos de longitud.
                            Por defecto, se usa "EPSG:32614".
        :type crs_metrico: str
        :param crs_original: Sistema de Referencia de Coordenadas (CRS) original para la conversión final.
                             Por defecto, se usa "EPSG:4326".
        :type crs_original: str
        """
        self.aristas_cdmx_shp = aristas_cdmx_shp
        self.nodos_cdmx_shp = nodos_cdmx_shp
        self.carpeta_sectores = carpeta_sectores
        self.carpeta_salida_calles = carpeta_salida_calles
        self.carpeta_salida_nodos = carpeta_salida_nodos
        self.carpeta_salida_final = carpeta_salida_final
        self.crs_metrico = crs_metrico
        self.crs_original = crs_original

        # Crea las carpetas de salida si no existen para almacenar los resultados
        os.makedirs(self.carpeta_salida_calles, exist_ok=True)
        os.makedirs(self.carpeta_salida_nodos, exist_ok=True)
        os.makedirs(self.carpeta_salida_final, exist_ok=True)

    # -------------------------------------------------------------------------
    # Métodos privados
    # -------------------------------------------------------------------------

    def _leer_shapefile_con_fallback(self, ruta: str) -> gpd.GeoDataFrame:
        """
        Lee un archivo shapefile utilizando codificación 'utf-8' y, en caso de fallo por error de decodificación,
        intenta leerlo con codificación 'cp1252'.

        Este método garantiza la robustez en la lectura de shapefiles con diferentes codificaciones de caracteres.

        :param ruta: Ruta al archivo shapefile que se desea leer.
        :type ruta: str
        :return: GeoDataFrame que contiene la información geográfica del shapefile leído.
        :rtype: gpd.GeoDataFrame
        :raises ErrorShapefile: Si falla al leer el shapefile con ambas codificaciones.
        """
        try:
            return gpd.read_file(ruta, encoding='utf-8')
        except UnicodeDecodeError:
            print(f"[Aviso] Error de decodificación con UTF-8 para: {ruta}. Probando con cp1252.")
            try:
                return gpd.read_file(ruta, encoding='cp1252')
            except Exception as e:
                raise ErrorShapefile(f"No se pudo leer el shapefile {ruta} con ninguna codificación: {e}")
        except Exception as e:
            raise ErrorShapefile(f"Error al leer el shapefile {ruta}: {e}")

    def _crear_grafica(
        self,
        gdf_nodos: gpd.GeoDataFrame,
        gdf_aristas: gpd.GeoDataFrame,
        col_nodo_id: str = "osmid",
        col_from: str = "from",
        col_to: str = "to",
        col_peso: str = "length"
    ) -> nx.DiGraph:
        """
        Crea una grafica dirigida a partir de GeoDataFrames que contienen información de nodos y aristas.

        Este método integra los nodos y aristas en una estructura de gráfica de NetworkX, asignando atributos
        relevantes como geometría y peso a cada elemento de la gráfica.

        :param gdf_nodos: GeoDataFrame que contiene la información de los nodos, incluyendo la columna identificadora.
        :type gdf_nodos: gpd.GeoDataFrame
        :param gdf_aristas: GeoDataFrame que contiene la información de las aristas, incluyendo columnas que indican los nodos de inicio y fin.
        :type gdf_aristas: gpd.GeoDataFrame
        :param col_nodo_id: Nombre de la columna en `gdf_nodos` que identifica de forma única cada nodo. Por defecto, "osmid".
        :type col_nodo_id: str
        :param col_from: Nombre de la columna en `gdf_aristas` que indica el nodo de inicio de la arista. Por defecto, "from".
        :type col_from: str
        :param col_to: Nombre de la columna en `gdf_aristas` que indica el nodo de fin de la arista. Por defecto, "to".
        :type col_to: str
        :param col_peso: Nombre de la columna utilizada como peso para las aristas, generalmente representando la longitud. Por defecto, "length".
        :type col_peso: str
        :return: Gráfica dirigida de NetworkX que representa la red vial.
        :rtype: nx.DiGraph
        :raises ErrorShapefile: Si faltan columnas requeridas en los shapefiles.
        """
        G = nx.DiGraph()

        try:
            # Agrega nodos a la gráfica con sus atributos correspondientes
            for _, nodo in gdf_nodos.iterrows():
                if col_nodo_id not in nodo:
                    raise ErrorShapefile(f"La columna '{col_nodo_id}' falta en el GeoDataFrame de nodos.")
                G.add_node(
                    nodo[col_nodo_id],
                    pos=(nodo.geometry.x, nodo.geometry.y),
                    geometry=nodo.geometry
                )

            # Agrega aristas a la gráfica con sus atributos correspondientes
            for _, arista in gdf_aristas.iterrows():
                if col_from not in arista or col_to not in arista:
                    raise ErrorShapefile(f"Las columnas '{col_from}' o '{col_to}' faltan en el GeoDataFrame de aristas.")
                peso = arista[col_peso] if col_peso in arista else None  # Asigna el peso si la columna existe
                G.add_edge(
                    arista[col_from],
                    arista[col_to],
                    geometry=arista.geometry,
                    weight=peso
                )
        except KeyError as e:
            raise ErrorShapefile(f"Columna faltante en el shapefile: {e}")
        except Exception as e:
            raise ErrorRedVial(f"Error al crear la gráfica: {e}")

        return G

    # -------------------------------------------------------------------------
    # Métodos públicos
    # -------------------------------------------------------------------------

    def recortar_red_vial_por_sectores(self) -> None:
        """
        Recorta la red vial de CDMX (aristas y nodos) según los sectores definidos en los shapefiles proporcionados.

        Este método realiza los siguientes pasos:
        1. Carga los shapefiles de aristas y nodos de la red vial completa de CDMX.
        2. Itera sobre cada shapefile de sector en la carpeta especificada:
           - Obtiene la intersección de las aristas con el polígono del sector.
           - Selecciona los nodos que están conectados a las aristas recortadas.
           - Guarda las aristas y nodos recortados en shapefiles separados dentro de las carpetas de salida designadas.
        3. Opcionalmente, genera una figura que muestra cada sector junto con su recorte correspondiente.

        Al finalizar, informa sobre la ubicación de los archivos guardados.

        :raises ErrorShapefile: Si ocurre un error al leer o escribir shapefiles.
        :raises ErrorRedVial: Si ocurre un error al procesar la red vial.
        """
        # Carga los shapefiles de aristas y nodos de la red vial completa de CDMX
        try:
            aristas_cdmx_gdf = self._leer_shapefile_con_fallback(self.aristas_cdmx_shp)
            nodos_cdmx_gdf = self._leer_shapefile_con_fallback(self.nodos_cdmx_shp)
        except ErrorShapefile as e:
            raise ErrorShapefile(f"Error al cargar shapefiles de CDMX: {e}")

        # Crea una figura para la visualización opcional de los recortes
        fig, ax = plt.subplots(figsize=(10, 10))

        # Lista todos los archivos shapefile de sectores en la carpeta especificada
        try:
            archivos_sectores = [
                f for f in os.listdir(self.carpeta_sectores) if f.endswith(".shp")
            ]
        except Exception as e:
            raise ErrorShapefile(f"Error al listar archivos en {self.carpeta_sectores}: {e}")

        ancho_barra = 80  # Define el ancho de la barra de progreso

        # Itera sobre cada archivo de sector para realizar el recorte
        for archivo in tqdm(archivos_sectores, desc="Recortando sectores", ncols=ancho_barra):
            ruta_sector = os.path.join(self.carpeta_sectores, archivo)
            try:
                sector_gdf = self._leer_shapefile_con_fallback(ruta_sector)
            except ErrorShapefile as e:
                print(f"[Error] No se pudo leer el sector {ruta_sector}: {e}")
                continue

            try:
                # Une múltiples polígonos en un solo polígono si es necesario
                sector_poligono = sector_gdf.unary_union

                # Filtra las aristas que intersectan con el polígono del sector
                calles_recortadas = aristas_cdmx_gdf[aristas_cdmx_gdf.intersects(sector_poligono)].copy()

                # Identifica los nodos que están conectados a las aristas recortadas
                nodos_ids = set()
                for _, calle in calles_recortadas.iterrows():
                    if 'u' not in calle or 'v' not in calle:
                        raise ErrorShapefile("Las columnas 'u' y/o 'v' faltan en el shapefile de aristas.")
                    nodos_ids.add(calle["u"])  # Nodo de origen
                    nodos_ids.add(calle["v"])  # Nodo de destino

                # Filtra los nodos que están presentes en las aristas recortadas
                nodos_recortados = nodos_cdmx_gdf[nodos_cdmx_gdf["osmid"].isin(nodos_ids)].copy()

                # Define las rutas de salida para las calles y nodos recortados
                nombre_base = os.path.splitext(archivo)[0]
                salida_calles = os.path.join(self.carpeta_salida_calles, f"{nombre_base}_calles.shp")
                salida_nodos = os.path.join(self.carpeta_salida_nodos, f"{nombre_base}_nodos.shp")

                # Guarda las calles y nodos recortados en los shapefiles correspondientes
                calles_recortadas.to_file(salida_calles, encoding="utf-8")
                nodos_recortados.to_file(salida_nodos, encoding="utf-8")

                # Opcionalmente, grafica el sector y sus recortes
                sector_gdf.plot(ax=ax, edgecolor="red", facecolor="none")
                calles_recortadas.plot(ax=ax, color="blue", linewidth=0.8)
                nodos_recortados.plot(ax=ax, color="green", markersize=8)
            except ErrorShapefile as e:
                print(f"[Error] Durante el procesamiento del sector {archivo}: {e}")
                continue
            except Exception as e:
                raise ErrorRedVial(f"Error durante el recorte del sector {archivo}: {e}")

        # Configura la visualización de la figura
        plt.yticks([])
        plt.xticks([])
        plt.show()

        # Informa al usuario sobre la finalización del proceso y la ubicación de los archivos guardados
        print("Proceso de recorte completado. Archivos guardados en:")
        print(f" - {self.carpeta_salida_calles}")
        print(f" - {self.carpeta_salida_nodos}")

    def corregir_conectividad(
            self, 
            nodos_recortados_shp: str, 
            aristas_recortadas_shp: str
    ) -> None:
        """
        Corrige la conectividad de un sector recortado de la red vial de CDMX utilizando la red completa como referencia.

        Este método garantiza que la red recortada sea totalmente conectada, añadiendo las aristas necesarias
        basándose en los caminos existentes en la red global de CDMX.

        Los pasos realizados son:
        1. Carga los shapefiles de nodos y aristas recortados del sector específico, así como los shapefiles
           de la red vial completa de CDMX.
        2. Convierte todas las geometrías a un Sistema de Referencia de Coordenadas (CRS) métrico para facilitar
           el cálculo preciso de longitudes.
        3. Crea gráficas dirigidas tanto para la red local (recortada) como para la red global.
        4. Identifica pares de nodos en la red local que no están conectados por ningún camino.
        5. Para cada par de nodos sin conexión, busca el camino más corto en la red global y agrega las
           aristas correspondientes a la red local para restablecer la conectividad.
        6. Convierte las geometrías de vuelta al CRS original y guarda los shapefiles finales corregidos.

        :param nodos_recortados_shp: Ruta al shapefile que contiene los nodos recortados del sector a corregir.
        :type nodos_recortados_shp: str
        :param aristas_recortadas_shp: Ruta al shapefile que contiene las aristas recortadas del sector a corregir.
        :type aristas_recortadas_shp: str
        :raises ErrorShapefile: Si ocurre un error al leer o escribir shapefiles.
        :raises ErrorRedVial: Si ocurre un error al procesar la red vial.
        """
        # (1) Cargar shapefiles locales (recortados) y completos (globales) de CDMX
        try:
            nodos_local_gdf = self._leer_shapefile_con_fallback(nodos_recortados_shp)
            aristas_local_gdf = self._leer_shapefile_con_fallback(aristas_recortadas_shp)
            nodos_cdmx_gdf = self._leer_shapefile_con_fallback(self.nodos_cdmx_shp)
            aristas_cdmx_gdf = self._leer_shapefile_con_fallback(self.aristas_cdmx_shp)
        except ErrorShapefile as e:
            raise ErrorShapefile(f"Error al cargar shapefiles: {e}")

        # (2) Convertir todas las geometrías al CRS métrico para cálculos precisos de longitud
        try:
            nodos_local_gdf = nodos_local_gdf.to_crs(self.crs_metrico)
            aristas_local_gdf = aristas_local_gdf.to_crs(self.crs_metrico)
            nodos_cdmx_gdf = nodos_cdmx_gdf.to_crs(self.crs_metrico)
            aristas_cdmx_gdf = aristas_cdmx_gdf.to_crs(self.crs_metrico)
        except Exception as e:
            raise ErrorRedVial(f"Error al convertir CRS: {e}")

        # (3) Calcular longitudes de las aristas y crear gráficas dirigidas para las redes local y global
        try:
            aristas_local_gdf["length"] = aristas_local_gdf.geometry.length
            aristas_cdmx_gdf["length"] = aristas_cdmx_gdf.geometry.length

            G_local = self._crear_grafica(
                gdf_nodos=nodos_local_gdf,
                gdf_aristas=aristas_local_gdf,
                col_nodo_id="osmid",
                col_from="from",
                col_to="to",
                col_peso="length"
            )

            G_cdmx = self._crear_grafica(
                gdf_nodos=nodos_cdmx_gdf,
                gdf_aristas=aristas_cdmx_gdf,
                col_nodo_id="osmid",
                col_from="from",
                col_to="to",
                col_peso="length"
            )
        except ErrorShapefile as e:
            raise ErrorShapefile(f"Error al crear gráficas dirigidas: {e}")
        except Exception as e:
            raise ErrorRedVial(f"Error al procesar las aristas: {e}")

        # (4) Identificar pares de nodos en la red local que no están conectados por ningún camino
        nodos_local = list(G_local.nodes)
        parejas_no_conexas = []
        try:
            for n_orig in nodos_local:
                for n_dest in nodos_local:
                    if n_orig != n_dest and not nx.has_path(G_local, n_orig, n_dest):
                        parejas_no_conexas.append((n_orig, n_dest))
        except Exception as e:
            raise ErrorRedVial(f"Error al identificar parejas no conectadas: {e}")

        print(f"Se encontraron {len(parejas_no_conexas)} pares de nodos sin ruta en la red local.")

        # (5) Para cada par sin conexión, buscar el camino más corto en la red global y agregar las aristas faltantes a la red local
        ancho_barra = 80
        sin_ruta_global = []  # Lista para almacenar pares de nodos que no tienen ruta ni en la red global
        try:
            with tqdm(total=len(parejas_no_conexas), desc="Corrigiendo conectividad", ncols=ancho_barra) as pbar:
                for (n_from, n_to) in parejas_no_conexas:
                    try:
                        ruta = nx.shortest_path(G_cdmx, source=n_from, target=n_to, weight='weight')
                    except nx.NetworkXNoPath:
                        sin_ruta_global.append((n_from, n_to))
                        pbar.update(1)
                        continue

                    aristas_nuevas = []
                    long_total = 0.0
                    oneway_vals = []

                    # Recorrer la ruta obtenida para unir las geometrías de las aristas
                    for i in range(len(ruta) - 1):
                        sub_from = ruta[i]
                        sub_to = ruta[i + 1]

                        filtro = ((aristas_cdmx_gdf["from"] == sub_from) & (aristas_cdmx_gdf["to"] == sub_to))
                        if not filtro.any():
                            # En caso de que la arista no exista en la red global (lo cual es raro)
                            continue

                        geom_arista = aristas_cdmx_gdf.loc[filtro, "geometry"].iloc[0]
                        if len(geom_arista.coords) <= 2:
                            # Ignora aristas demasiado simplificadas
                            continue

                        aristas_nuevas.append(geom_arista)
                        long_total += aristas_cdmx_gdf.loc[filtro, "length"].values[0]
                        oneway_vals.append(aristas_cdmx_gdf.loc[filtro, "oneway"].values[0])

                    if aristas_nuevas:
                        # Combina las geometrías de las aristas nuevas en un solo LineString
                        linea_combinada = LineString([pt for g in aristas_nuevas for pt in g.coords])

                        # Determina el valor final de "oneway" si es consistente a lo largo de la ruta
                        if len(set(oneway_vals)) == 1:
                            oneway_final = str(oneway_vals[0])
                        else:
                            oneway_final = ";".join(map(str, oneway_vals))

                        # Crea un nuevo registro de arista con la geometría combinada y los atributos correspondientes
                        nueva_arista = {
                            "from": n_from,
                            "to": n_to,
                            "geometry": linea_combinada,
                            "length": long_total,
                            "oneway": oneway_final
                        }
                        nueva_arista_gdf = gpd.GeoDataFrame(
                            [nueva_arista],
                            geometry="geometry",
                            crs=aristas_local_gdf.crs
                        )
                        # Añade la nueva arista al GeoDataFrame de aristas locales
                        aristas_local_gdf = pd.concat([aristas_local_gdf, nueva_arista_gdf], ignore_index=True)

                    pbar.update(1)
        except Exception as e:
            raise ErrorRedVial(f"Error durante la corrección de conectividad: {e}")

        # (6) Convertir las geometrías de vuelta al CRS original para mantener la consistencia geográfica
        try:
            nodos_local_gdf = nodos_local_gdf.to_crs(self.crs_original)
            aristas_local_gdf = aristas_local_gdf.to_crs(self.crs_original)
        except Exception as e:
            raise ErrorRedVial(f"Error al convertir CRS de vuelta al original: {e}")

        # (7) Guardar shapefiles resultantes utilizando el nombre base del archivo original
        try:
            # Extrae el nombre base del archivo de nodos recortados para construir el nombre de salida
            base_nodos = os.path.splitext(os.path.basename(nodos_recortados_shp))[0]
            # Define la ruta de salida para los nodos finales corregidos
            nombre_nodos_salida = os.path.join(
                self.carpeta_salida_final,
                base_nodos + "_finales.shp"
            )

            # Extrae el nombre base del archivo de aristas recortadas para construir el nombre de salida
            base_aristas = os.path.splitext(os.path.basename(aristas_recortadas_shp))[0]
            # Define la ruta de salida para las aristas finales corregidas
            nombre_aristas_salida = os.path.join(
                self.carpeta_salida_final,
                base_aristas + "_finales.shp"
            )

            # Guarda los shapefiles finales corregidos en las rutas de salida definidas
            nodos_local_gdf.to_file(nombre_nodos_salida)
            aristas_local_gdf.to_file(nombre_aristas_salida)
        except Exception as e:
            raise ErrorShapefile(f"Error al guardar shapefiles finales: {e}")

        # Informa al usuario sobre la finalización del proceso y la ubicación de los archivos guardados
        print("\nCorrección de conectividad completada.")
        print(f"Shapefiles guardados en: {self.carpeta_salida_final}")
        print(f" - {os.path.basename(nombre_nodos_salida)}")
        print(f" - {os.path.basename(nombre_aristas_salida)}")
        if sin_ruta_global:
            print("Los siguientes pares de nodos no tienen ruta ni en la red global:")
            print(sin_ruta_global)


# ============================================================================
# Clase ResolverTSP
# ============================================================================
class ResolverTSP:
    """
    Esta clase se encarga de resolver el Problema del Viajante (TSP) utilizando un algoritmo genético con elitismo.
    
    El algoritmo opera sobre una red vial representada por un MultiDiGraph de NetworkX, asignando pesos a las aristas 
    según la distancia (longitud) y asegurando la conectividad de la red. Busca la ruta que minimice la distancia total 
    recorrida para visitar cada nodo seleccionado exactamente una vez.
    """

    def __init__(
            self,
            ruta_nodos: str,
            ruta_aristas: str,
            grafica: nx.MultiDiGraph = None,
            tamano_poblacion: int = 750,
            generaciones: int = 1500,
            tasa_mutacion: float = 0.01,
            proporcion_elite: float = 0.1
    ):
        """
        Inicializa una instancia de ResolverTSP con los parámetros necesarios para la ejecución del algoritmo genético.

        :param ruta_nodos: Ruta al archivo shapefile que contiene los nodos de la red vial.
        :type ruta_nodos: str
        :param ruta_aristas: Ruta al archivo shapefile que contiene las aristas de la red vial.
        :type ruta_aristas: str
        :param grafica: Gráfica dirigida múltiple que representa la red vial. 
                        Si se proporciona, se utilizará en lugar de crear uno nuevo.
        :type grafica: nx.MultiDiGraph, optional
        :param tamano_poblacion: Número de individuos en la población inicial del algoritmo genético.
        :type tamano_poblacion: int, default=750
        :param generaciones: Número de iteraciones o generaciones que el algoritmo genético ejecutará para 
                             evolucionar la población.
        :type generaciones: int, default=1500
        :param tasa_mutacion: Probabilidad de que un individuo sufra una mutación durante el proceso de evolución.
        :type tasa_mutacion: float, default=0.01
        :param proporcion_elite: Proporción (entre 0 y 1) de individuos de élite que se preservarán directamente 
                                 en la siguiente generación.
        :type proporcion_elite: float, default=0.1
        """
        self.ruta_nodos = ruta_nodos
        self.ruta_aristas = ruta_aristas
        # Emplea la gráfica proporcionada o crea una nueva a partir de los shapefiles
        self.grafica = grafica if grafica is not None else self._crear_red_vial()  
        self.tamano_poblacion = tamano_poblacion
        self.generaciones = generaciones
        self.tasa_mutacion = tasa_mutacion
        self.proporcion_elite = proporcion_elite

    # -------------------------------------------------------------------------
    # Métodos privados
    # -------------------------------------------------------------------------

    def _crear_red_vial(self) -> nx.MultiDiGraph:
        """
        Crea la red vial a partir de los archivos shapefile de nodos y aristas.

        Utiliza GeoPandas para leer los shapefiles y luego integra la información en una gráfica 
        dirigido múltiple de NetworkX mediante OSMNX. Asigna atributos relevantes, como la geometría 
        y la longitud, a cada arista.

        :return: Gráfica dirigida múltiple que representa la red vial.
        :rtype: nx.MultiDiGraph
        :raises ErrorShapefile: Si faltan columnas requeridas en los shapefiles.
        :raises ErrorRedVial: Si ocurre un error al crear la gráfica.
        """
        try:
            gdf_nodes = gpd.read_file(self.ruta_nodos)
            gdf_edges = gpd.read_file(self.ruta_aristas)
        except Exception as e:
            raise ErrorShapefile(f"Error al leer shapefiles de nodos o aristas: {e}")

        # Verifica que la columna 'osmid' exista en los nodos para identificar de manera única cada nodo
        if 'osmid' not in gdf_nodes.columns:
            raise ErrorShapefile("El archivo de nodos debe tener una columna 'osmid' con identificadores únicos.")

        # Establece 'osmid' como índice para facilitar la asociación de datos
        gdf_nodes.set_index('osmid', inplace=True)

        # Verifica que las columnas 'from' y 'to' existan en las aristas para definir conexiones entre nodos
        if 'from' not in gdf_edges.columns or 'to' not in gdf_edges.columns:
            raise ErrorShapefile("El archivo de aristas debe tener columnas 'from' y 'to' con los identificadores de nodos.")

        # Asegura que cada arista tenga una clave única para evitar duplicados en la gráfica
        if 'key' not in gdf_edges.columns:
            gdf_edges['key'] = 0
        gdf_edges.set_index(['from', 'to', 'key'], inplace=True)

        try:
            # Crea la gráfica a partir de los GeoDataFrames de nodos y aristas utilizando OSMNX
            grafica = ox.graph_from_gdfs(gdf_nodes, gdf_edges)
        except Exception as e:
            raise ErrorRedVial(f"Error al crear la gráfica desde los GeoDataFrames: {e}")

        return grafica

    def _asignar_pesos_aristas(self) -> None:
        """
        Asigna pesos a las aristas de la gráfica basándose en la longitud de cada arista.

        Recorre todas las aristas de la gráfica y actualiza el atributo 'weight' con la longitud 
        de la arista, lo cual permite que el algoritmo genético utilice estos pesos para 
        calcular las distancias en la resolución del TSP.
        """
        for _, _, data in self.grafica.edges(data=True):
            data['weight'] = data.get('length', 1.0)

    def _verificar_conectividad(self) -> None:
        """
        Verifica si la gráfica es fuertemente conectada. Si no lo es, agrega aristas con pesos altos 
        para unir componentes desconectados, garantizando que exista al menos un camino entre cualquier 
        par de nodos.

        :raises ErrorRedVial: Si ocurre un error durante la verificación o modificación de la conectividad.
        """
        try:
            if not nx.is_strongly_connected(self.grafica):
                componentes = list(nx.strongly_connected_components(self.grafica))
                for i in range(len(componentes) - 1):
                    nodo1 = list(componentes[i])[0]
                    nodo2 = list(componentes[i + 1])[0]
                    # Añade una arista con un peso muy elevado para forzar la conexión entre componentes
                    self.grafica.add_edge(nodo1, nodo2, length=1e5, weight=1e5)
        except Exception as e:
            raise ErrorRedVial(f"Error al verificar o corregir la conectividad de la gráfica: {e}")

    def _calcular_matriz_distancias(
            self, 
            nodos: List[int]
    ) -> Dict[int, Dict[int, Union[float, int]]]:
        """
        Calcula la matriz de distancias entre todos los pares de nodos utilizando la longitud de 
        los caminos más cortos (basados en Dijkstra).

        :param nodos: Lista de nodos para los cuales se calcularán las distancias entre todos los pares.
        :type nodos: List[int]
        :return: Diccionario anidado que representa la matriz de distancias, donde las claves 
                 son nodos y los valores son diccionarios de nodos con sus respectivas distancias.
        :rtype: Dict[int, Dict[int, Union[float, int]]]
        :raises ErrorRedVial: Si ocurre un error durante el cálculo de las distancias.
        """
        matriz_distancias: Dict[int, Dict[int, Union[float, int]]] = {}
        try:
            for nodo_i in nodos:
                matriz_distancias[nodo_i] = {}
                # Calcula las distancias desde nodo_i a todos los demás nodos utilizando Dijkstra
                longitudes = nx.single_source_dijkstra_path_length(
                    self.grafica,
                    nodo_i,
                    weight='length'
                )
                for nodo_j in nodos:
                    if nodo_i != nodo_j:
                        distancia = longitudes.get(nodo_j, np.inf)
                        matriz_distancias[nodo_i][nodo_j] = distancia
                    else:
                        matriz_distancias[nodo_i][nodo_j] = 0.0
        except Exception as e:
            raise ErrorRedVial(f"Error al calcular la matriz de distancias: {e}")
        return matriz_distancias

    def _crear_poblacion(self, nodos: List[int]) -> List[List[int]]:
        """
        Crea la población inicial de rutas posibles para el algoritmo genético del TSP.

        Cada individuo en la población es una permutación aleatoria de los nodos, representando 
        una posible ruta que visita cada nodo una sola vez.

        :param nodos: Lista de nodos que conforman la ruta a optimizar.
        :type nodos: List[int]
        :return: Lista de individuos, donde cada individuo es una lista de nodos en un orden específico.
        :rtype: List[List[int]]
        """
        return [random.sample(nodos, len(nodos)) for _ in range(self.tamano_poblacion)]

    def _calcular_fitness(
            self, 
            individuo: List[int], 
            matriz_distancias: Dict[int, Dict[int, Union[float, int]]]
    ) -> float:
        """
        Calcula la aptitud (fitness) de un individuo basado en la distancia total de la ruta 
        que representa.

        La aptitud se define como la suma de las distancias entre nodos consecutivos en la ruta. 
        Cuanto menor sea la distancia total, mayor será la aptitud.

        :param individuo: Ruta representada como una lista de nodos en orden.
        :type individuo: List[int]
        :param matriz_distancias: Matriz de distancias entre nodos, previamente calculada.
        :type matriz_distancias: Dict[int, Dict[int, Union[float, int]]]
        :return: Suma total de las distancias entre nodos consecutivos en la ruta.
        :rtype: float
        """
        return sum(
            matriz_distancias[individuo[i]][individuo[i + 1]]
            for i in range(len(individuo) - 1)
        )

    def _seleccion(
            self, 
            poblacion: List[List[int]], 
            puntuaciones_aptitud: List[float], 
            cantidad_a_seleccionar: int
    ) -> List[List[int]]:
        """
        Selecciona a los mejores individuos de la población según su aptitud, ordenando de menor 
        a mayor la distancia total (menor distancia es mejor).

        :param poblacion: Lista de individuos presentes en la población actual.
        :type poblacion: List[List[int]]
        :param puntuaciones_aptitud: Lista de puntuaciones de aptitud (distancias totales) para cada individuo.
        :type puntuaciones_aptitud: List[float]
        :param cantidad_a_seleccionar: Número de individuos que serán seleccionados para la siguiente generación.
        :type cantidad_a_seleccionar: int
        :return: Lista de individuos seleccionados basándose en la aptitud.
        :rtype: List[List[int]]
        """
        # Combina la población con sus respectivas puntuaciones de aptitud
        combinacion = list(zip(poblacion, puntuaciones_aptitud))
        # Ordena la combinación de manera ascendente según el fitness (menor distancia primero)
        combinacion_ordenada = sorted(combinacion, key=lambda x: x[1])
        # Extrae los individuos con los mejores fitness
        seleccionados = [ind for ind, _ in combinacion_ordenada[:cantidad_a_seleccionar]]
        return seleccionados

    def _cruce(
            self, 
            padre1: List[int], 
            padre2: List[int]
    ) -> List[int]:
        """
        Realiza el cruce (crossover) entre dos individuos padre para generar un nuevo individuo 
        hijo que sea válido en el contexto del TSP (una permutación sin repeticiones).

        Utiliza un método de intercambio de segmentos para combinar partes de ambos padres, 
        asegurando que el hijo resultante sea una permutación válida de los nodos.

        :param padre1: Primer padre, representado como una lista de nodos en orden.
        :type padre1: List[int]
        :param padre2: Segundo padre, representado como una lista de nodos en orden.
        :type padre2: List[int]
        :return: Hijo resultante del cruce entre los dos padres.
        :rtype: List[int]
        """
        tamano = len(padre1)
        inicio, fin = sorted(random.sample(range(tamano), 2))  # Selecciona dos puntos de corte aleatorios
        hijo = [None] * tamano

        # Copia una subsección del primer padre al hijo
        hijo[inicio:fin] = padre1[inicio:fin]

        # Completa los nodos restantes en el orden en que aparecen en el segundo padre
        indice = fin
        for nodo in padre2:
            if nodo not in hijo:
                if indice >= tamano:
                    indice = 0
                hijo[indice] = nodo
                indice += 1

        return hijo

    def _mutar(
            self, 
            individuo: List[int]
    ) -> List[int]:
        """
        Aplica mutaciones al individuo intercambiando pares de nodos con cierta probabilidad, 
        introduciendo variabilidad genética en la población.

        :param individuo: Ruta representada como una lista de nodos en orden.
        :type individuo: List[int]
        :return: Individuo mutado con posibles cambios en el orden de los nodos.
        :rtype: List[int]
        """
        for i in range(len(individuo)):
            if random.random() < self.tasa_mutacion:
                j = random.randint(0, len(individuo) - 1)
                # Intercambia el nodo actual con otro nodo aleatorio
                individuo[i], individuo[j] = individuo[j], individuo[i]
        return individuo

    def _evolucionar_poblacion(
            self, 
            poblacion: List[List[int]], 
            matriz_distancias: Dict[int, Dict[int, Union[float, int]]]
    ) -> List[List[int]]:
        """
        Evoluciona la población mediante los operadores de selección, cruce, mutación y elitismo, 
        creando así la nueva generación.

        En primer lugar, se calcula la aptitud de cada individuo en la población. Luego:
         1. Se seleccionan los individuos de élite (un porcentaje de los mejores).
         2. Se selecciona un subconjunto más amplio de la población (por ejemplo, el 50% superior) 
            del cual se extraen parejas para cruzar.
         3. Se generan nuevos individuos por cruce y se aplican mutaciones.
         4. Se reincorporan los individuos de élite directamente a la nueva población.
         5. La nueva población mantiene el tamaño original.

        :param poblacion: Población actual de individuos (rutas).
        :type poblacion: List[List[int]]
        :param matriz_distancias: Matriz de distancias entre nodos para el cálculo de aptitud.
        :type matriz_distancias: Dict[int, Dict[int, Union[float, int]]]
        :return: Nueva población evolucionada tras aplicar selección, cruce, mutación y elitismo.
        :rtype: List[List[int]]
        """
        # Calcula la aptitud para cada individuo
        puntuaciones_aptitud = [self._calcular_fitness(ind, matriz_distancias) for ind in poblacion]

        # Determina el número de individuos élite que se preservarán en la siguiente generación
        numero_elite = max(1, int(self.proporcion_elite * len(poblacion)))

        # Selecciona a los mejores individuos élite
        elite = self._seleccion(poblacion, puntuaciones_aptitud, numero_elite)

        # Selecciona el 50% superior de la población para la reproducción (cruce)
        seleccionados = self._seleccion(poblacion, puntuaciones_aptitud, len(poblacion) // 2)

        # Genera la nueva población (sin contar a la élite)
        nueva_poblacion = []
        while len(nueva_poblacion) < (len(poblacion) - len(elite)):
            padre1, padre2 = random.sample(seleccionados, 2)
            hijo = self._mutar(self._cruce(padre1, padre2))
            nueva_poblacion.append(hijo)

        # Añade los individuos de élite a la nueva población
        nueva_poblacion += elite

        return nueva_poblacion

    def _calcular_ruta_completa(
            self, 
            mejor_ruta: List[int]
    ) -> List[int]:
        """
        Construye la ruta completa incluyendo los caminos más cortos entre nodos consecutivos 
        de la mejor ruta encontrada.

        Para cada par consecutivo de nodos en la mejor ruta, se obtienen los nodos intermedios que 
        conforman el trayecto más corto en la red vial, uniéndolos en un solo recorrido continuo.

        :param mejor_ruta: Ruta base (individuo) encontrada por el algoritmo genético como la mejor solución.
        :type mejor_ruta: List[int]
        :return: Ruta completa que incluye los nodos intermedios en cada segmento.
        :rtype: List[int]
        :raises ErrorRedVial: Si no se puede encontrar un camino entre dos nodos consecutivos.
        """
        ruta_completa = []
        for i in range(len(mejor_ruta) - 1):
            origen = mejor_ruta[i]
            destino = mejor_ruta[i + 1]
            try:
                subruta = nx.shortest_path(
                    self.grafica,
                    origen,
                    destino,
                    weight='length'
                )
                # Extiende la ruta agregando todos los nodos intermedios salvo el último 
                # (para no duplicar el destino)
                ruta_completa.extend(subruta[:-1])
            except nx.NetworkXNoPath:
                raise ErrorRedVial(f"No hay ruta entre los nodos {origen} y {destino}")
        ruta_completa.append(mejor_ruta[-1])  # Añade el último nodo al final de la ruta
        return ruta_completa

    def _calcular_distancia_total(
            self, 
            ruta_completa: List[int]
    ) -> float:
        """
        Calcula la distancia total de la ruta completa, sumando las longitudes de todas las aristas 
        que la componen.

        :param ruta_completa: Lista de nodos que conforman la ruta completa.
        :type ruta_completa: List[int]
        :return: Distancia total de la ruta en metros.
        :rtype: float
        """
        distancia_total = 0.0
        for i in range(len(ruta_completa) - 1):
            nodo_inicio = ruta_completa[i]
            nodo_destino = ruta_completa[i + 1]
            if self.grafica.has_edge(nodo_inicio, nodo_destino):
                datos_arista = self.grafica.get_edge_data(nodo_inicio, nodo_destino)
                # Selecciona la primera arista si existen múltiples entre los nodos
                arista_key = list(datos_arista.keys())[0]
                distancia_total += datos_arista[arista_key].get('length', 1.0)
        return distancia_total

    def _graficar_ruta(
            self, 
            ruta_completa: List[int]
    ) -> None:
        """
        Genera una representación visual de la ruta completa sobre la red vial, utilizando 
        las herramientas de OSMNX y Matplotlib.

        :param ruta_completa: Lista de nodos que conforman la ruta completa.
        :type ruta_completa: List[int]
        :raises ErrorRedVial: Si ocurre un error durante la generación de la visualización.
        """
        try:
            fig, ax = ox.plot_graph_route(
                self.grafica,
                ruta_completa,
                route_linewidth=6,
                node_size=10,
                bgcolor='white',
                edge_color='gray',
                edge_linewidth=0.5
            )
            plt.show()
        except Exception as e:
            raise ErrorRedVial(f"Error al graficar la ruta: {e}")

    # -------------------------------------------------------------------------
    # Métodos públicos
    # -------------------------------------------------------------------------

    def resolver_y_mostrar_tsp(self) -> None:
        """
        Resuelve el Problema del Viajante (TSP) utilizando un algoritmo genético con elitismo 
        y muestra la mejor ruta encontrada sobre la red vial.

        Pasos que realiza este método:
         1. Asigna pesos a las aristas basados en la longitud.
         2. Verifica y ajusta la conectividad de la red para asegurar la existencia de caminos 
            entre todos los nodos.
         3. Calcula la matriz de distancias utilizando el algoritmo de Dijkstra.
         4. Crea una población inicial de rutas (individuos).
         5. Evoluciona la población a lo largo de varias generaciones mediante los operadores de 
            selección, cruce, mutación y elitismo.
         6. Identifica la mejor ruta (individuo) y reconstruye el recorrido completo con los 
            caminos más cortos entre nodos consecutivos.
         7. Calcula la distancia total recorrida.
         8. Imprime por consola el resultado de la optimización y grafica la ruta para su 
            visualización.

        :raises ErrorRedVial: Si ocurre un error en la preparación de la red, en el cálculo 
                              de distancias o en la construcción de la ruta completa.
        """
        print('Resolviendo el problema del viajante (TSP) con un algoritmo genético elitista...')

        # Asigna pesos y verifica conectividad
        try:
            self._asignar_pesos_aristas()
            self._verificar_conectividad()
        except ErrorRedVial as e:
            raise ErrorRedVial(f"Error durante la preparación del TSP: {e}")

        # Obtiene la lista de nodos y calcula la matriz de distancias
        nodos = list(self.grafica.nodes)
        try:
            matriz_distancias = self._calcular_matriz_distancias(nodos)
        except ErrorRedVial as e:
            raise ErrorRedVial(f"Error al calcular la matriz de distancias: {e}")

        # Genera la población inicial
        try:
            poblacion = self._crear_poblacion(nodos)
        except Exception as e:
            raise ErrorRedVial(f"Error al crear la población inicial: {e}")

        # Evoluciona la población a través de las generaciones
        ancho_barra = 80
        try:
            for _ in tqdm(range(self.generaciones), desc="Generaciones", ncols=ancho_barra):
                poblacion = self._evolucionar_poblacion(poblacion, matriz_distancias)
        except Exception as e:
            raise ErrorRedVial(f"Error durante la evolución de la población: {e}")

        # Identifica el mejor individuo y calcula la ruta completa
        try:
            mejor_individuo = min(
                poblacion,
                key=lambda ind: self._calcular_fitness(ind, matriz_distancias)
            )
            ruta_completa = self._calcular_ruta_completa(mejor_individuo)
            distancia_total = self._calcular_distancia_total(ruta_completa)
            distancia_total_km = distancia_total / 1000.0  # Convierte a kilómetros
        except ErrorRedVial as e:
            raise ErrorRedVial(f"Error al construir la ruta completa: {e}")
        except Exception as e:
            raise ErrorRedVial(f"Error al calcular la distancia total: {e}")

        # Muestra los resultados obtenidos
        print(f"\nMejor ruta encontrada (orden de nodos):\n{mejor_individuo}")
        print(f"Ruta completa (incluye nodos intermedios):\n{ruta_completa}")
        print(f"Distancia total: {distancia_total_km:,.2f} km")

        # Genera una representación gráfica de la ruta
        try:
            print("\nGenerando la gráfica de la ruta, por favor espere...")
            self._graficar_ruta(ruta_completa)
        except ErrorRedVial as e:
            print(f"[Error] {e}")
