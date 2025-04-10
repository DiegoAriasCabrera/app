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
    Indica que ocurrió un problema relacionado con archivos shapefile.

    :param mensaje: Mensaje descriptivo del problema ocurrido.
    :type mensaje: str
    """
    def __init__(self, mensaje: str) -> None:
        super().__init__(mensaje)


class ErrorRedVial(Exception):
    """
    Indica que ocurrió un problema al construir o manejar la red vial,
    conformada por nodos y aristas.

    :param mensaje: Mensaje descriptivo del problema ocurrido.
    :type mensaje: str
    """
    def __init__(self, mensaje: str) -> None:
        super().__init__(mensaje)


# ============================================================================
# Clase Camion
# ============================================================================
class Camion:
    """
    Representa un camión que incluye información de capacidad, factor de reserva,
    cantidad disponible y un identificador.

    :param capacidad: Capacidad máxima del camión (peso máximo de carga).
    :type capacidad: float
    :param factor_reserva: Factor multiplicador para incluir margen adicional de capacidad.
    :type factor_reserva: float
    :param cantidad_camiones: Cantidad de camiones de este tipo disponibles.
    :type cantidad_camiones: int
    :param nombre: Nombre o identificador del camión.
    :type nombre: str
    """

    def __init__(
        self,
        capacidad: float,
        factor_reserva: float,
        cantidad_camiones: int,
        nombre: str
    ) -> None:

        self.capacidad: float = capacidad
        self.factor_reserva: float = factor_reserva
        self.cantidad_camiones: int = cantidad_camiones
        self.nombre: str = nombre

    def __repr__(self) -> str:
        """
        Devuelve la representación en cadena del objeto Camion.

        :return: La representación del camión en el formato "Camion(nombre, Capacidad=..., Cantidad=...)".
        :rtype: str
        """
        return f"Camion({self.nombre}, Capacidad={self.capacidad}, Cantidad={self.cantidad_camiones})"


# ============================================================================
# Clase AgrupamientoAGEB
# ============================================================================
class AgrupamientoAGEB:
    """
    Administra el proceso de agrupamiento de nodos mediante un algoritmo genético elitista.
    Facilita la asignación de camiones a cada grupo resultante y permite ajustar la asignación
    si se detectan grupos con sobrepeso.

    :param ruta_shp: Ruta al archivo Shapefile con información de las AGEB.
    :type ruta_shp: str
    :param tamano_poblacion: Cantidad de individuos en la población inicial del algoritmo genético.
    :type tamano_poblacion: int
    :param num_generaciones: Número de generaciones para la ejecución del algoritmo genético.
    :type num_generaciones: int
    :param tasa_mutacion: Probabilidad de mutación de cada gen (nodo) en un individuo.
    :type tasa_mutacion: float
    :param factor_basura: Factor que multiplica la población de cada nodo para estimar su peso en basura.
    :type factor_basura: float
    :param camiones: Lista de instancias Camion disponibles para los grupos.
    :type camiones: List[Camion]
    :param reconectar_grupos: Indica si se deben reconectar aristas entre nodos del mismo grupo tras el agrupamiento.
    :type reconectar_grupos: bool
    :param semilla_random: Semilla para reproducir resultados aleatorios.
    :type semilla_random: Optional[int]
    :param tamano_elite: Tamaño de la élite utilizada en la fase de selección elitista.
    :type tamano_elite: float
    """

    def __init__(
        self,
        ruta_shp: str,
        tamano_poblacion: int = 250,
        num_generaciones: int = 500,
        tasa_mutacion: float = 0.01,
        factor_basura: float = 1.071,
        camiones: List[Camion] = None,
        reconectar_grupos: bool = True,
        semilla_random: Optional[int] = None,
        tamano_elite: float = 0.3
    ) -> None:
        # Si se proporciona la semilla para números aleatorios, se configura.
        if semilla_random is not None:
            random.seed(semilla_random)

        self.ruta_shapefile: str = ruta_shp
        self.tamano_poblacion: int = tamano_poblacion
        self.num_generaciones: int = num_generaciones
        self.tasa_mutacion: float = tasa_mutacion
        self.factor_basura: float = factor_basura
        self.camiones: Optional[List[Camion]] = camiones
        self.reconectar_grupos: bool = reconectar_grupos
        self.tamano_elite: int = int(tamano_poblacion * tamano_elite)

        # Se intenta leer el shapefile para cargar los datos en un GeoDataFrame.
        try:
            self.gdf: gpd.GeoDataFrame = gpd.read_file(ruta_shp)
        except Exception as error:
            raise ErrorShapefile(f"Error al leer el shapefile: {error}")

        # Se construye la red principal de nodos y aristas.
        self.gráfica: nx.Graph = nx.Graph()
        self._crear_gráfica()

    # -------------------------------------------------------------------------
    # Métodos privados
    # -------------------------------------------------------------------------

    def _obtener_limite_peso(self) -> float:
        """
        Devuelve el límite de peso que se utiliza para agrupar. Si existen camiones disponibles, utiliza
        el valor máximo de sus capacidades efectivas (capacidad * factor_reserva).

        :return: Límite de peso para el agrupamiento.
        :rtype: float
        :raises ValueError: Si no se cuenta con un límite de peso y no se han proporcionado camiones.
        """
        # Verifica si hay camiones disponibles y calcula el límite de peso a partir de ellos.
        if self.camiones and len(self.camiones) > 0:
            return max(camion.capacidad * camion.factor_reserva for camion in self.camiones)
        else:
            # Si no hay camiones ni otro límite de peso especificado, se lanza excepción.
            raise ValueError("Debe especificarse un límite de peso o proporcionar camiones.")

    def _crear_gráfica(self) -> None:
        """
        Construye la red de nodos y aristas a partir de la información en el GeoDataFrame.
        Cada fila del GeoDataFrame se convierte en un nodo con atributos, y se establecen aristas
        entre nodos cuyas geometrías son adyacentes.
        
        :raises ErrorShapefile: Si la columna 'pob' no se localiza en el shapefile.
        :raises ErrorRedVial: Si ocurre un error durante la construcción de la red de nodos y aristas.
        """
        try:
            # Recorre cada fila para crear un nodo en la red.
            for indice, fila in self.gdf.iterrows():
                if 'pob' not in fila:
                    raise ErrorShapefile(
                        f"No se encontró la columna 'pob' en el shapefile para el nodo {indice}."
                    )
                # Calcula el peso del nodo con base en la población y el factor de basura.
                peso: float = fila['pob'] * self.factor_basura

                # Agrega el nodo con atributos relevantes (geometría, posición, peso).
                self.gráfica.add_node(
                    indice,
                    geometria=fila.geometry,
                    posicion=(fila.geometry.centroid.x, fila.geometry.centroid.y),
                    peso=peso
                )

            # Establece las aristas verificando la adyacencia (touches) entre geometrías.
            for i, fila1 in self.gdf.iterrows():
                for j, fila2 in self.gdf.iterrows():
                    if i < j and fila1.geometry.touches(fila2.geometry):
                        self.gráfica.add_edge(i, j)

        except Exception as error:
            raise ErrorRedVial(f"Error creando la red de nodos y aristas: {error}")

    def _crear_poblacion_inicial(self, grafica: nx.Graph, limite: float) -> List[List[int]]:
        """
        Crea la población inicial para el algoritmo genético sobre la gráfica especificada. 
        Cada individuo es una lista en la que cada posición indica el grupo asignado al nodo 
        correspondiente. Esta recorre los nodos y los agrupa sin excederse del límite de peso, 
        generando múltiples soluciones aleatorias que conformarán la población inicial.
        
        :param grafica: Estructura de nodos y aristas sobre la cual se generarán los individuos.
        :type grafica: nx.Graph
        :param limite: Límite de peso que no deben exceder los grupos.
        :type limite: float
        :return: Lista de individuos representados como listas de grupos asignados a cada nodo.
        :rtype: List[List[int]]
        """
        poblacion: List[List[int]] = []
        # Determina la cantidad total de nodos para poder indexar correctamente a cada uno.
        num_nodos: int = grafica.number_of_nodes()
        nodos_ordenados = list(grafica.nodes())

        # Se generan tantas soluciones (individuos) como se definió en tamano_poblacion.
        for _ in range(self.tamano_poblacion):
            # Cada individuo inicia con -1, indicando que no se ha asignado grupo.
            individuo: List[int] = [-1] * num_nodos
            nodos_visitados: set = set()
            grupo_actual: int = 0

            # Recorre los nodos y aplica una estrategia voraz para agruparlos sin exceder el límite.
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

                            # Intenta agregar vecinos si no se excede el límite de peso.
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
        Calcula el valor de fitness de un individuo sobre la gráfica indicada, basándose en 
        los siguientes criterios:
        
        1) Penalización para grupos que exceden el límite de peso.
        2) Recompensa por aristas dentro de grupos y penalización por aristas que unen grupos distintos.
        3) Penalización mayor para grupos cuyo peso supera la capacidad de todos los camiones.

        :param grafica: Estructura de nodos y aristas a la cual corresponde el individuo.
        :type grafica: nx.Graph
        :param individuo: Lista que asigna un grupo a cada nodo según su posición en la lista.
        :type individuo: List[int]
        :param limite: Límite de peso permitido en los grupos.
        :type limite: float
        :return: Valor de aptitud (fitness) calculado para el individuo.
        :rtype: float
        """
        puntaje: float = 0.0
        pesos_por_grupo: Dict[int, float] = {}
        nodos_ordenados = list(grafica.nodes())

        # Calcula el peso total por cada grupo.
        for i, nodo in enumerate(nodos_ordenados):
            grupo: int = individuo[i]
            pesos_por_grupo[grupo] = pesos_por_grupo.get(grupo, 0.0) + grafica.nodes[nodo]['peso']

        # Aplica penalizaciones si se excede el límite en algún grupo.
        for peso_grupo in pesos_por_grupo.values():
            if peso_grupo > limite:
                puntaje -= (peso_grupo - limite)

        # Recibe una recompensa por cada arista interna y penalización por aristas que conectan diferentes grupos.
        for u, v in grafica.edges():
            idx_u = nodos_ordenados.index(u)
            idx_v = nodos_ordenados.index(v)
            if individuo[idx_u] == individuo[idx_v]:
                puntaje += 2.0
            else:
                puntaje -= 1.0

        # Considera la capacidad de los camiones disponibles, penalizando fuertemente los grupos demasiado pesados.
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
        Realiza un proceso de selección elitista para elegir los individuos que pasarán a la siguiente 
        generación. Se ordena la población de mayor a menor según su fitness. Luego:
        
        - Se conservan los mejores individuos.
        - Se completa la selección tomando individuos de la mitad superior.

        :param poblacion: Conjunto de individuos de la generación actual.
        :type poblacion: List[List[int]]
        :param puntajes: Lista con los valores de aptitud (fitness) correspondientes a cada individuo.
        :type puntajes: List[float]
        :param num_seleccionar: Número total de individuos que se conservarán.
        :type num_seleccionar: int
        :return: Lista de individuos que formarán parte de la siguiente generación.
        :rtype: List[List[int]]
        """
        # Combina la población con los puntajes de aptitud para poder ordenarla.
        combinados = list(zip(poblacion, puntajes))
        combinados.sort(key=lambda par: par[1], reverse=True)

        # Se separa la élite y luego se toman más individuos de la mitad superior.
        elites = combinados[:self.tamano_elite]
        mitad_superior = combinados[:len(combinados)//2]
        faltan_por_seleccionar = num_seleccionar - self.tamano_elite
        seleccion_restante = mitad_superior[self.tamano_elite:self.tamano_elite + faltan_por_seleccionar]

        nueva_seleccion = elites + seleccion_restante
        return [ind for (ind, _) in nueva_seleccion]

    def _cruzar(self, padre1: List[int], padre2: List[int]) -> List[int]:
        """
        Realiza un cruce simple entre dos individuos, eligiendo un punto de cruce 
        aleatorio y combinando la parte inicial del primer padre con la parte final 
        del segundo.

        :param padre1: Primer individuo.
        :type padre1: List[int]
        :param padre2: Segundo individuo.
        :type padre2: List[int]
        :return: Lista de enteros que representa el hijo resultante del cruce.
        :rtype: List[int]
        """
        # Si los padres son muy pequeños, devuelve el primero sin cambio.
        if len(padre1) <= 2:
            return padre1
        
        # Elige un punto de cruce entre los extremos.
        punto_cruce = random.randint(1, len(padre1) - 2)
        return padre1[:punto_cruce] + padre2[punto_cruce:]

    def _mutar(self, individuo: List[int]) -> List[int]:
        """
        Aplica la operación de mutación con una probabilidad igual a la tasa de mutación. 
        Para cada gen, se puede reasignar el grupo a un nuevo valor (max_grupo + 1).

        :param individuo: Individuo sujeto a posible mutación.
        :type individuo: List[int]
        :return: Individuo tras la operación de mutación.
        :rtype: List[int]
        """
        max_grupo = max(individuo)
        for i in range(len(individuo)):
            if random.random() < self.tasa_mutacion:
                # Aumenta el índice de grupo para promover la diversidad.
                individuo[i] = max_grupo + 1
        return individuo

    def _evolucion_poblacion(
        self,
        grafica: nx.Graph,
        poblacion: List[List[int]],
        limite: float
    ) -> Tuple[List[List[int]], List[int], float]:
        """
        Ejecuta una generación de evolución genética (cálculo de fitness, selección, cruce y 
        mutación) y devuelve la nueva población, el mejor individuo y su fitness.

        :param grafica: Estructura de nodos y aristas sobre la cual se evalúan los individuos.
        :type grafica: nx.Graph
        :param poblacion: Población de individuos de la generación actual.
        :type poblacion: List[List[int]]
        :param limite: Límite de peso que no deben exceder los grupos.
        :type limite: float
        :return: Tupla con la nueva población, el mejor individuo y la aptitud de ese mejor individuo.
        :rtype: Tuple[List[List[int]], List[int], float]
        """
        # Calcula los puntajes (fitness) para todos los individuos.
        puntajes = [self._calcular_fitness(grafica, ind, limite) for ind in poblacion]

        # Selecciona un subconjunto de individuos para generar la siguiente población.
        seleccionados = self._seleccion_elitista(poblacion, puntajes, num_seleccionar=len(poblacion) // 2)

        nueva_poblacion: List[List[int]] = []
        # Genera la nueva población hasta alcanzar el tamaño original.
        while len(nueva_poblacion) < len(poblacion):
            padre1, padre2 = random.sample(seleccionados, 2)
            hijo = self._cruzar(padre1, padre2)
            hijo_mutado = self._mutar(hijo)
            nueva_poblacion.append(hijo_mutado)

        # Identifica al mejor individuo de la generación actual.
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
        Ejecuta el algoritmo genético elitista sobre la gráfica indicada. Devuelve la asignación 
        de grupos (mejor individuo), así como diccionarios de grupos y pesos.

        :param grafica: Estructura de nodos y aristas que se agrupa.
        :type grafica: nx.Graph
        :param limite: Límite de peso por grupo.
        :type limite: float
        :param reconectar: Indica si deben reconectarse aristas dentro de cada grupo al finalizar.
        :type reconectar: bool
        :return: Tupla con el mejor individuo, un diccionario de grupos (nodos) y un diccionario de pesos por grupo.
        :rtype: Tuple[List[int], Dict[int, List[int]], Dict[int, float]]
        :raises RuntimeError: Si no se identifica un individuo válido al concluir las generaciones.
        """
        # Genera la población inicial.
        poblacion = self._crear_poblacion_inicial(grafica, limite)
        mejor_fitness = float('-inf')
        mejor_individuo: Optional[List[int]] = None

        # Itera sobre el número de generaciones definidas.
        for _ in tqdm(range(self.num_generaciones), desc="Generaciones", ncols=80):
            poblacion, ind, fitness = self._evolucion_poblacion(grafica, poblacion, limite)
            if fitness > mejor_fitness:
                mejor_fitness = fitness
                mejor_individuo = ind

        # Verifica que se haya encontrado un individuo válido.
        if mejor_individuo is None:
            raise RuntimeError("No se encontró un individuo válido.")

        # Se construye la asignación final de grupos y sus pesos.
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

        # Elimina las aristas que conectan grupos distintos.
        aristas_eliminar = []
        for u, v in grafica.edges():
            idx_u = nodos_ordenados.index(u)
            idx_v = nodos_ordenados.index(v)
            if mejor_individuo[idx_u] != mejor_individuo[idx_v]:
                aristas_eliminar.append((u, v))
        grafica.remove_edges_from(aristas_eliminar)

        # Si se requiere, se reconectan los nodos dentro de cada grupo para formar subgráficas completas.
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
        Ejecuta el algoritmo genético elitista sobre la gráfica para encontrar la mejor partición 
        en grupos, respetando el límite de peso establecido.

        :return: Tupla con el mejor individuo, un diccionario de grupos (nodos) y un diccionario de pesos por grupo.
        :rtype: Tuple[List[int], Dict[int, List[int]], Dict[int, float]]
        """
        # Calcula o verifica el límite de peso con base en las capacidades de los camiones.
        limite_peso = self._obtener_limite_peso()
        # Ejecuta el proceso de agrupamiento genético completo.
        return self._ejecutar_agrupamiento_genetico(
            grafica=self.gráfica,
            limite=limite_peso,
            reconectar=self.reconectar_grupos
        )

    def asignar_camiones(
        self,
        grupos: Dict[int, List[int]],
        pesos_grupos: Dict[int, float]
    ) -> Tuple[Dict[int, Optional['Camion']], List['Camion']]:
        """
        Asigna camiones a los grupos generados según la capacidad efectiva de cada camión 
        (capacidad * factor_reserva). Además:

        - Identifica "grupos prioritarios": aquellos de un solo nodo con peso que excede la capacidad mínima.
        - Asigna dichos grupos prioritarios con camiones de mayor capacidad.
        - Para el resto, se usa el camión más pequeño que sea suficiente.

        :param grupos: Diccionario que vincula el identificador de grupo con la lista de nodos contenidos.
        :type grupos: Dict[int, List[int]]
        :param pesos_grupos: Diccionario que vincula el identificador de grupo con su peso total.
        :type pesos_grupos: Dict[int, float]
        :return: Tupla (asignaciones, camiones_restantes).
            - asignaciones: Diccionario que asigna cada grupo a un camión (o None si no pudo asignarse).
            - camiones_restantes: Lista de camiones no utilizados.
        :rtype: Tuple[Dict[int, Optional[Camion]], List[Camion]]
        """
        # Si no hay camiones, cada grupo queda sin asignación.
        if not self.camiones:
            return {g: None for g in grupos}, []

        # Crea la lista completa de camiones disponibles (según la cantidad de cada tipo).
        camiones_disponibles: List[Camion] = []
        for camion in self.camiones:
            for _ in range(camion.cantidad_camiones):
                camiones_disponibles.append(camion)

        # Si la lista resultante está vacía, todos los grupos quedan sin asignación.
        if not camiones_disponibles:
            return {g: None for g in grupos}, []

        # Encuentra la capacidad mínima de todos los camiones disponibles.
        min_cap = min(c.capacidad * c.factor_reserva for c in camiones_disponibles)

        grupos_prioritarios: Dict[int, List[int]] = {}
        grupos_restantes: Dict[int, List[int]] = {}

        # Determina qué grupos son prioritarios (un solo nodo con peso mayor a la capacidad mínima).
        for g, nodos in grupos.items():
            if len(nodos) == 1 and pesos_grupos[g] > min_cap:
                grupos_prioritarios[g] = nodos
            else:
                grupos_restantes[g] = nodos

        asignaciones: Dict[int, Optional[Camion]] = {}

        # Ordena camiones en orden descendente para asignar primero a los grupos prioritarios.
        camiones_disponibles.sort(key=lambda c: c.capacidad * c.factor_reserva, reverse=True)
        grupos_prioritarios_ordenados = sorted(
            grupos_prioritarios.items(),
            key=lambda item: pesos_grupos[item[0]],
            reverse=True
        )

        # Asigna camiones de mayor capacidad a grupos prioritarios.
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

        # Ordena en orden ascendente para asignar camiones más pequeños a los grupos restantes.
        camiones_disponibles.sort(key=lambda c: c.capacidad * c.factor_reserva)
        grupos_restantes_ordenados = sorted(
            grupos_restantes.items(),
            key=lambda item: pesos_grupos[item[0]],
            reverse=True
        )

        # Asigna camiones adecuados a los grupos restantes.
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

        :param nodos: Lista de nodos que integran la subgráfica.
        :type nodos: List[int]
        :param limite_peso_sub: Límite de peso para los grupos en la subgráfica.
        :type limite_peso_sub: float
        :return: Tupla con el mejor individuo, un diccionario de subgrupos (nodos) y un diccionario de pesos por subgrupo.
        :rtype: Tuple[List[int], Dict[int, List[int]], Dict[int, float]]
        """
        # Crea una copia de la red con solo los nodos y aristas del subgrupo.
        subgrafica = self.gráfica.subgraph(nodos).copy()
        # Ejecuta el agrupamiento genético sobre la subgrafica y regresa los resultados.
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

        - Subdivide los grupos que no cuentan con camión asignado (porque su peso excede la capacidad de todos los camiones disponibles) mediante un nuevo agrupamiento (subgráfica).
        - Vuelve a asignar camiones a la estructura resultante.
        - Intenta reagrupar los subgrupos que provengan del mismo grupo original, si existe un camión capaz de cubrirlos en conjunto.

        :param grupos: Diccionario original que asigna cada grupo a su lista de nodos.
        :type grupos: Dict[int, List[int]]
        :param pesos_grupos: Diccionario que asigna el peso total de cada grupo.
        :type pesos_grupos: Dict[int, float]
        :param asignaciones: Diccionario que vincula cada grupo a un camión o None.
        :type asignaciones: Dict[int, Optional[Camion]]
        :param camiones_restantes: Lista de camiones que no se utilizaron en la asignación previa.
        :type camiones_restantes: List[Camion]
        :return: Tupla con la nueva asignación de grupos, los pesos actualizados y la asignación de camiones.
        :rtype: Tuple[Dict[int, List[int]], Dict[int, float], Dict[int, Optional[Camion]]]
        """
        # Verifica si existen camiones restantes; si no, no es posible subdividir ni reagrupar.
        if not camiones_restantes:
            print("No hay camiones restantes para intentar dividir ni reagrupar grupos.")
            return grupos, pesos_grupos, asignaciones

        # Determina la máxima capacidad efectiva entre los camiones que quedan.
        capacidad_maxima = max(c.capacidad * c.factor_reserva for c in camiones_restantes)

        nuevos_grupos: Dict[int, List[int]] = {}
        nuevos_pesos: Dict[int, float] = {}
        nuevas_asignaciones: Dict[int, Optional[Camion]] = {}
        grupo_padre: Dict[int, int] = {}
        id_nuevo: int = 0

        # Subdivide los grupos cuyo peso excede la capacidad máxima disponible.
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

        # Reasigna camiones a la nueva estructura de grupos.
        nuevas_asignaciones, camiones_sobrantes = self.asignar_camiones(nuevos_grupos, nuevos_pesos)

        # Verifica si es posible reagrupar subgrupos surgidos de la misma división, usando un camión de mayor capacidad.
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

        # Reagrupa subgrupos compatibles con algún camión disponible.
        for (p_id, lista_sub_ids, peso_total) in ids_a_unir:
            todos_nodos = []
            for sid in lista_sub_ids:
                todos_nodos.extend(nuevos_grupos[sid])

            nuevo_id = min(lista_sub_ids)
            nuevos_grupos[nuevo_id] = todos_nodos
            nuevos_pesos[nuevo_id] = peso_total

            # Elimina los subgrupos previos tras reagruparlos en uno solo.
            for sid in lista_sub_ids:
                if sid != nuevo_id:
                    if sid in nuevos_grupos:
                        del nuevos_grupos[sid]
                    if sid in nuevos_pesos:
                        del nuevos_pesos[sid]
                    if sid in nuevas_asignaciones:
                        del nuevas_asignaciones[sid]

            # Busca un camión que pueda cargar la nueva agrupación.
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
        asignaciones: Dict[int, Optional[Camion]]
    ) -> None:
        """
        Genera una visualización que muestra los nodos y sus conexiones, coloreando cada grupo de 
        acuerdo con el camión asignado. Los grupos sin camión se representan en rojo.

        :param grupos: Diccionario que vincula cada grupo con los nodos que lo componen.
        :type grupos: Dict[int, List[int]]
        :param asignaciones: Diccionario que vincula cada grupo con un objeto Camion o con None.
        :type asignaciones: Dict[int, Optional[Camion]]
        :return: No devuelve un valor.
        :rtype: None
        """
        # Calcula posiciones (x, y) a partir del centroide de cada geometría.
        posiciones: Dict[Any, Tuple[float, float]] = {
            indice: (fila.geometry.centroid.x, fila.geometry.centroid.y)
            for indice, fila in self.gdf.iterrows()
        }

        # Define una lista de colores base.
        colores_base: List[str] = [
            'blue', 'green', 'orange', 'purple',
            'brown', 'pink', 'gray', 'olive',
            'cyan', 'magenta'
        ]
        colores_por_camion: Dict[str, str] = {}
        nombres_asignados: List[str] = []

        # Asocia cada camión con un color distinto, en caso de existir.
        if self.camiones:
            for indice, camion in enumerate(self.camiones):
                color_camion = colores_base[indice % len(colores_base)]
                if camion.nombre not in nombres_asignados:
                    colores_por_camion[camion.nombre] = color_camion
                    nombres_asignados.append(camion.nombre)

        # Determina el color de cada nodo según el camión asignado al grupo.
        mapa_colores: Dict[int, str] = {}
        for id_grupo, lista_nodos in grupos.items():
            camion = asignaciones.get(id_grupo, None)
            if camion is not None:
                color_grupo = colores_por_camion.get(camion.nombre, 'blue')
            else:
                color_grupo = 'red'
            for nodo in lista_nodos:
                mapa_colores[nodo] = color_grupo

        # Configura el tamaño de la figura y dibuja la red con NetworkX.
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

        # Construye la leyenda para la asignación de colores.
        leyenda: List[Patch] = []
        camiones_agregados: set = set()
        for id_grupo, camion in asignaciones.items():
            if camion is not None and camion.nombre not in camiones_agregados:
                color_camion = colores_por_camion[camion.nombre]
                leyenda.append(
                    Patch(facecolor=color_camion, edgecolor='black', label=camion.nombre)
                )
                camiones_agregados.add(camion.nombre)

        # Si hay grupos sin asignación, se agrega un ítem rojo a la leyenda.
        if any(camion is None for camion in asignaciones.values()):
            leyenda.append(Patch(facecolor='red', edgecolor='black', label='Sin asignación'))

        # Muestra la leyenda en la figura, si corresponde.
        if leyenda:
            plt.legend(handles=leyenda, loc='best', title="Camiones")
        plt.show()

    def guardar_resultados(self, mejor_individuo: List[int], archivo: str) -> None:
        """
        Guarda la asignación de grupos del mejor individuo (solución final) en un archivo JSON.

        :param mejor_individuo: Lista que indica el grupo asignado a cada nodo.
        :type mejor_individuo: List[int]
        :param archivo: Ruta o nombre del archivo para almacenar los resultados.
        :type archivo: str
        :return: No devuelve un valor.
        :rtype: None
        """
        try:
            with open(archivo, 'w', encoding='utf-8') as f:
                json.dump(mejor_individuo, f, ensure_ascii=False, indent=2)
            print(f"Resultados guardados en {archivo}")
        except Exception as error:
            print(f"Error al guardar los resultados: {error}")


# ============================================================================
# Clase ExportarSectoresAGEB
# ============================================================================
class ExportarSectoresAGEB:
    """
    Permite leer un archivo shapefile con geometrías de AGEB y un archivo JSON con la asignación
    de cada nodo a un grupo determinado. Luego, exporta cada grupo en un shapefile independiente.

    :param ruta_agrupamiento: Ruta al archivo JSON que define la agrupación.
    :type ruta_agrupamiento: str
    :param ruta_shapefile: Ruta al archivo Shapefile con las geometrías de AGEB.
    :type ruta_shapefile: str
    :param directorio_salida: Directorio donde se guardarán los shapefiles resultantes.
    :type directorio_salida: str
    """

    def __init__(
        self,
        ruta_agrupamiento: str,
        ruta_shapefile: str,
        directorio_salida: str
    ) -> None:
        """
        Inicializa la clase con la información necesaria para procesar y exportar
        cada grupo en un shapefile distinto.
        """
        self.ruta_agrupamiento = ruta_agrupamiento
        self.ruta_shapefile = ruta_shapefile
        self.directorio_salida = directorio_salida

        # Crea el directorio de salida si no existe
        os.makedirs(self.directorio_salida, exist_ok=True)

    # -------------------------------------------------------------------------
    # Métodos privados
    # -------------------------------------------------------------------------

    def _leer_agrupamiento(self) -> Dict[int, int]:
        """
        Lee el archivo JSON que contiene la asignación de los nodos a grupos. 

        :return: Diccionario que vincula cada nodo con su grupo.
        :rtype: Dict[int, int]
        :raises ValueError: Si el archivo JSON no contiene una estructura reconocible.
        """
        with open(self.ruta_agrupamiento, 'r', encoding='utf-8') as f:
            agrupamiento = json.load(f)

        # Si es una lista, la convierte a dict {índice: grupo}.
        if isinstance(agrupamiento, list):
            return {i: g for i, g in enumerate(agrupamiento)}
        elif isinstance(agrupamiento, dict):
            # Para garantizar la consistencia, convierte las llaves a enteros si es posible.
            nuevo_dict = {}
            for k, v in agrupamiento.items():
                try:
                    k_int = int(k)
                except ValueError:
                    # Si no se puede convertir, se deja tal cual, pero se recomienda que sea int.
                    k_int = k
                nuevo_dict[k_int] = v
            return nuevo_dict
        else:
            raise ValueError("La estructura del archivo JSON no es reconocible (dict o list).")

    def _leer_shapefile(self) -> gpd.GeoDataFrame:
        """
        Lee el shapefile de AGEB usando geopandas y lo devuelve como GeoDataFrame.

        :return: El GeoDataFrame resultante con la información de las AGEB.
        :rtype: gpd.GeoDataFrame
        :raises IOError: Si el shapefile no se puede leer.
        """
        try:
            gdf = gpd.read_file(self.ruta_shapefile)
            return gdf
        except Exception as e:
            raise IOError(f"No se pudo leer el shapefile: {e}")

    def _generar_grupo_shapefile(
        self,
        grupo: int,
        geometrias: List[Any],
        crs_original: Any
    ) -> None:
        """
        Crea un shapefile para un grupo específico, con todas las geometrías asignadas
        a dicho grupo.

        :param grupo: Identificador del grupo.
        :type grupo: int
        :param geometrias: Lista de geometrías (Shapely) pertenecientes al grupo.
        :type geometrias: List[Any]
        :param crs_original: Sistema de referencia de coordenadas (CRS) original del GeoDataFrame.
        :type crs_original: Any
        :return: No devuelve un valor.
        :rtype: None
        """
        # Crea un GeoDataFrame con las geometrías de este grupo
        gdf_grupo = gpd.GeoDataFrame(
            {
                "grupo": [grupo] * len(geometrias),
                "geometry": geometrias
            },
            crs=crs_original
        )

        # Define la ruta de salida para este grupo
        nombre_salida = os.path.join(self.directorio_salida, f"Sector_{grupo}.shp")

        # Exporta el GeoDataFrame como shapefile
        gdf_grupo.to_file(nombre_salida, driver='ESRI Shapefile', encoding='utf-8')
        print(f"Grupo {grupo} exportado a {nombre_salida}.")

    # -------------------------------------------------------------------------
    # Método público
    # -------------------------------------------------------------------------

    def exportar_sectores(self) -> None:
        """
        Ejecuta el proceso de lectura del agrupamiento, lectura del shapefile
        y partición de las geometrías en distintos shapefiles según la asignación
        de grupos.
        
        :return: No devuelve un valor.
        :rtype: None
        """
        # Lee el archivo JSON con la asignación de grupos
        mejor_agrupamiento = self._leer_agrupamiento()
        
        # Lee el shapefile de AGEB
        capa_gdf = self._leer_shapefile()

        # Preparación de la estructura donde se almacenarán las geometrías por grupo
        grupos_geometria: Dict[int, List[Any]] = {}

        # Itera sobre cada feature del shapefile
        for idx, row in capa_gdf.iterrows():
            # Se utiliza directamente el índice del GeoDataFrame para identificar el nodo.
            nodo = idx
            
            # Verifica si el nodo está en el agrupamiento
            if nodo not in mejor_agrupamiento:
                print(f"Advertencia: El nodo {nodo} no está en la asignación. Se omite.")
                continue
            
            grupo = mejor_agrupamiento[nodo]
            
            if grupo not in grupos_geometria:
                grupos_geometria[grupo] = []

            grupos_geometria[grupo].append(row.geometry)

        # Crea un shapefile separado para cada grupo
        for grupo, lista_geoms in grupos_geometria.items():
            self._generar_grupo_shapefile(
                grupo=grupo,
                geometrias=lista_geoms,
                crs_original=capa_gdf.crs
            )

        print("Exportación de sectores completada.")


# ============================================================================
# Clase ProcesadorCalles
# ============================================================================
class ProcesadorCalles:
    """
    Clase que se encarga de procesar y manipular la red vial de la Ciudad de México.
    Proporciona métodos para recortar la red vial según los sectores AGEB y para corregir
    la conectividad de las subgráficas resultantes, garantizando que cada una de estas
    sea fuertemente conexa.
    """

    def __init__(
        self,
        aristas_cdmx_shp: str,
        nodos_cdmx_shp: str,
        carpeta_sectores: str,
        carpeta_calles: str,
        carpeta_nodos: str,
        carpeta_salida_calles: str,
        carpeta_salida_nodos: str,
        crs_metrico: str = "EPSG:32614",
        crs_original: str = "EPSG:4326"
    ) -> None:
        """
        Inicializa una instancia de la clase ProcesadorCalles configurando las rutas a los
        shapefiles de la red vial de la Ciudad de México y estableciendo los directorios de
        entrada y salida para el recorte y la corrección de la red.

        :param aristas_cdmx_shp: Ruta al shapefile que contiene las aristas de la red vial de la Ciudad de México.
        :type aristas_cdmx_shp: str
        :param nodos_cdmx_shp: Ruta al shapefile que contiene los nodos de la red vial de la Ciudad de México.
        :type nodos_cdmx_shp: str
        :param carpeta_sectores: Directorio que contiene los shapefiles correspondientes a los sectores AGEB.
        :type carpeta_sectores: str
        :param carpeta_calles: Directorio donde se almacenarán los shapefiles de las calles recortadas.
        :type carpeta_calles: str
        :param carpeta_nodos: Directorio donde se almacenarán los shapefiles de los nodos recortados.
        :type carpeta_nodos: str
        :param carpeta_salida_calles: Directorio de salida para los shapefiles finales de calles corregidas.
        :type carpeta_salida_calles: str
        :param carpeta_salida_nodos: Directorio de salida para los shapefiles finales de nodos corregidos.
        :type carpeta_salida_nodos: str
        :param crs_metrico: Sistema de referencia de coordenadas métrico a utilizar. Por defecto "EPSG:32614".
        :type crs_metrico: str
        :param crs_original: Sistema de referencia de coordenadas original. Por defecto "EPSG:4326".
        :type crs_original: str
        :return: No devuelve ningún valor.
        :rtype: None
        """
        self.aristas_cdmx_shp = aristas_cdmx_shp
        self.nodos_cdmx_shp = nodos_cdmx_shp
        self.carpeta_sectores = carpeta_sectores
        self.carpeta_calles = carpeta_calles
        self.carpeta_nodos = carpeta_nodos
        self.carpeta_salida_calles = carpeta_salida_calles
        self.carpeta_salida_nodos = carpeta_salida_nodos
        self.crs_metrico = crs_metrico
        self.crs_original = crs_original

        # Crea los directorios de salida si no existen.
        os.makedirs(self.carpeta_calles, exist_ok=True)
        os.makedirs(self.carpeta_nodos, exist_ok=True)
        os.makedirs(self.carpeta_salida_calles, exist_ok=True)
        os.makedirs(self.carpeta_salida_nodos, exist_ok=True)

    # -------------------------------------------------------------------------
    # Métodos privados
    # -------------------------------------------------------------------------

    def _leer_shapefile(self, ruta: str) -> gpd.GeoDataFrame:
        """
        Lee un shapefile desde la ruta especificada utilizando una estrategia de fallback
        en caso de error de codificación.

        Se intenta inicialmente leer el shapefile con codificación 'utf-8'. En caso de producirse
        un error de decodificación, se reintenta la lectura utilizando la codificación 'cp1252'. Si
        ambos intentos fallan, se lanza una excepción.

        :param ruta: Ruta del shapefile a leer.
        :type ruta: str
        :return: GeoDataFrame que contiene los datos del shapefile.
        :rtype: gpd.GeoDataFrame
        :raises ErrorShapefile: Si no se puede leer el shapefile con ninguna de las codificaciones.
        """
        try:
            # Se intenta cargar el shapefile utilizando la codificación UTF-8.
            return gpd.read_file(ruta, encoding='utf-8')
        except UnicodeDecodeError:
            # Se informa que la codificación UTF-8 falló y se procede a intentar con la codificación cp1252.
            print(f"[Aviso] Error de decodificación con UTF-8 para: {ruta}. Se intenta con cp1252.")
            try:
                return gpd.read_file(ruta, encoding='cp1252')
            except Exception as excepcion:
                # Se lanza una excepción personalizada si ambos intentos de lectura fallan.
                raise ErrorShapefile(f"No se pudo leer el shapefile {ruta} con ninguna codificación: {excepcion}")
        except Exception as excepcion:
            # Se lanza una excepción en caso de producirse cualquier otro error al intentar leer el shapefile.
            raise ErrorShapefile(f"Error al leer el shapefile {ruta}: {excepcion}")

    def _crear_grafica(
        self,
        gdf_nodos: gpd.GeoDataFrame,
        gdf_aristas: gpd.GeoDataFrame,
        col_id_nodo: str = "osmid",
        col_origen: str = "from",
        col_destino: str = "to",
        col_longitud: str = "length"
    ) -> nx.DiGraph:
        """
        Construye una gráfica dirigida a partir de los GeoDataFrames de nodos y aristas.

        Cada nodo se añade a la gráfica con su posición (coordenadas X e Y) y su geometría, mientras que
        cada arista se incorpora con su geometría y se utiliza la longitud como peso.

        :param gdf_nodos: GeoDataFrame que contiene la información de los nodos.
        :type gdf_nodos: gpd.GeoDataFrame
        :param gdf_aristas: GeoDataFrame que contiene la información de las aristas.
        :type gdf_aristas: gpd.GeoDataFrame
        :param col_id_nodo: Nombre de la columna que identifica de forma única cada nodo. Por defecto "osmid".
        :type col_id_nodo: str
        :param col_origen: Nombre de la columna que indica el nodo de origen de cada arista. Por defecto "from".
        :type col_origen: str
        :param col_destino: Nombre de la columna que indica el nodo de destino de cada arista. Por defecto "to".
        :type col_destino: str
        :param col_longitud: Nombre de la columna que representa la longitud o peso de cada arista. Por defecto "length".
        :type col_longitud: str
        :return: Gráfica dirigida construida a partir de la información de nodos y aristas.
        :rtype: nx.DiGraph
        :raises ErrorShapefile: Si falta alguna columna necesaria en los datos de nodos o aristas.
        :raises ErrorRedVial: Si ocurre un error durante la construcción de la estructura gráfica.
        """
        # Se inicializa una gráfica dirigida vacía para representar la red vial.
        grafica = nx.DiGraph()
        try:
            # Se recorren los nodos del GeoDataFrame y se añaden a la gráfica con su posición y geometría.
            for indice, nodo_actual in gdf_nodos.iterrows():
                if col_id_nodo not in nodo_actual:
                    # Se lanza un error si el identificador del nodo requerido no se encuentra presente.
                    raise ErrorShapefile(f"La columna '{col_id_nodo}' falta en los nodos.")
                grafica.add_node(nodo_actual[col_id_nodo],
                                 pos=(nodo_actual.geometry.x, nodo_actual.geometry.y),
                                 geometry=nodo_actual.geometry)
            # Se recorren las aristas del GeoDataFrame y se añaden a la gráfica con su geometría y peso.
            for indice, arista_actual in gdf_aristas.iterrows():
                if col_origen not in arista_actual or col_destino not in arista_actual:
                    # Se lanza un error si no se encuentran las columnas que indican el origen y destino de la arista.
                    raise ErrorShapefile(f"Las columnas '{col_origen}' o '{col_destino}' faltan en las aristas.")
                peso = arista_actual[col_longitud] if col_longitud in arista_actual else None
                grafica.add_edge(arista_actual[col_origen],
                                 arista_actual[col_destino],
                                 geometry=arista_actual.geometry,
                                 weight=peso)
        except KeyError as excepcion:
            # Se captura el error relacionado con la ausencia de alguna columna clave.
            raise ErrorShapefile(f"Falta una columna en el shapefile: {excepcion}")
        except Exception as excepcion:
            # Se captura cualquier otro error que ocurra durante la construcción de la estructura gráfica.
            raise ErrorRedVial(f"Error al construir la estructura de nodos y aristas: {excepcion}")
        # Se retorna la gráfica dirigida construida a partir de la información proporcionada.
        return grafica

    def _remover_nodos_grado_uno(
        self, 
        gdf_aristas: gpd.GeoDataFrame, 
        gdf_nodos: gpd.GeoDataFrame, 
        col_u: str = "u", 
        col_v: str = "v"
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Elimina de forma iterativa los nodos con grado uno y ajusta las aristas correspondientes en la red.

        Se construye una gráfica temporal para determinar el grado de cada nodo y se remueven aquellos
        nodos cuyo grado sea uno. Posteriormente, se filtran los GeoDataFrames de nodos y aristas para conservar
        únicamente los elementos que conforman la subgráfica resultante.

        :param gdf_aristas: GeoDataFrame que contiene las aristas de la red.
        :type gdf_aristas: gpd.GeoDataFrame
        :param gdf_nodos: GeoDataFrame que contiene los nodos de la red.
        :type gdf_nodos: gpd.GeoDataFrame
        :param col_u: Nombre de la columna que representa el nodo de origen en las aristas. Por defecto "u".
        :type col_u: str
        :param col_v: Nombre de la columna que representa el nodo de destino en las aristas. Por defecto "v".
        :type col_v: str
        :return: Tupla que contiene el GeoDataFrame de aristas filtrado y el GeoDataFrame de nodos filtrado.
        :rtype: Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        """
        # Se crea una gráfica temporal para calcular el grado de cada nodo.
        grafica_temporal = nx.DiGraph()
        for indice, nodo_actual in gdf_nodos.iterrows():
            grafica_temporal.add_node(nodo_actual["osmid"])
        for indice, fila in gdf_aristas.iterrows():
            # Se añaden las aristas a la gráfica temporal utilizando las columnas definidas.
            grafica_temporal.add_edge(fila[col_u], fila[col_v])
        
        # Se identifican los nodos que tienen grado uno.
        nodos_a_remover = [nodo for nodo, grado in dict(grafica_temporal.degree()).items() if grado == 1]
        # Se eliminan de forma iterativa los nodos con grado uno hasta que ninguno quede.
        while nodos_a_remover:
            grafica_temporal.remove_nodes_from(nodos_a_remover)
            nodos_a_remover = [nodo for nodo, grado in dict(grafica_temporal.degree()).items() if grado == 1]
        
        # Se define el conjunto de nodos finales que permanecen en la gráfica tras las eliminaciones.
        nodos_finales = set(grafica_temporal.nodes())
        # Se filtra el GeoDataFrame de nodos para conservar solamente aquellos que siguen en la subgráfica.
        gdf_nodos_filtrado = gdf_nodos[gdf_nodos["osmid"].isin(nodos_finales)].copy()
        # Se filtra el GeoDataFrame de aristas para incluir solamente aquellas conectadas a nodos válidos.
        gdf_aristas_filtrado = gdf_aristas[
            gdf_aristas.apply(lambda fila: fila[col_u] in nodos_finales and fila[col_v] in nodos_finales, axis=1)
        ].copy()
        
        # Se retorna una tupla conteniendo los GeoDataFrames filtrados de aristas y nodos.
        return gdf_aristas_filtrado, gdf_nodos_filtrado

    # -------------------------------------------------------------------------
    # Métodos públicos
    # -------------------------------------------------------------------------

    def recortar_grafica_por_sectores(self) -> None:
        """
        Recorta la red vial de la Ciudad de México de acuerdo con los sectores AGEB definidos en shapefiles.

        Para cada sector, se unen todos los polígonos del shapefile correspondiente para generar una única
        geometría de recorte. Después, se seleccionan las aristas que intersectan dicha geometría y se
        identifican los nodos asociados. Los resultados se exportan a los directorios de salida configurados,
        y se genera una visualización que superpone el polígono del sector, las calles recortadas y los nodos.

        :return: No devuelve ningún valor.
        :rtype: None
        :raises ErrorShapefile: Si ocurre un error al leer, listar o escribir los shapefiles.
        :raises ErrorRedVial: Si se produce un error durante el proceso de recorte.
        """
        try:
            # Se cargan los shapefiles de aristas y nodos correspondientes a la red vial completa.
            gdf_aristas_cdmx = self._leer_shapefile(self.aristas_cdmx_shp)
            gdf_nodos_cdmx = self._leer_shapefile(self.nodos_cdmx_shp)
        except ErrorShapefile as excepcion:
            # Se lanza una excepción si ocurre un error durante la carga de los shapefiles.
            raise ErrorShapefile(f"Error al cargar los shapefiles de la Ciudad de México: {excepcion}")

        # Se configura la figura y los ejes para la visualización de la red y los sectores.
        figura, eje = plt.subplots(figsize=(10, 10))
        try:
            # Se obtiene la lista de archivos shapefile que contienen los sectores AGEB.
            archivos_sectores = [archivo for archivo in os.listdir(self.carpeta_sectores) if archivo.endswith(".shp")]
        except Exception as excepcion:
            # Se lanza una excepción si ocurre un error al acceder al directorio de sectores.
            raise ErrorShapefile(f"Error al listar archivos en {self.carpeta_sectores}: {excepcion}")

        # Se itera sobre cada sector encontrado para procesarlo.
        for archivo in tqdm(archivos_sectores, desc="Recortando sectores", ncols=80):
            # Se construye la ruta completa del shapefile del sector.
            ruta_sector = os.path.join(self.carpeta_sectores, archivo)
            try:
                # Se lee el shapefile del sector.
                gdf_sector = self._leer_shapefile(ruta_sector)
            except ErrorShapefile as excepcion:
                # Se informa de un error específico al leer el sector y se continúa con el siguiente.
                print(f"[Error] No se pudo leer el sector {ruta_sector}: {excepcion}")
                continue

            try:
                # Se crea una única geometría de recorte a partir de la unión de todos los polígonos del sector.
                poligono_sector = gdf_sector.unary_union
                # Se seleccionan las aristas de la red que intersectan la geometría del sector.
                calles_recortadas = gdf_aristas_cdmx[gdf_aristas_cdmx.intersects(poligono_sector)].copy()
                # Se recopilan los identificadores de los nodos involucrados en las aristas recortadas.
                ids_nodos = set()
                for indice, calle in calles_recortadas.iterrows():
                    if 'u' not in calle or 'v' not in calle:
                        # Se lanza un error si las columnas 'u' o 'v' no están presentes en la información de la arista.
                        raise ErrorShapefile("Faltan columnas 'u' o 'v' en las aristas.")
                    ids_nodos.add(calle["u"])
                    ids_nodos.add(calle["v"])
                # Se filtran los nodos de la red que participan en las aristas recortadas.
                nodos_recortados = gdf_nodos_cdmx[gdf_nodos_cdmx["osmid"].isin(ids_nodos)].copy()
                # Se definen las rutas de salida para guardar los shapefiles de calles y nodos recortados.
                ruta_salida_calles = os.path.join(self.carpeta_calles, f"{os.path.splitext(archivo)[0]}_calles.shp")
                ruta_salida_nodos = os.path.join(self.carpeta_nodos, f"{os.path.splitext(archivo)[0]}_nodos.shp")
                # Se exportan los datos recortados a archivos shapefile utilizando UTF-8.
                calles_recortadas.to_file(ruta_salida_calles, encoding="utf-8")
                nodos_recortados.to_file(ruta_salida_nodos, encoding="utf-8")
                # Se procede a graficar el sector, las calles recortadas y los nodos sobre el mismo eje.
                gdf_sector.plot(ax=eje, edgecolor="red", facecolor="none")
                calles_recortadas.plot(ax=eje, color="blue", linewidth=0.8)
                nodos_recortados.plot(ax=eje, color="green", markersize=8)
            except ErrorShapefile as excepcion:
                # Se informa de un error ocurrido durante el procesamiento del sector actual y se continúa con el siguiente.
                print(f"[Error] En el sector {archivo}: {excepcion}")
                continue
            except Exception as excepcion:
                # Se lanza una excepción general en caso de otros errores durante el recorte.
                raise ErrorRedVial(f"Error al recortar el sector {archivo}: {excepcion}")

        # Se definen los elementos de la leyenda para la visualización de la gráfica.
        elementos_leyenda = [
            Line2D([0], [0], color='red', lw=2, label='Sectores'),
            Line2D([0], [0], color='blue', lw=2, label='Calles'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Nodos')
        ]
        # Se muestra la leyenda en la gráfica.
        plt.legend(handles=elementos_leyenda, loc='best')
        plt.yticks([])
        plt.xticks([])
        # Se despliega la figura resultante.
        plt.show()
        # Se informa en consola la finalización del proceso y la ubicación de los archivos generados.
        print("Proceso de recorte completado. Archivos guardados en:")
        print(f" - Calles recortadas: {self.carpeta_calles}")
        print(f" - Nodos recortados: {self.carpeta_nodos}")


    def corregir_conectividad(
        self, 
        nodos_recortados_shp: str, 
        aristas_recortadas_shp: str
    ) -> None:
        """
        Corrige la conectividad de un sector recortado para garantizar que la subgráfica resultante
        sea fuertemente conexa. Toma los siguientes pasos:

        - Carga y convierte los shapefiles del sector recortado y de la red vial completa al CRS métrico.
        - Cálculo de la longitud de las aristas y construcción de las gráficas dirigidas para la subgráfica
            local y la red global de la Ciudad de México.
        - Identificación de los nodos originales de la subgráfica y verificación de sus componentes fuertemente conexas.
        - Generación de pares de nodos entre componentes distintas y búsqueda de la ruta más corta en la red global.
        - Segmentación de las rutas encontradas para crear aristas de conexión e incorporación de dichas aristas a la subgráfica local.
        - Adición de aristas directas de la red global que conecten nodos pertenecientes a la subgráfica.
        - Eliminación iterativa de nodos de grado uno para depurar la subgráfica.
        - Conversión de los resultados finales al CRS original y exportación de los shapefiles corregidos.

        :param nodos_recortados_shp: Ruta al shapefile que contiene los nodos del sector recortado.
        :type nodos_recortados_shp: str
        :param aristas_recortadas_shp: Ruta al shapefile que contiene las aristas del sector recortado.
        :type aristas_recortadas_shp: str
        :return: No devuelve ningún valor.
        :rtype: None
        :raises ErrorShapefile: Si ocurre un error al leer o escribir alguno de los shapefiles.
        :raises ErrorRedVial: Si ocurre un error durante la conversión del CRS o en el proceso de corrección de conectividad.
        """
        try:
            # Se cargan los shapefiles del sector recortado y de la red vial completa.
            gdf_nodos_local = self._leer_shapefile(nodos_recortados_shp)
            gdf_aristas_local = self._leer_shapefile(aristas_recortadas_shp)
            gdf_nodos_cdmx = self._leer_shapefile(self.nodos_cdmx_shp)
            gdf_aristas_cdmx = self._leer_shapefile(self.aristas_cdmx_shp)
        except ErrorShapefile as excepcion:
            # Se lanza una excepción si falla la carga de alguno de los shapefiles.
            raise ErrorShapefile(f"Error al cargar los shapefiles: {excepcion}")

        try:
            # Se convierten todos los GeoDataFrames al sistema de referencia métrico configurado.
            gdf_nodos_local = gdf_nodos_local.to_crs(self.crs_metrico)
            gdf_aristas_local = gdf_aristas_local.to_crs(self.crs_metrico)
            gdf_nodos_cdmx = gdf_nodos_cdmx.to_crs(self.crs_metrico)
            gdf_aristas_cdmx = gdf_aristas_cdmx.to_crs(self.crs_metrico)
        except Exception as excepcion:
            # Se lanza una excepción en caso de error durante la conversión de CRS.
            raise ErrorRedVial(f"Error al convertir el CRS: {excepcion}")

        # Se calcula la longitud de cada arista en los GeoDataFrames locales y globales.
        gdf_aristas_local["length"] = gdf_aristas_local.geometry.length
        gdf_aristas_cdmx["length"] = gdf_aristas_cdmx.geometry.length

        # Se construyen las gráficas dirigidas para la subred local y la red vial completa.
        grafica_local = self._crear_grafica(
            gdf_nodos=gdf_nodos_local,
            gdf_aristas=gdf_aristas_local,
            col_id_nodo="osmid",
            col_origen="from",
            col_destino="to",
            col_longitud="length"
        )
        grafica_cdmx = self._crear_grafica(
            gdf_nodos=gdf_nodos_cdmx,
            gdf_aristas=gdf_aristas_cdmx,
            col_id_nodo="osmid",
            col_origen="from",
            col_destino="to",
            col_longitud="length"
        )

        # Se determina el conjunto de nodos originales que forman parte del sector recortado.
        nodos_originales = set(gdf_nodos_local["osmid"])
        # Se obtienen las componentes fuertemente conexas de la subred local.
        componentes_fuertemente_conexas = list(nx.strongly_connected_components(grafica_local))
        if len(componentes_fuertemente_conexas) <= 1:
            # Se informa que la subred ya es fuertemente conexa y no requiere correcciones.
            print("La subgráfica ya es fuertemente conexa. No se requieren correcciones de rutas.")
        else:
            # Se informa la cantidad de componentes encontradas que requieren ser conectadas.
            print(f"Se encontraron {len(componentes_fuertemente_conexas)} componentes fuertemente conexas en la subgráfica recortada.")
            pares_entre_componentes = []
            # Se generan pares de nodos pertenecientes a componentes distintas para intentar conectarlos.
            for i in range(len(componentes_fuertemente_conexas)):
                for j in range(i + 1, len(componentes_fuertemente_conexas)):
                    for nodo_i in componentes_fuertemente_conexas[i]:
                        for nodo_j in componentes_fuertemente_conexas[j]:
                            pares_entre_componentes.append((nodo_i, nodo_j))
                            pares_entre_componentes.append((nodo_j, nodo_i))
            
            print(f"Se identificaron {len(pares_entre_componentes)} pares de nodos para conectar entre componentes.")
            
            rutas_globales = {}
            aristas_agregadas = 0
            # Se itera sobre cada par de nodos para intentar establecer una conexión en la red.
            with tqdm(total=len(pares_entre_componentes), desc="Corrigiendo conectividad", ncols=80) as barra:
                for (nodo_desde, nodo_hasta) in pares_entre_componentes:
                    if nx.has_path(grafica_local, nodo_desde, nodo_hasta):
                        barra.update(1)
                        continue

                    try:
                        # Se obtiene o calcula la ruta más corta en la red vial completa entre los nodos indicados.
                        if (nodo_desde, nodo_hasta) in rutas_globales:
                            ruta_global, longitud_global = rutas_globales[(nodo_desde, nodo_hasta)]
                        else:
                            ruta_global = nx.shortest_path(grafica_cdmx, source=nodo_desde, target=nodo_hasta, weight="weight")
                            longitud_global = sum(
                                grafica_cdmx[ruta_global[i]][ruta_global[i+1]]["weight"] 
                                for i in range(len(ruta_global) - 1)
                            )
                            rutas_globales[(nodo_desde, nodo_hasta)] = (ruta_global, longitud_global)
                    except nx.NetworkXNoPath:
                        barra.update(1)
                        continue
                    
                    # Se identifican los índices de los nodos de la ruta que pertenecen a la subred local.
                    indices_nodos_locales = [i for i, nodo in enumerate(ruta_global) if nodo in nodos_originales]
                    segmentos_procesados = []
                    # Se segmenta la ruta global en tramos conectados entre nodos presentes en la subred.
                    for indice_seg in range(len(indices_nodos_locales) - 1):
                        idx_inicio = indices_nodos_locales[indice_seg]
                        idx_fin = indices_nodos_locales[indice_seg + 1]
                        nodo_inicial = ruta_global[idx_inicio]
                        nodo_final = ruta_global[idx_fin]
                        
                        if grafica_local.has_edge(nodo_inicial, nodo_final):
                            continue
                        
                        # Se inicializan acumuladores para las coordenadas, longitud y valores de oneway de cada tramo.
                        coordenadas_segmento = []
                        longitud_segmento = 0
                        oneways_segmento = []
                        
                        # Se recorre la subruta para agregar las características de cada segmento entre nodos.
                        for sub_idx in range(idx_inicio, idx_fin):
                            nodo_actual = ruta_global[sub_idx]
                            nodo_siguiente = ruta_global[sub_idx + 1]
                            datos_arista = grafica_cdmx.get_edge_data(nodo_actual, nodo_siguiente)
                            if datos_arista is None:
                                continue
                                
                            geometria = datos_arista.get("geometry")
                            longitud_tramo = datos_arista.get("weight", 0.0)
                            valor_oneway = datos_arista.get("oneway", "False")
                            
                            if geometria is None:
                                continue
                            
                            coordenadas_arista = list(geometria.coords)
                            if not coordenadas_segmento:
                                coordenadas_segmento = coordenadas_arista
                            else:
                                if coordenadas_segmento[-1] == coordenadas_arista[0]:
                                    coordenadas_segmento.extend(coordenadas_arista[1:])
                                else:
                                    coordenadas_segmento.extend(coordenadas_arista)
                                    
                            longitud_segmento += longitud_tramo
                            oneways_segmento.append(valor_oneway)
                        
                        if coordenadas_segmento and len(coordenadas_segmento) >= 2:
                            # Se construye una nueva línea a partir de las coordenadas acumuladas.
                            nueva_linea = LineString(coordenadas_segmento)
                            conjunto_oneway = set(oneways_segmento)
                            if conjunto_oneway == {"True"}:
                                valor_oneway_final = 1
                            elif conjunto_oneway == {"False"}:
                                valor_oneway_final = 0
                            else:
                                valor_oneway_final = 1
                            
                            segmentos_procesados.append({
                                "from": nodo_inicial,
                                "to": nodo_final,
                                "geometry": nueva_linea,
                                "length": longitud_segmento,
                                "oneway": valor_oneway_final
                            })
                    
                    # Se agregan los segmentos procesados a la subred si no existen ya como aristas.
                    for segmento in segmentos_procesados:
                        if not grafica_local.has_edge(segmento["from"], segmento["to"]):
                            nueva_arista_flag = True
                            for arista_existente in grafica_local.edges(segmento["from"]):
                                if arista_existente[1] == segmento["to"]:
                                    nueva_arista_flag = False
                                    break
                            if nueva_arista_flag:
                                gdf_nueva_arista = gpd.GeoDataFrame([segmento], geometry="geometry", crs=gdf_aristas_local.crs)
                                gdf_aristas_local = pd.concat([gdf_aristas_local, gdf_nueva_arista], ignore_index=True)
                                grafica_local.add_edge(
                                    segmento["from"],
                                    segmento["to"],
                                    geometry=segmento["geometry"],
                                    weight=segmento["length"],
                                    oneway=segmento["oneway"]
                                )
                                aristas_agregadas += 1
                    
                    if aristas_agregadas > 0 and aristas_agregadas % 10 == 0:
                        if nx.is_strongly_connected(grafica_local):
                            print("\nLa subgráfica recortada es fuertemente conexa.")
                            break
                    
                    barra.update(1)

        print("\nAgregando aristas directas:")
        aristas_agregadas_existentes = 0
        # Se agregan aristas directas que existan en la red global para conectar nodos de la subred.
        for (nodo_origen, nodo_destino) in grafica_cdmx.edges():
            if nodo_origen in nodos_originales and nodo_destino in nodos_originales:
                if not grafica_local.has_edge(nodo_origen, nodo_destino):
                    datos_arista = grafica_cdmx.get_edge_data(nodo_origen, nodo_destino, default={})
                    geometria = datos_arista.get("geometry", None)
                    longitud_arista = datos_arista.get("weight", 0.0)
                    valor_oneway = datos_arista.get("oneway", 1)
                    
                    if geometria is not None:
                        nueva_arista = {
                            "from": nodo_origen,
                            "to": nodo_destino,
                            "geometry": geometria,
                            "length": longitud_arista,
                            "oneway": valor_oneway
                        }
                        gdf_nueva_arista = gpd.GeoDataFrame([nueva_arista], geometry="geometry", crs=gdf_aristas_local.crs)
                        gdf_aristas_local = pd.concat([gdf_aristas_local, gdf_nueva_arista], ignore_index=True)
                        grafica_local.add_edge(
                            nodo_origen,
                            nodo_destino,
                            geometry=geometria,
                            weight=longitud_arista,
                            oneway=valor_oneway
                        )
                        aristas_agregadas_existentes += 1

        if aristas_agregadas_existentes > 0:
            print(f"Se agregaron {aristas_agregadas_existentes} aristas directas que existían en la red global.")
        else:
            print("No se encontraron aristas directas adicionales para agregar.")

        print("\nEliminando nodos de grado 1 de la subgráfica recortada...")
        # Se invoca el método para eliminar nodos con grado uno y limpiar la subred.
        gdf_aristas_local, gdf_nodos_local = self._remover_nodos_grado_uno(
            gdf_aristas=gdf_aristas_local,
            gdf_nodos=gdf_nodos_local,
            col_u="from",
            col_v="to"
        )
        
        try:
            # Se reestablece el CRS original para los GeoDataFrames finales.
            gdf_nodos_local = gdf_nodos_local.to_crs(self.crs_original)
            gdf_aristas_local = gdf_aristas_local.to_crs(self.crs_original)
        except Exception as excepcion:
            raise ErrorRedVial(f"Error al convertir el CRS de regreso al original: {excepcion}")
            
        try:
            # Se definen los nombres base y las rutas finales para guardar los shapefiles corregidos.
            base_nodos = os.path.splitext(os.path.basename(nodos_recortados_shp))[0]
            ruta_salida_nodos_finales = os.path.join(self.carpeta_salida_nodos, base_nodos + "_finales.shp")
            base_aristas = os.path.splitext(os.path.basename(aristas_recortadas_shp))[0]
            ruta_salida_aristas_finales = os.path.join(self.carpeta_salida_calles, base_aristas + "_finales.shp")
            # Se guardan los shapefiles finales con codificación UTF-8.
            gdf_nodos_local.to_file(ruta_salida_nodos_finales, encoding="utf-8")
            gdf_aristas_local.to_file(ruta_salida_aristas_finales, encoding="utf-8")
        except Exception as excepcion:
            raise ErrorShapefile(f"Error al guardar los shapefiles finales: {excepcion}")
        
        # Se informa en consola la cantidad de componentes fuertes antes de la corrección y la finalización del proceso.
        print(f'\nHabía {len(componentes_fuertemente_conexas)} componentes fuertemente conexas antes de la corrección.')
        print("Corrección de conectividad completada.")
        print("Shapefiles finales guardados en:")
        print(f" - Nodos: {ruta_salida_nodos_finales}")
        print(f" - Calles: {ruta_salida_aristas_finales}")


# ============================================================================
# Clase ResolverMTSP
# ============================================================================
class ResolverTSP:
    """
    Permite resolver el Problema del Agente Viajero usando el algoritmo genético eltista
    sobre una red vial representada como un DiGraph de NetworkX. A cada arista se le asigna un
    peso basado en la longitud, y se busca la ruta (ciclo) que minimice la distancia total para
    visitar la mayor cantidad de nodos posible (idealmente todos), ya sea en redes fuertemente
    conexas o en aquellas compuestas por múltiples SCC.

    Se han incorporado dos variantes en la construcción de la ruta completa:
      - Para redes fuertemente conexas se evita la inclusión de nodos intermedios ya visitados.
        El objetivo es aproximarse a un ciclo hamiltoniano que abarque los nodos requeridos.
      - Para redes con múltiples SCC se realiza el mismo procedimiento en cada componente,
        calculando un ciclo interno (o aproximación) en cada SCC.
    """

    def __init__(
            self,
            ruta_nodos: str,
            ruta_aristas: str,
            grafica: nx.DiGraph = None,
            tamano_poblacion: int = 750,
            generaciones: int = 1500,
            tasa_mutacion: float = 0.01,
            tamano_elite: float = 0.3
    ):
        """
        Inicializa los parámetros necesarios para ejecutar el algoritmo genético que soluciona el TSP.
        
        :param ruta_nodos: Ruta al shapefile con los nodos de la red vial.
        :param ruta_aristas: Ruta al shapefile con las aristas de la red vial.
        :param grafica: Gráfica dirigida (opcional) que representa la red vial. Si no se proporciona,
                        se crea a partir de los shapefiles.
        :param tamano_poblacion: Número de individuos en la población inicial.
        :param generaciones: Número de iteraciones (generaciones) para el algoritmo genético.
        :param tasa_mutacion: Probabilidad de aplicación de mutación a cada individuo.
        :param tamano_elite: Proporción de individuos de la población que se preservan (elitismo) en cada generación.
        """
        self.ruta_nodos = ruta_nodos
        self.ruta_aristas = ruta_aristas
        self.grafica = grafica if grafica is not None else self._crear_red_vial()
        self.tamano_poblacion = tamano_poblacion
        self.generaciones = generaciones
        self.tasa_mutacion = tasa_mutacion
        self.tamano_elite = tamano_elite

    # -------------------------------------------------------------------------
    # Métodos privados
    # -------------------------------------------------------------------------
    def _crear_red_vial(self) -> nx.DiGraph:
        """
        Lee los shapefiles de nodos y aristas utilizando GeoPandas y construye la red vial.
        
        Verifica la existencia de las columnas mínimas requeridas, crea la columna 'key' si no existe
        y replica las aristas correspondientes a calles de doble sentido (oneway=0) para garantizar la bidireccionalidad.
        
        :devuelve: Gráfica dirigida (DiGraph) que representa la red vial.
        :raises ErrorShapefile: Si ocurre un error en la lectura o procesamiento de los shapefiles.
        :raises ErrorRedVial: Si ocurre un error al construir la gráfica.
        """
        try:
            gdf_nodos = gpd.read_file(self.ruta_nodos)
            gdf_aristas = gpd.read_file(self.ruta_aristas)
        except Exception as e:
            raise ErrorShapefile(f"Error al leer los shapefiles: {e}")

        if 'osmid' not in gdf_nodos.columns:
            raise ErrorShapefile("El shapefile de nodos debe incluir la columna 'osmid'.")
        gdf_nodos.set_index('osmid', inplace=True)

        columnas_requeridas = ['from', 'to']
        for col in columnas_requeridas:
            if col not in gdf_aristas.columns:
                raise ErrorShapefile(f"El shapefile de aristas debe incluir la columna '{col}'.")

        if 'key' not in gdf_aristas.columns:
            gdf_aristas['key'] = 0
        if 'oneway' not in gdf_aristas.columns:
            gdf_aristas['oneway'] = 0  # Se asume que las calles son de doble sentido

        edges_extra = []
        for idx, row in gdf_aristas.iterrows():
            if row['oneway'] == 0:
                reverse_row = copy.deepcopy(row)
                reverse_row['from'], reverse_row['to'] = row['to'], row['from']
                edges_extra.append(reverse_row)
        if edges_extra:
            import pandas as pd
            df_extra = pd.DataFrame(edges_extra, columns=gdf_aristas.columns)
            gdf_extra = gpd.GeoDataFrame(df_extra, geometry=gdf_aristas.geometry.name, crs=gdf_aristas.crs)
            gdf_aristas = pd.concat([gdf_aristas, gdf_extra], ignore_index=True)

        gdf_aristas.set_index(['from', 'to', 'key'], inplace=True)

        try:
            grafica = ox.graph_from_gdfs(gdf_nodos, gdf_aristas)
        except Exception as e:
            raise ErrorRedVial(f"Error al crear la red vial: {e}")

        return grafica

    def _asignar_pesos_aristas(self) -> None:
        """
        Asigna a cada arista de la red vial un peso basado en su longitud.
        
        Utiliza el valor contenido en la columna 'length' y asigna el valor 1.0 en caso de que dicha columna no exista.
        """
        for _, _, datos in self.grafica.edges(data=True):
            datos['weight'] = datos.get('length', 1.0)

    def _calcular_matriz_distancias(self, nodos: List[int]) -> Dict[int, Dict[int, Union[float, int]]]:
        """
        Calcula la matriz de distancias entre todos los pares de nodos especificados, usando el algoritmo de Dijkstra.
        
        Construye un diccionario anidado en el que, para cada nodo, se asigna la distancia mínima
        hacia cada otro nodo, usando la longitud de las aristas como peso.
        
        :param nodos: Lista de identificadores de nodos a considerar.
        :devuelve: Matriz de distancias en forma de diccionario anidado.
        :raises ErrorRedVial: Si ocurre algún error durante el cálculo de las distancias.
        """
        matriz = {}
        try:
            for nodo_i in nodos:
                matriz[nodo_i] = {}
                distancias_origen = nx.single_source_dijkstra_path_length(self.grafica, nodo_i, weight='length')
                for nodo_j in nodos:
                    if nodo_i != nodo_j:
                        matriz[nodo_i][nodo_j] = distancias_origen.get(nodo_j, np.inf)
                    else:
                        matriz[nodo_i][nodo_j] = 0.0
        except Exception as e:
            raise ErrorRedVial(f"Error al calcular la matriz de distancias: {e}")
        return matriz

    def _crear_poblacion(self, nodos: List[int]) -> List[List[int]]:
        """
        Genera la población inicial para el algoritmo genético, produciendo ciclos que inician y terminan
        en el mismo nodo. Cada individuo es una aproximación a un ciclo hamiltoniano
        que incluye todos los nodos especificados.
        
        :param nodos: Lista de identificadores de nodos que deben visitarse.
        :devuelve: Lista de individuos (ciclos).
        """
        poblacion = []
        for _ in range(self.tamano_poblacion):
            # Selecciona aleatoriamente un nodo de inicio
            nodo_inicial = random.choice(nodos)
            
            # Genera una permutación de los restantes
            restantes = [n for n in nodos if n != nodo_inicial]
            perm_restantes = random.sample(restantes, len(restantes))
            
            # Forma el ciclo => [nodo_inicial, ..., otros..., nodo_inicial]
            individuo = [nodo_inicial] + perm_restantes + [nodo_inicial]
            poblacion.append(individuo)
        return poblacion

    def _calcular_fitness(self, individuo: List[int], matriz: Dict[int, Dict[int, Union[float, int]]]) -> float:
        """
        Calcula la medida de fitness de un individuo que representa un ciclo. Se evalúa
        si se incluyen todos los nodos sin repetirlos (aparte del retorno al inicial),
        y se calcula la distancia total. Se penaliza fuertemente la imposibilidad de enlazar
        nodos consecutivos o la repetición indebida de nodos.
        
        :param individuo: Ciclo representado como lista de nodos (el primero y el último son el mismo).
        :param matriz: Matriz de distancias entre nodos.
        :devuelve: Valor de fitness (mientras más bajo, mejor). Se asigna un valor alto si la ruta es inviable.
        """
        distancia_total = 0.0
        visitados_unicos = set()

        for i in range(len(individuo) - 1):
            ni = individuo[i]
            nj = individuo[i + 1]

            # Ignora la repetición entre el primero y el último que cierran el ciclo
            if i < len(individuo) - 2:  
                # Se excluye el último para no contar la igualdad (primer==último)
                if ni in visitados_unicos:
                    return 1e12
                visitados_unicos.add(ni)

            dist = matriz[ni].get(nj, np.inf)
            if np.isinf(dist):
                return 1e12

            distancia_total += dist

        if len(visitados_unicos) < (len(individuo) - 1):
            return 1e12

        return distancia_total

    def _seleccionar(self, poblacion: List[List[int]], aptitudes: List[float], cantidad: int) -> List[List[int]]:
        """
        Selecciona los mejores individuos de la población según su fitness (menor es mejor).
        
        :param poblacion: Lista de individuos.
        :param aptitudes: Lista de valores de fitness asociados a cada individuo.
        :param cantidad: Número de individuos a seleccionar.
        :devuelve: Lista de individuos seleccionados.
        """
        combinacion = list(zip(poblacion, aptitudes))
        combinacion_ordenada = sorted(combinacion, key=lambda x: x[1])
        return [ind for ind, _ in combinacion_ordenada[:cantidad]]

    def _cruzar(self, padre1: List[int], padre2: List[int]) -> List[int]:
        """
        Realiza el cruce entre dos individuos (ciclos) mediante el intercambio aleatorio de un segmento.
        
        Copia un segmento del primer padre y completa los nodos faltantes con el orden del segundo padre.
        
        :param padre1: Primer individuo (lista de nodos, ciclo).
        :param padre2: Segundo individuo (lista de nodos, ciclo).
        :devuelve: Nuevo individuo (hijo) resultante del cruce, conservando la idea de ciclo.
        """
        base1 = padre1[:-1]  # Sin repetir el último
        base2 = padre2[:-1]

        tamano = len(base1)
        inicio, fin = sorted(random.sample(range(tamano), 2))
        hijo_parcial = [None] * tamano

        # Copia el segmento [inicio:fin] de base1
        hijo_parcial[inicio:fin] = base1[inicio:fin]

        # Completa con el orden del segundo padre
        indice = fin
        for nodo in base2:
            if nodo not in hijo_parcial:
                if indice >= tamano:
                    indice = 0
                hijo_parcial[indice] = nodo
                indice += 1

        # Reconstruye el ciclo agregando al final el mismo nodo que al inicio
        hijo = hijo_parcial + [hijo_parcial[0]]
        return hijo

    def _mutar(self, individuo: List[int]) -> List[int]:
        """
        Aplica mutación a un individuo, intercambiando aleatoriamente dos nodos
        según la tasa de mutación establecida. Mantiene la estructura de ciclo.
        
        :param individuo: Lista de nodos que representa el ciclo.
        :devuelve: Nuevo individuo tras aplicar la mutación.
        """
        ciclo_base = individuo[:-1]
        for i in range(len(ciclo_base)):
            if random.random() < self.tasa_mutacion:
                j = random.randint(0, len(ciclo_base) - 1)
                ciclo_base[i], ciclo_base[j] = ciclo_base[j], ciclo_base[i]
        nuevo = ciclo_base + [ciclo_base[0]]
        return nuevo

    def _evolucionar_poblacion(self, poblacion: List[List[int]], matriz: Dict[int, Dict[int, Union[float, int]]]) -> List[List[int]]:
        """
        Evoluciona la población mediante selección, cruce, mutación y elitismo.
        
        Se calculan los fitness, se selecciona una parte de la población y se genera
        nueva población hasta completar el número original, preservando la élite.
        
        :param poblacion: Población actual (lista de ciclos).
        :param matriz: Matriz de distancias.
        :devuelve: Nueva población evolucionada.
        """
        aptitudes = [self._calcular_fitness(ind, matriz) for ind in poblacion]
        numero_elite = max(1, int(self.tamano_elite * len(poblacion)))
        elite = self._seleccionar(poblacion, aptitudes, numero_elite)
        seleccionados = self._seleccionar(poblacion, aptitudes, len(poblacion) // 2)
        nueva_poblacion = []
        while len(nueva_poblacion) < (len(poblacion) - len(elite)):
            padre1, padre2 = random.sample(seleccionados, 2)
            hijo = self._mutar(self._cruzar(padre1, padre2))
            nueva_poblacion.append(hijo)
        nueva_poblacion += elite
        return nueva_poblacion

    def _calcular_ruta_completa_fuertemente_conexa(self, mejor_ruta: List[int]) -> List[int]:
        """
        Construye la ruta completa a partir del individuo óptimo para redes fuertemente conexas.
        
        El individuo ya representa un ciclo (mismo primer y último nodo), pero se verifica la
        conexión entre nodos consecutivos. Si no existe un enlace directo, se calcula la
        ruta más corta mediante nx.shortest_path. Si algún nodo intermedio ya fue visitado,
        se opta por la conexión directa para evitar duplicados.
        
        :param mejor_ruta: Lista de nodos ordenada de la solución óptima (ciclo).
        :devuelve: Lista de nodos que conforman la ruta completa (ciclo expandido).
        :raises ErrorRedVial: Si no existe ruta entre dos nodos consecutivos.
        """
        ruta_completa = []
        for i in range(len(mejor_ruta) - 1):
            origen = mejor_ruta[i]
            destino = mejor_ruta[i + 1]
            if self.grafica.has_edge(origen, destino):
                subruta = [origen, destino]
            else:
                try:
                    subruta = nx.shortest_path(self.grafica, origen, destino, weight='length')
                except nx.NetworkXNoPath:
                    raise ErrorRedVial(f"No existe ruta entre {origen} y {destino}")
                if any(n in ruta_completa for n in subruta[1:-1]):
                    subruta = [origen, destino]
            if not ruta_completa:
                ruta_completa.extend(subruta)
            else:
                if ruta_completa[-1] == subruta[0]:
                    ruta_completa.extend(subruta[1:])
                else:
                    ruta_completa.extend(subruta)
        return ruta_completa

    def _calcular_ruta_completa_scc(self, mejor_ruta: List[int]) -> List[int]:
        """
        Construye la ruta completa a partir del individuo óptimo en redes con múltiples SCC.
        
        Para cada par de nodos consecutivos se calcula la subruta con nx.shortest_path,
        filtrando nodos intermedios ya visitados. Se emplea un conjunto de nodos ya visitados
        para evitar repeticiones, manteniendo siempre el origen y destino.
        
        :param mejor_ruta: Lista de nodos que representa la solución óptima en el SCC (ciclo).
        :devuelve: Lista de nodos que conforman la ruta completa en el SCC.
        :raises ErrorRedVial: Si no existe un camino entre dos nodos consecutivos.
        """
        ruta_completa = []
        visitados = set()
        for i in range(len(mejor_ruta) - 1):
            origen = mejor_ruta[i]
            destino = mejor_ruta[i + 1]
            if self.grafica.has_edge(origen, destino):
                subruta = [origen, destino]
            else:
                try:
                    subruta = nx.shortest_path(self.grafica, origen, destino, weight='length')
                except nx.NetworkXNoPath:
                    raise ErrorRedVial(f"No existe ruta entre {origen} y {destino}")
            nueva_subruta = []
            for idx, nodo in enumerate(subruta):
                if idx == 0 or idx == len(subruta) - 1:
                    nueva_subruta.append(nodo)
                else:
                    if nodo not in visitados:
                        nueva_subruta.append(nodo)
            if not ruta_completa:
                ruta_completa.extend(nueva_subruta)
            else:
                if ruta_completa[-1] == nueva_subruta[0]:
                    ruta_completa.extend(nueva_subruta[1:])
                else:
                    ruta_completa.extend(nueva_subruta)
            for nodo in nueva_subruta:
                visitados.add(nodo)
        return ruta_completa

    def _calcular_distancia_total(self, ruta_completa: List[int]) -> float:
        """
        Calcula la distancia total de la ruta, sumando la longitud de cada arista utilizada.
        
        Se recorre la secuencia de nodos y, para cada par consecutivo, se obtiene el valor
        de la columna 'length' de la arista.
        
        :param ruta_completa: Lista de nodos que conforman la ruta completa.
        :devuelve: Distancia total en metros.
        """
        distancia_total = 0.0
        for i in range(len(ruta_completa) - 1):
            nodo_inicio = ruta_completa[i]
            nodo_destino = ruta_completa[i + 1]
            if self.grafica.has_edge(nodo_inicio, nodo_destino):
                datos_arista = self.grafica.get_edge_data(nodo_inicio, nodo_destino)
                llave = list(datos_arista.keys())[0]
                distancia_total += datos_arista[llave].get('length', 1.0)
        return distancia_total

    def _graficar_ruta(self, ruta_completa: List[int]) -> None:
        """
        Visualiza la ruta completa sobre la red vial empleando OSMNX y Matplotlib.
        
        Ajusta el tamaño de los nodos con base en la cantidad de visitas (apariciones) en la ruta.
        
        :param ruta_completa: Lista de nodos que conforman la ruta completa.
        :raises ErrorRedVial: Si ocurre algún error durante la visualización.
        """
        from matplotlib import pyplot as plt
        conteo = Counter(ruta_completa)
        tamanio_nodos = [10 * conteo.get(nodo, 0) for nodo in self.grafica.nodes()]
        try:
            fig, ax = ox.plot_graph_route(
                self.grafica,
                ruta_completa,
                route_linewidth=6,
                node_size=tamanio_nodos,
                node_color='blue',
                bgcolor='white',
                edge_color='gray',
                edge_linewidth=0.5
            )
            plt.show()
        except Exception as e:
            raise ErrorRedVial(f"Error al graficar la ruta: {e}")

    def _resolver_MTSP_fuertemente_conexa(self) -> None:
        """
        Soluciona el TSP para redes fuertemente conexas por medio de un algoritmo genético,
        buscando ciclos hamiltonianos (o aproximaciones) que visiten cada nodo una sola vez.
        
        Asigna los pesos a las aristas, calcula la matriz de distancias, genera la población inicial
        y la evoluciona durante un número determinado de generaciones. Finalmente, selecciona el mejor
        individuo, construye la ruta completa y presenta la distancia total.
        """
        try:
            self._asignar_pesos_aristas()
        except ErrorRedVial as e:
            raise ErrorRedVial(f"Error durante la asignación de pesos: {e}")

        nodos = list(self.grafica.nodes)
        try:
            matriz = self._calcular_matriz_distancias(nodos)
        except ErrorRedVial as e:
            raise ErrorRedVial(f"Error al calcular la matriz de distancias: {e}")

        try:
            poblacion = self._crear_poblacion(nodos)
        except Exception as e:
            raise ErrorRedVial(f"Error al crear la población inicial: {e}")

        try:
            for _ in tqdm(range(self.generaciones), desc="Generaciones", ncols=80):
                poblacion = self._evolucionar_poblacion(poblacion, matriz)
        except Exception as e:
            raise ErrorRedVial(f"Error durante la evolución de la población: {e}")

        try:
            mejor_individuo = min(poblacion, key=lambda ind: self._calcular_fitness(ind, matriz))
            ruta_completa = self._calcular_ruta_completa_fuertemente_conexa(mejor_individuo)
            dist_total = self._calcular_distancia_total(ruta_completa)
            dist_total_km = dist_total / 1000.0
            self.ruta_completa = ruta_completa
        except ErrorRedVial as e:
            raise ErrorRedVial(f"Error al construir la ruta completa: {e}")
        except Exception as e:
            raise ErrorRedVial(f"Error al calcular la distancia total: {e}")

        print(f"\nMejor ciclo encontrado (orden de nodos):\n{mejor_individuo}")
        print(f"Distancia total: {dist_total_km:,.2f} km")

        try:
            self._graficar_ruta(ruta_completa)
        except ErrorRedVial as e:
            print(f"[Error] {e}")

    def _resolver_MTSP_SCC(self) -> None:
        """
        Soluciona el TSP para redes compuestas por múltiples componentes fuertemente conexas (SCC).
        
        Asigna los pesos a las aristas y calcula la matriz de distancias global. Para cada SCC,
        halla un ciclo interno (o aproximación) mediante el algoritmo genético. Luego, ensambla
        dichas soluciones en una o varias rutas finales, mostrando su distancia total.
        """
        try:
            self._asignar_pesos_aristas()
        except ErrorRedVial as e:
            raise ErrorRedVial(f"Error al asignar pesos: {e}")

        nodos_totales = list(self.grafica.nodes())
        matriz_global = self._calcular_matriz_distancias(nodos_totales)
        scc_lista = list(nx.strongly_connected_components(self.grafica))
        print(f"La red contiene {len(scc_lista)} componentes fuertemente conexas.")

        soluciones_scc = []
        for idx, scc in enumerate(scc_lista, start=1):
            nodos_scc = list(scc)
            if len(nodos_scc) == 1:
                soluciones_scc.append({'ruta': nodos_scc + nodos_scc, 'nodos': nodos_scc})
                continue

            print(f"Procesando SCC {idx} con {len(nodos_scc)} nodos...")
            submatriz = {ni: {nj: matriz_global[ni][nj] for nj in nodos_scc} for ni in nodos_scc}
            poblacion_scc = self._crear_poblacion(nodos_scc)

            for _ in tqdm(range(self.generaciones), desc=f"Generaciones en SCC {idx}", ncols=80):
                poblacion_scc = self._evolucionar_poblacion(poblacion_scc, submatriz)

            mejor_scc = min(poblacion_scc, key=lambda ind: self._calcular_fitness(ind, submatriz))
            soluciones_scc.append({'ruta': mejor_scc, 'nodos': nodos_scc})

        # Se concatenan los ciclos de cada SCC
        indices_restantes = set(range(len(soluciones_scc)))
        rutas_finales = []
        while indices_restantes:
            actual_idx = indices_restantes.pop()
            ruta_actual = list(soluciones_scc[actual_idx]['ruta'])
            extendido = True
            while extendido and indices_restantes:
                extendido = False
                mejor_candidato = None
                mejor_conexion = None
                menor_longitud = float('inf')
                nodo_final = ruta_actual[-1]
                for cand in list(indices_restantes):
                    nodo_inicio_cand = soluciones_scc[cand]['ruta'][0]
                    conexion = []
                    if nx.has_path(self.grafica, nodo_final, nodo_inicio_cand):
                        try:
                            conexion = nx.shortest_path(self.grafica, nodo_final, nodo_inicio_cand, weight='length')
                        except nx.NetworkXNoPath:
                            conexion = []
                    if conexion:
                        if any(n in ruta_actual for n in conexion[1:-1]):
                            continue
                        dist_conexion = self._calcular_distancia_total(conexion)
                        if dist_conexion < menor_longitud:
                            menor_longitud = dist_conexion
                            mejor_candidato = cand
                            mejor_conexion = conexion
                if mejor_candidato is not None:
                    if ruta_actual[-1] == mejor_conexion[0]:
                        ruta_actual.extend(mejor_conexion[1:])
                    else:
                        ruta_actual.extend(mejor_conexion)
                    ruta_cand = soluciones_scc[mejor_candidato]['ruta']
                    if ruta_actual[-1] == ruta_cand[0]:
                        ruta_actual.extend(ruta_cand[1:])
                    else:
                        ruta_actual.extend(ruta_cand)
                    indices_restantes.remove(mejor_candidato)
                    extendido = True
            rutas_finales.append(ruta_actual)
        
        # Se grafican y presentan los resultados de cada ruta generada
        for idx, ruta in enumerate(rutas_finales, start=1):
            distancia = self._calcular_distancia_total(ruta) / 1000.0
            print(f"\n[Resultados - Ruta {idx} en SCC]:")
            print(f"Número de nodos en la ruta: {len(ruta)}")
            print(f"Distancia: {distancia:,.2f} km")
            try:
                self._graficar_ruta(ruta)
            except ErrorRedVial as e:
                print(f"[Error al graficar SCC {idx}] {e}")

        self.ruta_completa = rutas_finales

    # -------------------------------------------------------------------------
    # Métodos públicos
    # -------------------------------------------------------------------------
    def resolver_MTSP(self) -> None:
        """
        Determina de manera automática si la red vial es fuertemente conexa o no, y
        aplica la estrategia apropiada para resolver el problema TSP.
        
        Si la red es fuertemente conexa, aplica la estrategia de búsqueda de un ciclo
        hamiltoniano que visite todos los nodos; de lo contrario, aplica el método
        para redes con múltiples SCC.
        """
        if nx.is_strongly_connected(self.grafica):
            print("La red es fuertemente conexa.")
            self._resolver_MTSP_fuertemente_conexa()
        else:
            print("La red no es fuertemente conexa.")
            self._resolver_MTSP_SCC()

    def exportar_ruta_shapefiles(self, salida_nodos: str, salida_aristas: str) -> None:
        """
        Exporta la(s) ruta(s) generada(s) a dos shapefiles (nodos y aristas), renumerando de forma secuencial los nodos.
        
        Para redes fuertemente conexas, reconstruye la secuencia de aristas usando las subrutas reales,
        evitando la duplicación de nodos intermedios. Para varias SCC, emplea la ruta tal cual fue calculada,
        de modo que la suma de las longitudes en la tabla de atributos coincida con la distancia total calculada.
        
        :param salida_nodos: Ruta de salida para el shapefile de nodos.
        :param salida_aristas: Ruta de salida para el shapefile de aristas.
        :raises Exception: Si la ruta no ha sido calculada o si ocurre algún error al procesar la geometría.
        """
        import pandas as pd
        if not hasattr(self, 'ruta_completa'):
            raise Exception("No se ha calculado la ruta. Ejecute resolver_MTSP() primero.")

        nodos_gdf, _ = ox.graph_to_gdfs(self.grafica)
        
        # Verifica si se trata de múltiples SCC (lista de rutas) o de una sola secuencia
        if (
            isinstance(self.ruta_completa, list)
            and len(self.ruta_completa) > 0
            and isinstance(self.ruta_completa[0], list)
        ):
            rutas = self.ruta_completa
            multiple_scc = True
        else:
            rutas = [self.ruta_completa]
            multiple_scc = False
        
        todos_nodos_ruta = set()
        aristas_ruta = []

        # Recorre cada ruta para reconstruir las aristas
        for ruta in rutas:
            if len(ruta) < 2:
                continue
            for i in range(len(ruta) - 1):
                u = ruta[i]
                v = ruta[i+1]
                if u not in self.grafica.nodes or v not in self.grafica.nodes:
                    print(f"[Advertencia] Uno de los nodos ({u}, {v}) no se encuentra en la gráfica, se omite.")
                    continue

                if multiple_scc:
                    subruta = [u, v]
                else:
                    if self.grafica.has_edge(u, v):
                        subruta = [u, v]
                    else:
                        try:
                            subruta = list(nx.shortest_path(self.grafica, u, v, weight='length'))
                        except nx.NetworkXNoPath:
                            print(f"No se encontró camino entre {u} y {v} en la gráfica.")
                            continue
                        if any(n in todos_nodos_ruta for n in subruta[1:-1]):
                            subruta = [u, v]

                todos_nodos_ruta.update(subruta)
                for j in range(len(subruta) - 1):
                    aristas_ruta.append((subruta[j], subruta[j+1]))
        
        # Ordena los nodos para renumerarlos
        nodos_ruta = sorted(list(todos_nodos_ruta))
        mapeo_nodos = {nodo: nuevo_id for nuevo_id, nodo in enumerate(nodos_ruta, start=1)}

        geom_col = 'geometry'
        if geom_col not in nodos_gdf.columns:
            if hasattr(nodos_gdf, '_geometry_column_name'):
                geom_col = nodos_gdf._geometry_column_name
            else:
                raise ValueError("No se encontró la columna 'geometry' en el GeoDataFrame de nodos.")
        
        nodos_exportar = []
        for nodo in nodos_ruta:
            if nodo in nodos_gdf.index:
                datos = {
                    'node_id': mapeo_nodos[nodo],
                    'osmid': nodo,
                    'geometry': nodos_gdf.loc[nodo, geom_col]
                }
                nodos_exportar.append(datos)
            else:
                print(f"[Advertencia] El nodo {nodo} no se encontró en el GeoDataFrame de nodos.")
        
        if not nodos_exportar:
            raise ValueError("No se recolectaron nodos para exportación. Verifique la ruta calculada.")
        
        df_nodos = pd.DataFrame(nodos_exportar)
        if 'geometry' not in df_nodos.columns:
            raise ValueError("La lista de nodos no contiene la clave 'geometry'.")
        
        nodos_gdf_exportar = gpd.GeoDataFrame(df_nodos, geometry='geometry', crs=nodos_gdf.crs)
        nodos_gdf_exportar.to_file(salida_nodos)
        
        aristas_exportar = []
        for (u, v) in aristas_ruta:
            if u in mapeo_nodos and v in mapeo_nodos:
                if self.grafica.has_edge(u, v):
                    datos_arista = self.grafica.get_edge_data(u, v)
                    llave = list(datos_arista.keys())[0]
                    datos = datos_arista[llave]
                    if 'geometry' in datos and datos['geometry'] is not None:
                        geom = datos['geometry']
                    else:
                        try:
                            origen_geom = nodos_gdf.loc[u, geom_col]
                            destino_geom = nodos_gdf.loc[v, geom_col]
                            geom = LineString([origen_geom, destino_geom])
                        except KeyError:
                            print(f"No se encontró geometría para los nodos {u} o {v}, se omite la arista.")
                            continue
                    info_arista = {
                        'u': mapeo_nodos[u],
                        'v': mapeo_nodos[v],
                        'length': datos.get('length', None),
                        'geometry': geom
                    }
                    aristas_exportar.append(info_arista)
                else:
                    print(f"No se encontró arista directa entre {u} y {v}, se usa 'arista directa'.")
            else:
                print(f"Uno de los nodos ({u}, {v}) no está en el mapeo, se omite la arista.")
        
        df_aristas = pd.DataFrame(aristas_exportar)
        if 'geometry' not in df_aristas.columns:
            raise ValueError("La lista de aristas no contiene la clave 'geometry'.")
        
        aristas_gdf_exportar = gpd.GeoDataFrame(df_aristas, geometry='geometry', crs=nodos_gdf.crs)
        aristas_gdf_exportar.to_file(salida_aristas)
        
        print(f"Shapefiles exportados exitosamente: {len(nodos_exportar)} nodos y {len(aristas_exportar)} aristas.")