import os
import json
import geopandas as gpd
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory, session, flash
from RecolectarBasura import *  # Se importan las clases: AgrupamientoAGEB, Camion, ExportarSectoresAGEB, ProcesadorCalles, etc.

app = Flask(__name__)
app.secret_key = os.urandom(32)


@app.route('/')
def index():
    """
    Redirige al listado de alcaldías disponibles.
    """
    return redirect(url_for('listar_alcaldias'))


@app.route('/Alcaldías', methods=['GET'])
def listar_alcaldias():
    """
    Lista las alcaldías disponibles leyendo los archivos shapefile (.shp) ubicados en la carpeta "Alcaldías/".
    Renderiza el template "Alcaldías.html", el cual muestra un menú desplegable (<select>) con las opciones de alcaldías.
    """
    alcaldias_dir = os.path.join(app.root_path, 'Alcaldías')
    alcaldias = []
    if os.path.exists(alcaldias_dir):
        for archivo in os.listdir(alcaldias_dir):
            if archivo.endswith('.shp'):
                # Se extrae el nombre del archivo sin la extensión para utilizarlo como identificador de la alcaldía.
                nombre_alcaldia = os.path.splitext(archivo)[0]
                alcaldias.append(nombre_alcaldia)
    else:
        return jsonify({'error': 'No se encontró la carpeta de alcaldías.'}), 404

    seleccionada = session.get('alcaldia', None)
    return render_template('Alcaldías.html', alcaldías=alcaldias, seleccionada=seleccionada)


@app.route('/seleccionar_alcaldia', methods=['POST'])
def seleccionar_alcaldia():
    """
    Recibe la alcaldía seleccionada desde el formulario y la guarda en la sesión para su uso posterior.
    Posteriormente, redirige a la ruta que lista las alcaldías para mostrar los parámetros asociados.
    """
    alcaldia = request.form.get('alcaldía')
    if not alcaldia:
        return redirect(url_for('listar_alcaldias'))
    session['alcaldia'] = alcaldia
    return redirect(url_for('listar_alcaldias'))


@app.route('/Alcaldías/<alcaldia>/progreso_agrupamiento', methods=['POST'])
def progreso_agrupamiento(alcaldia):
    """
    Ejecuta el proceso de agrupamiento utilizando la clase AgrupamientoAGEB.
    
    Se leen los parámetros del formulario (tamaño de población, número de generaciones, tasa de mutación y factor de basura),
    se instancian los objetos Camion y se verifica la existencia del shapefile correspondiente a la alcaldía.
    Luego se ejecuta el agrupamiento, se asignan los camiones, se realiza un post-procesamiento y se grafica la asignación,
    guardándose la imagen y el resultado en formato JSON en la carpeta "temp/".
    
    Finalmente, se redirige a la ruta de visualización de resultados para la alcaldía.
    """
    # 1) Leer los parámetros del formulario
    tamano_poblacion = request.form.get('agrupamiento[tamano_poblacion]', type=int)
    num_generaciones = request.form.get('agrupamiento[num_generaciones]', type=int)
    tasa_mutacion    = request.form.get('agrupamiento[tasa_mutacion]', type=float)
    factor_basura    = request.form.get('agrupamiento[factor_basura]', type=float)

    # 2) Parsear la información de los camiones
    camiones = []
    i = 0
    while True:
        nombre_key    = f"camiones[{i}][nombre]"
        capacidad_key = f"camiones[{i}][capacidad]"
        reserva_key   = f"camiones[{i}][factor_reserva]"
        cantidad_key  = f"camiones[{i}][cantidad_camiones]"
        if nombre_key in request.form:
            nombre_camion = request.form.get(nombre_key)
            capacidad     = float(request.form.get(capacidad_key, 0))
            factor_res    = float(request.form.get(reserva_key, 1))
            cant          = int(request.form.get(cantidad_key, 0))
            c = Camion(
                capacidad=capacidad,
                factor_reserva=factor_res,
                cantidad_camiones=cant,
                nombre=nombre_camion
            )
            camiones.append(c)
            i += 1
        else:
            break

    # 3) Validar que existan todos los parámetros requeridos y al menos un camión
    if (not tamano_poblacion or not num_generaciones or 
        not tasa_mutacion or not factor_basura or len(camiones) == 0):
        return jsonify({'error': 'Faltan parámetros o camiones para el agrupamiento.'}), 400

    # 4) Verificar que existe el shapefile de la alcaldía
    ruta_shp = os.path.join(app.root_path, 'Alcaldías', f'{alcaldia}.shp')
    if not os.path.exists(ruta_shp):
        return jsonify({'error': f'No se encontró el shapefile para {alcaldia}.'}), 404

    # 5) Crear la instancia de AgrupamientoAGEB
    agrupador = AgrupamientoAGEB(
        ruta_shp=ruta_shp,
        tamano_poblacion=tamano_poblacion,
        num_generaciones=num_generaciones,
        tasa_mutacion=tasa_mutacion,
        factor_basura=factor_basura,
        camiones=camiones,
        reconectar_grupos=True
    )

    # 6) Ejecutar el agrupamiento para obtener el mejor individuo, los grupos y los pesos de cada grupo
    mejor_individuo, grupos, pesos_grupos = agrupador.ejecutar_agrupamiento()

    # 7) Asignar camiones a los grupos y obtener los camiones restantes
    asignaciones, camiones_restantes = agrupador.asignar_camiones(grupos, pesos_grupos)

    # 8) Realizar el post-procesamiento de la asignación
    nuevos_grupos, nuevos_pesos, nuevas_asignaciones = agrupador.post_procesar_asignacion(
        grupos, pesos_grupos, asignaciones, camiones_restantes
    )

    # 9) Graficar la asignación final con camiones y guardar la imagen resultante en la carpeta "temp/imagenes"
    carpeta_imagenes = os.path.join(app.root_path, 'temp', 'imagenes')
    os.makedirs(carpeta_imagenes, exist_ok=True)
    ruta_img = os.path.join(carpeta_imagenes, f"{alcaldia}_agrupamiento.png")

    # IMPORTANTE: Usar "output_path" (o el nombre que corresponda a la definición interna del método):
    agrupador.graficar_con_camiones(nuevos_grupos, nuevas_asignaciones, output_path=ruta_img)

    # 10) Guardar el resultado del mejor individuo (agrupamiento) en formato JSON dentro de "temp/agrupamiento"
    carpeta_agrupamiento = os.path.join(app.root_path, 'temp', 'agrupamiento')
    os.makedirs(carpeta_agrupamiento, exist_ok=True)
    ruta_json = os.path.join(carpeta_agrupamiento, f"{alcaldia}_mejor_agrupamiento.json")
    agrupador.guardar_resultados(mejor_individuo, ruta_json)

    # 11) Redirigir a la ruta que muestra los resultados y sectores generados
    return redirect(url_for('mostrar_resultados', alcaldia=alcaldia))


@app.route('/mostrar_resultados/<alcaldia>', methods=['GET'])
def mostrar_resultados(alcaldia):
    """
    Carga el archivo JSON con el mejor agrupamiento, segmenta el shapefile de la alcaldía en sectores
    y muestra la imagen resultante en el template "mostrar_resultados.html".
    """
    # 1) Definir la ruta del archivo JSON que contiene el agrupamiento
    carpeta_agrupamiento = os.path.join(app.root_path, 'temp', 'agrupamiento')
    ruta_json = os.path.join(carpeta_agrupamiento, f"{alcaldia}_mejor_agrupamiento.json")
    if not os.path.exists(ruta_json):
        return f"No existe el archivo de agrupamiento para {alcaldia}", 404

    with open(ruta_json, 'r', encoding='utf-8') as file:
        mejor_agrupamiento = json.load(file)
    if isinstance(mejor_agrupamiento, list):
        mejor_agrupamiento = {i: grp for i, grp in enumerate(mejor_agrupamiento)}

    # 2) Cargar el shapefile correspondiente a la alcaldía
    ruta_shp = os.path.join(app.root_path, 'Alcaldías', f'{alcaldia}.shp')
    if not os.path.exists(ruta_shp):
        return f"No existe el shapefile para {alcaldia}", 404
    gdf_ageb = gpd.read_file(ruta_shp)
    if gdf_ageb.empty:
        return "El shapefile está vacío.", 400

    # 3) Generar sectores: cada sector se exporta como un shapefile independiente en "temp/sectores/<alcaldia>/"
    carpeta_sectores_alcaldia = os.path.join(app.root_path, 'temp', 'sectores', alcaldia)
    os.makedirs(carpeta_sectores_alcaldia, exist_ok=True)
    grupos_dict = {}
    for idx, row in gdf_ageb.iterrows():
        if idx not in mejor_agrupamiento:
            continue
        grupo = mejor_agrupamiento[idx]
        if grupo not in grupos_dict:
            grupos_dict[grupo] = []
        grupos_dict[grupo].append(row)
    for grupo_id, rows in grupos_dict.items():
        subset_gdf = gpd.GeoDataFrame(rows, crs=gdf_ageb.crs)
        subset_gdf['grupo'] = grupo_id
        ruta_sector = os.path.join(carpeta_sectores_alcaldia, f"Sector_{grupo_id}.shp")
        subset_gdf.to_file(ruta_sector, driver='ESRI Shapefile')

    # 4) Preparar la imagen resultante del agrupamiento para mostrarla en el template
    carpeta_imagenes = os.path.join(app.root_path, 'temp', 'imagenes')
    nombre_imagen = f"{alcaldia}_agrupamiento.png"
    ruta_imagen = os.path.join(carpeta_imagenes, nombre_imagen)
    if not os.path.exists(ruta_imagen):
        return f"No se encontró la imagen de agrupamiento para {alcaldia}", 404

    # 5) Renderizar el template "mostrar_resultados.html" pasando la imagen y el número de grupos obtenidos
    return render_template(
        'mostrar_resultados.html',
        alcaldia=alcaldia,
        num_grupos=len(grupos_dict),
        imagen_url=url_for('ver_imagen', alcaldia=alcaldia)
    )


@app.route('/ver_imagen/<alcaldia>')
def ver_imagen(alcaldia):
    """
    Sirve la imagen del agrupamiento almacenada en "temp/imagenes" utilizando send_from_directory.
    """
    carpeta_imagenes = os.path.join(app.root_path, 'temp', 'imagenes')
    nombre_imagen = f"{alcaldia}_agrupamiento.png"
    return send_from_directory(carpeta_imagenes, nombre_imagen)


@app.route('/exportar_sectores/<alcaldia>', methods=['POST'])
def exportar_sectores(alcaldia):
    """
    Utiliza la clase ExportarSectoresAGEB para exportar cada sector en un shapefile independiente.
    
    Se definen las rutas de entrada (archivo JSON del agrupamiento y shapefile de la alcaldía)
    y la ruta de salida para guardar los sectores exportados. Ante error se retorna un mensaje apropiado.
    """
    ruta_agrupamiento = os.path.join(app.root_path, 'temp', 'agrupamiento', f"{alcaldia}_mejor_agrupamiento.json")
    ruta_shapefile = os.path.join(app.root_path, 'Alcaldías', f"{alcaldia}.shp")
    directorio_salida = os.path.join(app.root_path, 'temp', 'sectores_exportados', alcaldia)
    
    if not os.path.exists(ruta_agrupamiento):
        return f"No se encontró el archivo JSON de agrupamiento para {alcaldia}", 404
    if not os.path.exists(ruta_shapefile):
        return f"No se encontró el shapefile para {alcaldia}", 404
    
    try:
        exportador = ExportarSectoresAGEB(
            ruta_agrupamiento=ruta_agrupamiento,
            ruta_shapefile=ruta_shapefile,
            directorio_salida=directorio_salida
        )
        exportador.exportar_sectores()
        flash("Sectores exportados correctamente.", "success")
    except Exception as e:
        return f"Error al exportar sectores: {e}", 500

    return render_template('exportacion_exitosa.html', alcaldia=alcaldia)


@app.route('/seleccionar_sector/<alcaldia>', methods=['GET'])
def seleccionar_sector(alcaldia):
    """
    Carga los sectores (shapefiles) previamente generados para la alcaldía,
    los convierte a GeoJSON y renderiza un mapa interactivo para que el usuario
    seleccione el sector a procesar haciendo clic sobre su polígono.
    """
    carpeta_sectores = os.path.join(app.root_path, 'temp', 'sectores', alcaldia)
    sectores = []
    
    if not os.path.exists(carpeta_sectores):
        return f"No se encontraron sectores para {alcaldia}.", 404

    # Se recorre cada archivo shapefile en la carpeta de sectores y se convierte a GeoJSON.
    for archivo in os.listdir(carpeta_sectores):
        if archivo.endswith('.shp'):
            ruta_sector = os.path.join(carpeta_sectores, archivo)
            try:
                gdf_sector = gpd.read_file(ruta_sector)
                geojson_sector = gdf_sector.to_json()
                # Se utiliza el nombre del archivo sin extensión como identificador del sector.
                sector_id = os.path.splitext(archivo)[0]
                sectores.append({
                    'id': sector_id,
                    'geojson': geojson_sector
                })
            except Exception as e:
                print(f"Error al leer {ruta_sector}: {e}")
                continue

    return render_template('seleccionar_sector.html', alcaldia=alcaldia, sectores=sectores)


@app.route('/procesar_sector/<alcaldia>', methods=['POST'])
@app.route('/procesar_sector/<alcaldia>', methods=['POST'])
def procesar_sector(alcaldia):
    """
    Ruta que procesa el sector seleccionado sin modificar la clase ProcesadorCalles.
    Se define una función adicional (procesar_sector_sin_clase) que recorta la red vial
    usando la geometría del shapefile del sector elegido.
    """
    sector_id = request.form.get('sector_id')
    if not sector_id:
        return "No se ha seleccionado ningún sector.", 400

    # Rutas de los shapefiles de la red vial (aristas y nodos)
    aristas_cdmx_shp = os.path.join(app.root_path, 'data', 'CDMX_aristas.shp')
    nodos_cdmx_shp   = os.path.join(app.root_path, 'data', 'CDMX_nodos.shp')

    # Carpeta donde se ubican los shapefiles de cada sector
    carpeta_sectores = os.path.join(app.root_path, 'temp', 'sectores', alcaldia)
    # Directorios de salida para el recorte
    carpeta_calles = os.path.join(app.root_path, 'temp', 'calles_recortadas', alcaldia)
    carpeta_nodos  = os.path.join(app.root_path, 'temp', 'nodos_recortados', alcaldia)

    # Shapefile del sector seleccionado
    ruta_sector = os.path.join(carpeta_sectores, f"{sector_id}.shp")
    if not os.path.exists(ruta_sector):
        return f"No se encontró el shapefile para el sector '{sector_id}'.", 404

    # Asegurar que los directorios de salida existan
    os.makedirs(carpeta_calles, exist_ok=True)
    os.makedirs(carpeta_nodos, exist_ok=True)

    def procesar_sector_sin_clase(aristas_shp, nodos_shp, sector_shp, out_calles, out_nodos, sector_id):
        """
        Procesa (recorta) la red vial utilizando la geometría del shapefile del sector seleccionado,
        sin depender de la clase ProcesadorCalles.

        :param aristas_shp: Ruta al shapefile con las aristas de la red vial.
        :param nodos_shp:   Ruta al shapefile con los nodos de la red vial.
        :param sector_shp:  Ruta al shapefile del sector seleccionado.
        :param out_calles:  Carpeta donde se guardarán las aristas recortadas.
        :param out_nodos:   Carpeta donde se guardarán los nodos recortados.
        :param sector_id:   Identificador del sector.
        :return: (bool, str) -> Indica si fue exitoso y un mensaje de estado.
        """
        try:
            # 1. Cargar la red vial (aristas y nodos)
            gdf_aristas = gpd.read_file(aristas_shp)
            gdf_nodos   = gpd.read_file(nodos_shp)

            # 2. Cargar la geometría del sector
            gdf_sector  = gpd.read_file(sector_shp)

            # 3. Unir todas las geometrías del sector en una sola (por si tiene varias features)
            poligono_sector = gdf_sector.unary_union

            # 4. Recortar las aristas que intersectan con el polígono del sector
            aristas_recortadas = gdf_aristas[gdf_aristas.intersects(poligono_sector)]

            # 5. Recortar los nodos que están dentro del polígono
            nodos_recortados   = gdf_nodos[gdf_nodos.within(poligono_sector)]

            # 6. Guardar los resultados en shapefiles nuevos
            aristas_salida = os.path.join(out_calles, f"{sector_id}_calles.shp")
            nodos_salida   = os.path.join(out_nodos, f"{sector_id}_nodos.shp")
            aristas_recortadas.to_file(aristas_salida)
            nodos_recortados.to_file(nodos_salida)

            return (True, f"Sector '{sector_id}' recortado exitosamente.")
        except Exception as e:
            return (False, f"Error interno: {e}")

    # Invocar la función de procesado independiente
    resultado, mensaje = procesar_sector_sin_clase(
        aristas_shp=aristas_cdmx_shp,
        nodos_shp=nodos_cdmx_shp,
        sector_shp=ruta_sector,
        out_calles=carpeta_calles,
        out_nodos=carpeta_nodos,
        sector_id=sector_id
    )

    if not resultado:
        return f"Error al procesar el sector '{sector_id}': {mensaje}", 500

    flash(mensaje, "success")
    return render_template('procesamiento_exitoso.html', alcaldia=alcaldia)

if __name__ == '__main__':
    app.run(debug=True, port=5005)