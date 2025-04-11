import os
import shutil
import zipfile
import json
import geopandas as gpd
import networkx as nx
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory, session, flash
from RecolectarBasura import *

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
    carpeta_agrupamiento = os.path.join(app.root_path, 'temp', 'agrupamiento')
    ruta_json = os.path.join(carpeta_agrupamiento, f"{alcaldia}_mejor_agrupamiento.json")
    if not os.path.exists(ruta_json):
        return f"No existe el archivo de agrupamiento para {alcaldia}", 404

    with open(ruta_json, 'r', encoding='utf-8') as file:
        mejor_agrupamiento = json.load(file)
    if isinstance(mejor_agrupamiento, list):
        mejor_agrupamiento = {i: grp for i, grp in enumerate(mejor_agrupamiento)}

    # Cargar el shapefile de la alcaldía
    ruta_shp = os.path.join(app.root_path, 'Alcaldías', f'{alcaldia}.shp')
    if not os.path.exists(ruta_shp):
        return f"No existe el shapefile para {alcaldia}", 404
    gdf_ageb = gpd.read_file(ruta_shp)
    if gdf_ageb.empty:
        return "El shapefile está vacío.", 400

    # Generar sectores como shapefiles independientes en "temp/sectores/<alcaldia>/"
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
        ruta_sector_out = os.path.join(carpeta_sectores_alcaldia, f"Sector_{grupo_id}.shp")
        subset_gdf.to_file(ruta_sector_out, driver='ESRI Shapefile')

    carpeta_imagenes = os.path.join(app.root_path, 'temp', 'imagenes')
    nombre_imagen = f"{alcaldia}_agrupamiento.png"
    ruta_imagen = os.path.join(carpeta_imagenes, nombre_imagen)
    if not os.path.exists(ruta_imagen):
        return f"No se encontró la imagen de agrupamiento para {alcaldia}", 404

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
    directorio_salida = os.path.join(app.root_path, 'temp', 'sectores', alcaldia)
    
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

    for archivo in os.listdir(carpeta_sectores):
        if archivo.endswith('.shp'):
            ruta_sector = os.path.join(carpeta_sectores, archivo)
            try:
                gdf_sector = gpd.read_file(ruta_sector)
                geojson_sector = gdf_sector.to_json()
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
def procesar_sector(alcaldia):
    """
    Ruta que procesa el sector seleccionado:
    Primero recorta la red vial (calles y nodos) solamente para ese sector y,
    posteriormente, aplica la corrección de conectividad para el sector.
    
    Se utiliza una función independiente para recortar el sector y otra para la corrección de conectividad,
    sin modificar la clase ProcesadorCalles.
    """
    sector_id = request.form.get('sector_id')
    if not sector_id:
        return "No se ha seleccionado ningún sector.", 400

    # Rutas de los shapefiles de la red vial (aristas y nodos) completos
    aristas_cdmx_shp = os.path.join(app.root_path, 'data', 'CDMX_aristas.shp')
    nodos_cdmx_shp   = os.path.join(app.root_path, 'data', 'CDMX_nodos.shp')

    # Carpetas de entrada (shapefiles de sectores) y salida para los resultados intermedios
    carpeta_sectores = os.path.join(app.root_path, 'temp', 'sectores', alcaldia)
    carpeta_calles   = os.path.join(app.root_path, 'temp', 'calles_recortadas', alcaldia)
    carpeta_nodos    = os.path.join(app.root_path, 'temp', 'nodos_recortados', alcaldia)

    # Shapefile del sector seleccionado
    ruta_sector = os.path.join(carpeta_sectores, f"{sector_id}.shp")
    if not os.path.exists(ruta_sector):
        return f"No se encontró el shapefile para el sector '{sector_id}'.", 404

    # Asegurar que los directorios de salida existan
    os.makedirs(carpeta_calles, exist_ok=True)
    os.makedirs(carpeta_nodos, exist_ok=True)

    def recortar_sector(aristas_shp, nodos_shp, sector_shp, out_calles, out_nodos, sector_id):
        """
        Recorta la red vial utilizando la geometría del sector seleccionado.
        
        :param aristas_shp: Ruta al shapefile con las aristas de la red vial completa.
        :param nodos_shp:   Ruta al shapefile con los nodos de la red vial completa.
        :param sector_shp:  Ruta al shapefile del sector seleccionado.
        :param out_calles:  Carpeta de salida para las aristas recortadas.
        :param out_nodos:   Carpeta de salida para los nodos recortados.
        :param sector_id:   Identificador del sector.
        :return: (bool, str) indicando éxito y mensaje.
        """
        try:
            gdf_aristas = gpd.read_file(aristas_shp)
            gdf_nodos   = gpd.read_file(nodos_shp)
            gdf_sector  = gpd.read_file(sector_shp)
            
            # Unir las geometrías del sector en un solo polígono
            poligono_sector = gdf_sector.unary_union
            
            # Recortar aristas y nodos de acuerdo con el polígono del sector
            aristas_recortadas = gdf_aristas[gdf_aristas.intersects(poligono_sector)]
            nodos_recortados   = gdf_nodos[gdf_nodos.within(poligono_sector)]
            
            aristas_salida = os.path.join(out_calles, f"{sector_id}_calles.shp")
            nodos_salida   = os.path.join(out_nodos, f"{sector_id}_nodos.shp")
            aristas_recortadas.to_file(aristas_salida)
            nodos_recortados.to_file(nodos_salida)
            
            return (True, f"Sector '{sector_id}' recortado exitosamente.", aristas_salida, nodos_salida)
        except Exception as e:
            return (False, f"Error al recortar el sector '{sector_id}': {e}", None, None)

    def corregir_conectividad_sector(nodos_shp, aristas_shp, out_nodos, out_calles, sector_id):
        """
        Corrige la conectividad de la red vial recortada para el sector seleccionado.
        Esta función carga los shapefiles recortados, construye una gráfica usando NetworkX,
        y conserva la componente conexa más grande. Los resultados se guardan en directorios finales.
        
        :param nodos_shp: Ruta del shapefile de nodos recortados.
        :param aristas_shp: Ruta del shapefile de aristas recortadas.
        :param out_nodos: Carpeta de salida para los nodos finales.
        :param out_calles: Carpeta de salida para las aristas finales.
        :param sector_id: Identificador del sector.
        :return: (bool, str) indicando éxito y mensaje.
        """
        try:
            gdf_nodos = gpd.read_file(nodos_shp)
            gdf_aristas = gpd.read_file(aristas_shp)
            
            # Crear una gráfica no dirigida (para conectividad)
            G = nx.Graph()
            # Agregar nodos: se supone que la columna 'osmid' identifica los nodos
            for idx, node in gdf_nodos.iterrows():
                G.add_node(node['osmid'], geometry=node.geometry)
            # Agregar aristas: se asume que las columnas 'from' y 'to' existen en el shapefile de aristas
            for idx, row in gdf_aristas.iterrows():
                if 'from' in row and 'to' in row:
                    G.add_edge(row['from'], row['to'], geometry=row.geometry)
            
            # Si la gráfica no es conexa, conservar solo la componente más grande.
            if not nx.is_connected(G):
                components = list(nx.connected_components(G))
                largest_component = max(components, key=len)
                gdf_nodos_corr = gdf_nodos[gdf_nodos['osmid'].isin(largest_component)]
                gdf_aristas_corr = gdf_aristas[
                    gdf_aristas.apply(lambda row: row['from'] in largest_component and row['to'] in largest_component, axis=1)
                ]
            else:
                gdf_nodos_corr = gdf_nodos
                gdf_aristas_corr = gdf_aristas
            
            nodos_salida = os.path.join(out_nodos, f"{sector_id}_nodos_finales.shp")
            aristas_salida = os.path.join(out_calles, f"{sector_id}_calles_finales.shp")
            gdf_nodos_corr.to_file(nodos_salida)
            gdf_aristas_corr.to_file(aristas_salida)
            
            return (True, f"Conectividad corregida para sector '{sector_id}'.")
        except Exception as e:
            return (False, f"Error en la corrección de conectividad para sector '{sector_id}': {e}")

    # Ejecutar el recorte del sector
    resultado_recorte, mensaje_recorte, aristas_recortadas_path, nodos_recortados_path = recortar_sector(
        aristas_shp=aristas_cdmx_shp,
        nodos_shp=nodos_cdmx_shp,
        sector_shp=ruta_sector,
        out_calles=carpeta_calles,
        out_nodos=carpeta_nodos,
        sector_id=sector_id
    )
    if not resultado_recorte:
        return f"Error al procesar el sector '{sector_id}': {mensaje_recorte}", 500

    carpeta_salida_calles_finales = os.path.join(app.root_path, 'temp', 'calles_finales', alcaldia)
    carpeta_salida_nodos_finales  = os.path.join(app.root_path, 'temp', 'nodos_finales', alcaldia)
    os.makedirs(carpeta_salida_calles_finales, exist_ok=True)
    os.makedirs(carpeta_salida_nodos_finales, exist_ok=True)
    resultado_conectividad, mensaje_conectividad = corregir_conectividad_sector(
        nodos_shp=nodos_recortados_path,
        aristas_shp=aristas_recortadas_path,
        out_nodos=carpeta_salida_nodos_finales,
        out_calles=carpeta_salida_calles_finales,
        sector_id=sector_id
    )
    if not resultado_conectividad:
        return f"Error al corregir la conectividad del sector '{sector_id}': {mensaje_conectividad}", 500

    flash(f"{mensaje_recorte} {mensaje_conectividad}", "success")
    # Se renderiza el nuevo template 'resolver_tsp.html', enviando la alcaldía y el sector procesado.
    return render_template('resolver_tsp.html', alcaldia=alcaldia, sector=sector_id)


@app.route('/resolver_tsp/<alcaldia>/<sector>', methods=['POST'])
def resolver_tsp(alcaldia, sector):
    # Ubicar los shapefiles de nodos y aristas para el sector
    nodos_dir = os.path.join(app.root_path, 'temp', 'nodos_finales', alcaldia)
    calles_dir = os.path.join(app.root_path, 'temp', 'calles_finales', alcaldia)
    ruta_nodos = os.path.join(nodos_dir, f"{sector}_nodos_finales.shp")
    ruta_aristas = os.path.join(calles_dir, f"{sector}_calles_finales.shp")

    if not os.path.exists(ruta_nodos) or not os.path.exists(ruta_aristas):
        return f"Shapefile del sector {sector} no encontrado.", 404

    try:
        mtsp_solver = ResolverMTSP(
            ruta_nodos=ruta_nodos,
            ruta_aristas=ruta_aristas,
            tamano_poblacion=int(request.form.get("tsp[tamano_poblacion]")),
            generaciones=int(request.form.get("tsp[num_generaciones]")),
            tasa_mutacion=float(request.form.get("tsp[tasa_mutacion]")),
            tamano_elite=float(request.form.get("tsp[tamano_elite]"))
        )
        mtsp_solver.resolver_MTSP()  # Se ejecuta el algoritmo TSP
        
        # Exportar la ruta generada a shapefiles y crear un ZIP
        export_dir = os.path.join(app.root_path, 'temp', 'resolver_tsp', alcaldia, sector)
        os.makedirs(export_dir, exist_ok=True)
        salida_nodos = os.path.join(export_dir, f"{sector}_ruta_nodos.shp")
        salida_aristas = os.path.join(export_dir, f"{sector}_ruta_aristas.shp")
        mtsp_solver.exportar_ruta_shapefiles(salida_nodos, salida_aristas)
        
        # Comprimir el directorio en un ZIP
        zip_filename = f"{sector}_ruta_shapefiles.zip"
        zip_path = os.path.join(export_dir, zip_filename)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(export_dir):
                for file in files:
                    if file == zip_filename:
                        continue
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_dir)
                    zipf.write(file_path, arcname)
                    
        download_url = url_for('download_resolver_tsp', alcaldia=alcaldia, sector=sector, filename=zip_filename)
        return render_template('resultado_tsp.html',
                               alcaldia=alcaldia,
                               sector=sector,
                               download_url=download_url)
    except Exception as e:
        return f"Error al resolver MTSP: {e}", 500


@app.route('/download_resolver_tsp/<alcaldia>/<sector>/<filename>')
def download_resolver_tsp(alcaldia, sector, filename):
    """
    Sirve el archivo ZIP generado por ResolverMTSP para descarga.
    """
    export_dir = os.path.join(app.root_path, 'temp', 'resolver_tsp', alcaldia, sector)
    return send_from_directory(export_dir, filename, as_attachment=True)


@app.route('/limpiar_temp', methods=['GET'])
def limpiar_temp():
    """
    Elimina todos los archivos de las carpetas que se encuentran en la carpeta 'temp'
    y redirige al listado de alcaldías.
    """
    temp_path = os.path.join(app.root_path, 'temp')
    try:
        for root, dirs, files in os.walk(temp_path):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
    except Exception as e:
        flash(f"Error al limpiar archivos temporales: {e}", "error")
    return redirect(url_for('listar_alcaldias'))


if __name__ == '__main__':
    app.run(debug=True, port=5005)