import os
import json
import geopandas as gpd
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory, session, flash
from RecolectarBasura import *

app = Flask(__name__)
app.secret_key = os.urandom(32)

@app.route('/')
def index():
    """
    Redirige a la lista de alcaldías.
    """
    return redirect(url_for('listar_alcaldias'))


@app.route('/Alcaldías', methods=['GET'])
def listar_alcaldias():
    """
    Lista las alcaldías disponibles leyendo .shp en la carpeta Alcaldías/.
    Renderiza la plantilla Alcaldías.html, que muestra un <select> con las opciones.
    """
    alcaldias_dir = os.path.join(app.root_path, 'Alcaldías')
    alcaldias = []
    if os.path.exists(alcaldias_dir):
        for archivo in os.listdir(alcaldias_dir):
            if archivo.endswith('.shp'):
                # Se extrae el nombre sin extensión
                nombre_alcaldia = os.path.splitext(archivo)[0]
                alcaldias.append(nombre_alcaldia)
    else:
        return jsonify({'error': 'No se encontró la carpeta de alcaldías.'}), 404

    seleccionada = session.get('alcaldia', None)

    return render_template('Alcaldías.html', alcaldías=alcaldias, seleccionada=seleccionada)


@app.route('/seleccionar_alcaldia', methods=['POST'])
def seleccionar_alcaldia():
    """
    Recibe la alcaldía seleccionada del formulario y la guarda en session
    para volver a renderizar la página con la alcaldía elegida.
    """
    alcaldia = request.form.get('alcaldía')
    if not alcaldia:
        return redirect(url_for('listar_alcaldias'))

    session['alcaldia'] = alcaldia
    # Regresamos a /Alcaldías (GET) para que muestre los camiones y parámetros
    return redirect(url_for('listar_alcaldias'))


@app.route('/Alcaldías/<alcaldia>/progreso_agrupamiento', methods=['POST'])
def progreso_agrupamiento(alcaldia):
    """
    Ejecuta el proceso de agrupamiento con la clase AgrupamientoAGEB y 
    guarda los resultados (JSON + imagen) en temp/.
    Luego redirige a mostrar_resultados/<alcaldia>.
    """

    # 1) Leer parámetros del formulario
    tamano_poblacion = request.form.get('agrupamiento[tamano_poblacion]', type=int)
    num_generaciones = request.form.get('agrupamiento[num_generaciones]', type=int)
    tasa_mutacion    = request.form.get('agrupamiento[tasa_mutacion]', type=float)
    factor_basura    = request.form.get('agrupamiento[factor_basura]', type=float)

    # 2) Parsear camiones
    camiones = []
    i = 0
    while True:
        nombre_key      = f"camiones[{i}][nombre]"
        capacidad_key   = f"camiones[{i}][capacidad]"
        reserva_key     = f"camiones[{i}][factor_reserva]"
        cantidad_key    = f"camiones[{i}][cantidad_camiones]"
        if nombre_key in request.form:
            # Convertimos valores
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

    # 3) Validar que existan todos los parámetros y al menos un camión
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

    # 6) Ejecutar el agrupamiento
    mejor_individuo, grupos, pesos_grupos = agrupador.ejecutar_agrupamiento()

    # 7) Asignar camiones
    asignaciones, camiones_restantes = agrupador.asignar_camiones(grupos, pesos_grupos)

    # 8) Post-procesar
    nuevos_grupos, nuevos_pesos, nuevas_asignaciones = agrupador.post_procesar_asignacion(
        grupos, pesos_grupos, asignaciones, camiones_restantes
    )

    # 9) Graficar y guardar la imagen
    carpeta_imagenes = os.path.join(app.root_path, 'temp', 'imagenes')
    os.makedirs(carpeta_imagenes, exist_ok=True)
    ruta_img = os.path.join(carpeta_imagenes, f"{alcaldia}_agrupamiento.png")
    agrupador.graficar_con_camiones(nuevos_grupos, nuevas_asignaciones, output_path=ruta_img)

    # 10) Guardar el mejor_individuo en temp/agrupamiento
    carpeta_agrupamiento = os.path.join(app.root_path, 'temp', 'agrupamiento')
    os.makedirs(carpeta_agrupamiento, exist_ok=True)
    ruta_json = os.path.join(carpeta_agrupamiento, f"{alcaldia}_mejor_agrupamiento.json")
    agrupador.guardar_resultados(mejor_individuo, ruta_json)

    # 11) Redirigir a mostrar_resultados
    return redirect(url_for('mostrar_resultados', alcaldia=alcaldia))


@app.route('/mostrar_resultados/<alcaldia>', methods=['GET'])
def mostrar_resultados(alcaldia):
    """
    Carga el JSON de agrupamiento, parte el shapefile en sectores
    y muestra la imagen final en la plantilla 'mostrar_resultados.html'.
    """
    # 1) Ruta JSON
    carpeta_agrupamiento = os.path.join(app.root_path, 'temp', 'agrupamiento')
    ruta_json = os.path.join(carpeta_agrupamiento, f"{alcaldia}_mejor_agrupamiento.json")
    if not os.path.exists(ruta_json):
        return f"No existe el archivo de agrupamiento para {alcaldia}", 404

    with open(ruta_json, 'r', encoding='utf-8') as file:
        mejor_agrupamiento = json.load(file)
    if isinstance(mejor_agrupamiento, list):
        mejor_agrupamiento = {i: grp for i, grp in enumerate(mejor_agrupamiento)}

    # 2) Cargar shapefile
    ruta_shp = os.path.join(app.root_path, 'Alcaldías', f'{alcaldia}.shp')
    if not os.path.exists(ruta_shp):
        return f"No existe el shapefile para {alcaldia}", 404

    gdf_ageb = gpd.read_file(ruta_shp)
    if gdf_ageb.empty:
        return "El shapefile está vacío.", 400

    # 3) Generar sectores en /temp/sectores/<alcaldia>/
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

    # 4) Imagen
    carpeta_imagenes = os.path.join(app.root_path, 'temp', 'imagenes')
    nombre_imagen = f"{alcaldia}_agrupamiento.png"
    ruta_imagen = os.path.join(carpeta_imagenes, nombre_imagen)
    if not os.path.exists(ruta_imagen):
        return f"No se encontró la imagen de agrupamiento para {alcaldia}", 404

    # 5) Renderizar template con la imagen
    return render_template(
        'mostrar_resultados.html',
        alcaldia=alcaldia,
        num_grupos=len(grupos_dict),
        imagen_url=url_for('ver_imagen', alcaldia=alcaldia)
    )


@app.route('/ver_imagen/<alcaldia>')
def ver_imagen(alcaldia):
    """
    Sirve la imagen desde la carpeta temp/imagenes usando send_from_directory.
    """
    carpeta_imagenes = os.path.join(app.root_path, 'temp', 'imagenes')
    nombre_imagen = f"{alcaldia}_agrupamiento.png"
    return send_from_directory(carpeta_imagenes, nombre_imagen)

@app.route('/exportar_sectores/<alcaldia>', methods=['POST'])
def exportar_sectores(alcaldia):
    """
    Ruta que usa la clase ExportarSectoresAGEB para exportar cada sector en un shapefile independiente.
    """
    # Definir las rutas de entrada y salida
    ruta_agrupamiento = os.path.join(app.root_path, 'temp', 'agrupamiento', f"{alcaldia}_mejor_agrupamiento.json")
    ruta_shapefile = os.path.join(app.root_path, 'Alcaldías', f"{alcaldia}.shp")
    directorio_salida = os.path.join(app.root_path, 'temp', 'sectores_exportados', alcaldia)
    
    # Verificar que los archivos o directorios existen (opcional)
    if not os.path.exists(ruta_agrupamiento):
        return f"No se encontró el archivo JSON de agrupamiento para {alcaldia}", 404
    if not os.path.exists(ruta_shapefile):
        return f"No se encontró el shapefile para {alcaldia}", 404
    
    # Crear la instancia y ejecutar la exportación
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

    # Renderiza un template que informe que la exportación fue exitosa y muestre el botón para procesar calles
    return render_template('exportacion_exitosa.html', alcaldia=alcaldia)

@app.route('/procesar_calles/<alcaldia>', methods=['POST'])
def procesar_calles(alcaldia):
    """
    Ruta que invoca la clase ProcesadorCalles para recortar la red vial de la Ciudad de México
    y corregir la conectividad según los sectores.
    """
    # Definir rutas de entrada para la red vial (estas rutas pueden variar según su estructura)
    aristas_cdmx_shp = os.path.join(app.root_path, 'data', 'aristas_cdmx.shp')
    nodos_cdmx_shp = os.path.join(app.root_path, 'data', 'nodos_cdmx.shp')
    
    # Usar los sectores previamente exportados
    carpeta_sectores = os.path.join(app.root_path, 'temp', 'sectores', alcaldia)
    
    # Directorios para guardar los shapefiles recortados y los finales corregidos
    carpeta_calles = os.path.join(app.root_path, 'temp', 'calles_recortadas', alcaldia)
    carpeta_nodos = os.path.join(app.root_path, 'temp', 'nodos_recortados', alcaldia)
    carpeta_salida_calles = os.path.join(app.root_path, 'temp', 'calles_finales', alcaldia)
    carpeta_salida_nodos = os.path.join(app.root_path, 'temp', 'nodos_finales', alcaldia)
    
    # Instanciar la clase ProcesadorCalles
    try:
        procesador = ProcesadorCalles(
            aristas_cdmx_shp=aristas_cdmx_shp,
            nodos_cdmx_shp=nodos_cdmx_shp,
            carpeta_sectores=carpeta_sectores,
            carpeta_calles=carpeta_calles,
            carpeta_nodos=carpeta_nodos,
            carpeta_salida_calles=carpeta_salida_calles,
            carpeta_salida_nodos=carpeta_salida_nodos
        )
        # Método para recortar la gráfica según los sectores
        procesador.recortar_grafica_por_sectores()

    except Exception as e:
        return f"Error al procesar la red vial: {e}", 500

    flash("La red vial fue procesada correctamente.", "success")
    return render_template('procesamiento_exitoso.html', alcaldia=alcaldia)

if __name__ == '__main__':
    app.run(debug=True, port=5005)