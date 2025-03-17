import os
import json
import uuid
import time

from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from RecolectarBasura import Camion, AgrupamientoAGEB, ProcesadorCalles

app = Flask(__name__)
app.secret_key = "TU_SUPER_SECRETO"


@app.route('/')
def index():
    # Redirige a /Alcaldías
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
                # Se extrae el nombre sin la extensión
                nombre_alcaldia = os.path.splitext(archivo)[0]
                alcaldias.append(nombre_alcaldia)
    else:
        return jsonify({'error': 'No se encontró la carpeta de alcaldías.'}), 404

    return render_template('Alcaldías.html', alcaldías=alcaldias)


@app.route('/seleccionar_alcaldia', methods=['POST'])
def seleccionar_alcaldia():
    """
    Recibe la alcaldía seleccionada del formulario y redirige 
    a la pantalla donde se configuran camiones y parámetros.
    """
    alcaldia = request.form.get('alcaldía')
    if not alcaldia:
        return redirect(url_for('listar_alcaldias'))
    # Redirige a la ruta donde se piden datos de camiones
    return redirect(url_for('agregar_camiones', alcaldia=alcaldia))


@app.route('/Alcaldías/<alcaldia>/agregar_camiones', methods=['GET', 'POST'])
def agregar_camiones(alcaldia):
    """
    Muestra el mismo template Alcaldías.html (pero con la alcaldía seleccionada) 
    para capturar camiones y parámetros.
    """
    if request.method == 'POST':
        # Si se envía el formulario, redirige a procesar_agrupamiento
        return redirect(url_for('procesar_agrupamiento', alcaldia=alcaldia))

    # Renderiza la plantilla con la alcaldía seleccionada
    # (El "alcaldías=[]" es para que no se muestren de nuevo en el select)
    return render_template('Alcaldías.html', alcaldías=[], seleccionada=alcaldia)


@app.route('/Alcaldías/<alcaldia>/procesar_agrupamiento', methods=['POST'])
def procesar_agrupamiento(alcaldia):
    """
    1) Recoge datos (camiones, parámetros) del formulario 
    2) Ejecuta AgrupamientoAGEB
    3) Guarda JSON y la imagen en carpeta 'temp/'
    4) Llama automáticamente a ProcesadorCalles
    5) Genera la imagen final de calles y la guarda en 'temp/'
    6) Renderiza un 'resultado_final.html' mostrando ambas imágenes.
    """
    # 1) Recoger datos del formulario:
    # El formulario tiene inputs como: camiones[0][nombre], camiones[0][capacidad], etc.
    # y agrupamiento[tamano_poblacion], agrupamiento[num_generaciones], etc.
    
    # OBTENER CAMIONES
    camiones = []
    # La estructura "camiones[x][campo]" sugiere que vendrán varios indices x.
    # Por ejemplo:
    # camiones[0][nombre], camiones[0][capacidad], ...
    # camiones[1][nombre], ...
    # Los enumeramos:
    # Podrías buscar keys en request.form, 
    #   request.form["camiones[0][nombre]"], 
    #   request.form["camiones[0][capacidad]"], etc.
    
    # Para saber cuántas filas de camiones hay, revisamos "camionesContainer" en el HTML.
    # O más simple, buscamos las keys que tengan "camiones[":
    indices_camiones = []
    for key in request.form.keys():
        if key.startswith("camiones[") and key.endswith("[nombre]"):
            # Extraer el índice, p.ej. key="camiones[0][nombre]" -> idx=0
            idx_str = key.split("[")[1].split("]")[0]
            indices_camiones.append(idx_str)
    indices_camiones = sorted(set(indices_camiones))

    for idx_str in indices_camiones:
        base = f"camiones[{idx_str}]"
        nombre  = request.form.get(f"{base}[nombre]", "Camion")
        cap     = float(request.form.get(f"{base}[capacidad]", 0))
        frese   = float(request.form.get(f"{base}[factor_reserva]", 1))
        canti   = int(request.form.get(f"{base}[cantidad_camiones]", 1))
        cam = Camion(
            capacidad=cap,
            factor_reserva=frese,
            cantidad_camiones=canti,
            nombre=nombre
        )
        camiones.append(cam)

    # OBTENER PARÁMETROS
    agrupamiento_data = request.form.get("agrupamiento", None)
    # Pero en tu HTML, usas name="agrupamiento[tamano_poblacion]", etc.
    # Accedemos a cada input por su name:
    tamano_poblacion = int(request.form.get("agrupamiento[tamano_poblacion]", 500))
    num_generaciones = int(request.form.get("agrupamiento[num_generaciones]", 500))
    tasa_mutacion    = float(request.form.get("agrupamiento[tasa_mutacion]", 0.01))
    factor_basura    = float(request.form.get("agrupamiento[factor_basura]", 1.071))

    # Supongamos que "limite_peso" es None y "reconectar_grupos" sea True, etc.
    limite_peso = None
    reconectar_grupos = True
    semilla_random = None

    # Revisa si la alcaldía es válida
    if not alcaldia:
        return jsonify({'error': 'No se especificó la alcaldía.'}), 400

    # Verifica que el shapefile exista
    shapefile_path = os.path.join(app.root_path, 'Alcaldías', f'{alcaldia}.shp')
    if not os.path.exists(shapefile_path):
        return jsonify({'error': f"No se encontró el shapefile para la alcaldía {alcaldia}"}), 404

    # 2) Ejecutar AgrupamientoAGEB
    try:
        agrupador = AgrupamientoAGEB(
            ruta_shp=shapefile_path,
            limite_peso=limite_peso,
            tamano_poblacion=tamano_poblacion,
            num_generaciones=num_generaciones,
            tasa_mutacion=tasa_mutacion,
            factor_basura=factor_basura,
            camiones=camiones,
            reconectar_grupos=reconectar_grupos,
            semilla_random=semilla_random
        )

        mejor_individuo, grupos, pesos_grupos = agrupador.ejecutar_agrupamiento()
        asignaciones, camiones_restantes = agrupador.asignar_camiones(grupos, pesos_grupos)

        # 3) Guardar JSON en 'temp/' (no se devuelve al usuario)
        resultado = {
            "alcaldia": alcaldia,
            "mejor_individuo": mejor_individuo,
            "grupos": grupos,
            "pesos_grupos": pesos_grupos,
            "asignaciones": {
                str(g): (camion.nombre if camion else None)
                for g, camion in asignaciones.items()
            },
            "camiones_restantes": [c.nombre for c in camiones_restantes],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }

        temp_dir = os.path.join(app.root_path, 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        json_filename = f"agrupamiento_{alcaldia}_{uuid.uuid4().hex}.json"
        json_path = os.path.join(temp_dir, json_filename)
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(resultado, jf, ensure_ascii=False, indent=2)

        # 4) Generar la imagen del agrupamiento y guardarla en temp/
        # Modificaremos la clase para que reciba un parámetro extra "output_path" en graficar_con_camiones
        # o simplemente guardamos con plt.savefig(...) en la clase. A continuación la clase no lo hace,
        # pero demos por hecho que ya has modificado la clase y tienes un "graficar_con_camiones(..., output_path)".
        # Por ejemplo:

        # EJEMPLO: supondremos que extends con:
        # def graficar_con_camiones(..., output_path=None):
        #   if output_path:
        #       plt.savefig(output_path)
        #   else:
        #       plt.show()
        #
        image_filename_agrup = f"agrupamiento_{alcaldia}_{uuid.uuid4().hex}.png"
        image_path_agrup = os.path.join(temp_dir, image_filename_agrup)

        # Aquí un ejemplo de guardado (si tu método no lo hace, tendrías que sobrescribirlo).
        # Suponemos que tu AgrupamientoAGEB clase ya lo soporta:
        agrupador.graficar_con_camiones(grupos, asignaciones, output_path=image_path_agrup)

        # 5) AUTOMÁTICAMENTE: invocar ProcesadorCalles
        # Ajusta las rutas a tus shapefiles globales, carpeta sectores, etc.
        # O podrías usar un constructor con variables definidas.
        # Ejemplo:

        aristas_global = os.path.join(app.root_path, "calles_global.shp")    # <-- Asegúrate que exista
        nodos_global   = os.path.join(app.root_path, "nodos_global.shp")     # <-- Asegúrate que exista
        carpeta_sectores = os.path.join(app.root_path, "sectores")           # <-- idem
        carpeta_calles_salida = os.path.join(temp_dir, "calles_salida")
        carpeta_nodos_salida  = os.path.join(temp_dir, "nodos_salida")
        carpeta_final_calles  = os.path.join(temp_dir, "calles_final")

        os.makedirs(carpeta_calles_salida, exist_ok=True)
        os.makedirs(carpeta_nodos_salida, exist_ok=True)
        os.makedirs(carpeta_final_calles, exist_ok=True)

        pc = ProcesadorCalles(
            aristas_cdmx_shp=aristas_global,
            nodos_cdmx_shp=nodos_global,
            carpeta_sectores=carpeta_sectores,
            carpeta_salida_calles=carpeta_calles_salida,
            carpeta_salida_nodos=carpeta_nodos_salida,
            carpeta_salida_final=carpeta_final_calles
        )

        # 6) Ejecutar el método (ejemplo) recortar_red_vial_por_sectores 
        # y luego generamos una imagen. 
        # En tu caso, podrías tener un método que devuelva la "ruta" a una imagen.
        # Aquí, a modo de ejemplo, generamos algo muy sencillo (no implementado en la clase base).
        # Supongamos que has extendido con un método "graficar_calles_final(output_path)"
        # Por simplicidad, sólo llamamos recortar, y no graficamos en la clase original. 
        # Lo harás a tu gusto.

        pc.recortar_red_vial_por_sectores()
        # Imaginemos que produce un .png con un método tuyo:
        calles_image_filename = f"calles_{uuid.uuid4().hex}.png"
        calles_image_path = os.path.join(temp_dir, calles_image_filename)
        # Suponiendo "pc.graficar_calle_result(output_path=calles_image_path)" 
        # (No existe en tu clase actual, deberías implementar algo similar)
        # pc.graficar_calle_result(output_path=calles_image_path)
        #
        # Aquí sólo simulamos con un simple plt.savefig():
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title("Ejemplo de resultado de ProcesadorCalles")
        plt.plot([0,1,2], [3,1,4], 'r-')  # Ejemplo
        plt.savefig(calles_image_path)
        plt.close()

        # 7) Finalmente, renderizamos 'resultado_final.html' para mostrar ambas imágenes
        return render_template(
            'resultado_final.html',
            alcaldia=alcaldia,
            image_agrup=f"temp/{image_filename_agrup}",
            image_calles=f"temp/{calles_image_filename}"
        )

    except Exception as e:
        # Si algo falla, devolvemos un error
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)