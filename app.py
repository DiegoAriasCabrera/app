import os
import json
import uuid
import time
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, Response
from RecolectarBasura import Camion, AgrupamientoAGEB, ProcesadorCalles
from tqdm import tqdm
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "TU_SUPER_SECRETO"


@app.route('/')
def index():
    """
    Muestra la lista de alcaldías, pero ahora tenemos la lógica en /Alcaldías
    o podrías redirigir a una página 'index.html' con botones.
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
        return redirect(url_for('pagina_sse', alcaldia=alcaldia))

    # Renderiza la plantilla con la alcaldía seleccionada
    # (El "alcaldías=[]" es para que no se muestren de nuevo en el select)
    return render_template('Alcaldías.html', alcaldías=[], seleccionada=alcaldia)


@app.route('/Alcaldías/<alcaldia>/sse')
def pagina_sse(alcaldia):
    """
    Esta ruta renderiza un HTML (index.html o algo similar)
    que abrirá EventSource("/Alcaldías/<alcaldia>/progreso_agrupamiento") 
    y mostrará las barras de progreso de ambos procesos.
    """
    return render_template('index.html', alcaldia=alcaldia)


@app.route('/Alcaldías/<alcaldia>/progreso_agrupamiento')
def progreso_agrupamiento(alcaldia):
    """
    1) Recoge datos de session o algo similar (o se los pasa en param),
    2) Ejecuta AgrupamientoAGEB con tqdm, enviando SSE.
    3) Al terminar, manda "DONE" y genera su imagen en /temp.
    """
    # Para hacerlo simple, los datos del formulario que enviaste 
    # están en request.form. Pero no los tenemos aquí porque se hace un redirect...
    # Solución: guardamos en session los datos al POST.
    # O leemos en la DB. Aquí haremos un "hack": re-leer request.form. 
    # Pero no es lo usual. Lo correcto es guardarlos en session antes del redirect.

    # Vamos a simular la lectura de form con session (opcional).
    # O, si tu formulario era un POST, lo perdiste en el redirect. 
    # Lo más fácil: guardaste la data en session en 'agregar_camiones' (no implementado aquí).

    # Haré un ejemplo con datos fijos.
    # En la realidad, obtendrías: tamano_poblacion, num_generaciones, etc. de tu session/form.
    tamano_poblacion = 50
    num_generaciones = 10
    tasa_mutacion    = 0.01
    factor_basura    = 1.071
    limite_peso      = None
    reconectar_grupos= True
    semilla_random   = None

    # Camiones fake:
    camiones = [Camion(1000, 1.1, 2, "Camión A")]

    shapefile_path = os.path.join(app.root_path, 'Alcaldías', f'{alcaldia}.shp')
    if not os.path.exists(shapefile_path):
        def gen_err():
            yield "data: ERROR\n\n"
        return Response(gen_err(), mimetype="text/event-stream")

    # Empleamos SSE + tqdm
    def generate():
        # 1) Instanciamos el agrupador
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

        # 2) Vamos a reescribir (temporalmente) el loop genético con tqdm
        #    para mandar SSE. En tu clase tienes:
        #    for _ in tqdm(range(self.num_generaciones)):
        #       poblacion, ind, fitness = self._evolucionar_poblacion(poblacion)
        #    ...
        #    => Lo haremos inline.

        # Creamos la población inicial
        poblacion = agrupador._crear_poblacion()
        mejor_fitness = float('-inf')
        mejor_individuo = None

        for i in tqdm(range(num_generaciones), desc="Agrupando"):
            poblacion, ind, fitness = agrupador._evolucionar_poblacion(poblacion)
            if fitness > mejor_fitness:
                mejor_fitness = fitness
                mejor_individuo = ind

            # Mandamos SSE con un porcentaje
            porc = int((i+1) * 100 / num_generaciones)
            yield f"data: {porc}\n\n"
            time.sleep(0.1)  # simulamos retardo

        if not mejor_individuo:
            yield "data: ERROR\n\n"
            return

        # Creamos la asignación final
        grupos, pesos_grupos = {}, {}
        for nodo, id_grupo in enumerate(mejor_individuo):
            grupos.setdefault(id_grupo, []).append(nodo)
            pesos_grupos[id_grupo] = pesos_grupos.get(id_grupo, 0.0) + agrupador.gráfica.nodes[nodo]['peso']

        asignaciones, camiones_restantes = agrupador.asignar_camiones(grupos, pesos_grupos)

        # Guardar un JSON
        resultado_agrupamiento_dir = os.path.join(app.root_path, 'resultado_agrupamiento')
        os.makedirs(resultado_agrupamiento_dir, exist_ok=True)
        json_filename = f"agrupamiento_{alcaldia}_{uuid.uuid4().hex}.json"
        json_path = os.path.join(resultado_agrupamiento_dir, json_filename)
        resultado = {
            "alcaldia": alcaldia,
            "mejor_individuo": mejor_individuo,
            "num_generaciones": num_generaciones,
            "asignaciones": {str(g): (camion.nombre if camion else None)
                             for g, camion in asignaciones.items()}
        }
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(resultado, jf, ensure_ascii=False, indent=2)

        # Generar imagen
        image_filename = f"agrupamiento_{alcaldia}_{uuid.uuid4().hex}.png"
        image_path = os.path.join(resultado_agrupamiento_dir, image_filename)

        # Si tu clase ya soporta output_path:
        agrupador.graficar_con_camiones(grupos, asignaciones, output_path=image_path)

        # Guardamos la info en session para que la segunda SSE (procesadorCalles) sepa 
        session['agrup_img'] = image_filename

        # Al terminar, enviamos "DONE"
        yield "data: DONE\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route('/Alcaldías/<alcaldia>/progreso_calles')
def progreso_calles(alcaldia):
    """
    Segundo proceso (ProcesadorCalles), también largo.
    Usa SSE + tqdm, y al final manda "DONE2" y genera la imagen en /temp.
    """
    def generate():
        # Vamos a simular 10 iteraciones
        total = 10

        # Instanciamos algo de tu clase:
        temp_dir = os.path.join(app.root_path, "temp")
        aristas_global = os.path.join(app.root_path, "CDMX_aristas.shp")
        nodos_global   = os.path.join(app.root_path, "CDMX_nodos.shp")
        carpeta_sectores = os.path.join(app.root_path, "sectores")

        if not os.path.exists(aristas_global):
            yield "data: ERROR\n\n"
            return

        pc = ProcesadorCalles(
            aristas_cdmx_shp=aristas_global,
            nodos_cdmx_shp=nodos_global,
            carpeta_sectores=carpeta_sectores,
            carpeta_salida_calles=os.path.join(temp_dir,"calles_salida"),
            carpeta_salida_nodos=os.path.join(temp_dir,"nodos_salida"),
            carpeta_salida_final=os.path.join(temp_dir,"calles_final")
        )

        # Simulamos un método largo recortar_red_vial_por_sectores
        # y mientras tanto enviamos SSE:
        for i in tqdm(range(total), desc="Calles"):
            time.sleep(0.3)
            porc = int((i+1)*100/total)
            yield f"data: {porc}\n\n"

        calles_img_filename = f"calles_{uuid.uuid4().hex}.png"
        calles_img_path = os.path.join(temp_dir, calles_img_filename)

        plt.figure()
        plt.title("Resultado ProcesadorCalles")
        plt.plot([0,1,2],[2,1,2],'g-')
        plt.savefig(calles_img_path)
        plt.close()

        session['calles_img'] = calles_img_filename

        yield "data: DONE2\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route('/final')
def final():
    """
    Muestra la página final con ambas imágenes:
    la del agrupamiento y la de calles.
    """
    agr = session.get('agrup_img')
    cal = session.get('calles_img')
    return render_template(
        'resultado_final.html',
        alcaldia='DESCONOCIDA',
        image_agrup=f"temp/{agr}" if agr else None,
        image_calles=f"temp/{cal}" if cal else None
    )


if __name__ == '__main__':
    # En modo debug no hay timeout. Para producción:
    # gunicorn --timeout 600 app:app
    app.run(debug=True, port=5005)