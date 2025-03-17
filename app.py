import os
from flask import Flask, request, jsonify, render_template, redirect, url_for
from RecolectarBasura import *

app = Flask(__name__)

@app.route('/')
def index():
    return redirect(url_for('listar_alcaldias'))

@app.route('/Alcaldías', methods=['GET'])
def listar_alcaldias():
    """
    Lista las alcaldías disponibles.
    """
    alcaldias_dir = os.path.join(app.root_path, 'Alcaldías')
    alcaldias = []
    if os.path.exists(alcaldias_dir):
        for archivo in os.listdir(alcaldias_dir):
            if archivo.endswith('.shp'):
                # Se extrae el nombre sin la extensión
                alcaldia = os.path.splitext(archivo)[0]
                alcaldias.append(alcaldia)
    else:
        return jsonify({'error': 'No se encontró la carpeta de alcaldías.'}), 404

    return render_template('Alcaldías.html', alcaldías=alcaldias)

@app.route('/seleccionar_alcaldia', methods=['POST'])
def seleccionar_alcaldia():
    alcaldia = request.form.get('alcaldía')
    if not alcaldia:
        return redirect(url_for('listar_alcaldias'))
    return redirect(url_for('agregar_camiones', alcaldia=alcaldia))

@app.route('/Alcaldías/<alcaldia>/agregar_camiones', methods=['GET', 'POST'])
def agregar_camiones(alcaldia):
    if request.method == 'POST':
        # Procesar el formulario de camiones
        camiones = request.form.getlist('camiones')
        # Redirigir a la ruta de procesamiento de agrupamiento
        return redirect(url_for('procesar_agrupamiento', alcaldia=alcaldia))
    
    # Renderizar la plantilla con el formulario de camiones
    return render_template('Alcaldías.html', alcaldías=[], seleccionada=alcaldia)

@app.route('/Alcaldías/<alcaldia>/procesar_agrupamiento', methods=['POST'])
def procesar_agrupamiento(alcaldia):
    """
    Recibe mediante JSON los parámetros para:
      1) Seleccionar la alcaldía (nombre del shapefile sin extensión)
      2) Definir los parámetros deseados para los camiones
      3) Definir los parámetros del agrupamiento (AgrupamientoAGEB)
    
    Ejemplo de JSON recibido:
    {
      "alcaldia": "nombre_alcaldia",
      "camiones": [
          {"capacidad": 10000, "factor_reserva": 1.2, "cantidad_camiones": 2, "nombre": "Camión A"},
          {"capacidad": 8000, "factor_reserva": 1.1, "cantidad_camiones": 3, "nombre": "Camión B"}
      ],
      "agrupamiento": {
          "limite_peso": None,
          "tamano_poblacion": 750,
          "num_generaciones": 1500,
          "tasa_mutacion": 0.01,
          "factor_basura": 1.071,
          "reconectar_grupos": true,
          "semilla_random": 42
      }
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No se recibieron datos en la solicitud.'}), 400

    # 1. Selección de alcaldía
    if not alcaldia:
        return jsonify({'error': 'El parámetro "alcaldia" es obligatorio.'}), 400

    shapefile_path = os.path.join(app.root_path, 'Alcaldías', f'{alcaldia}.shp')
    if not os.path.exists(shapefile_path):
        return jsonify({'error': f"Shapefile para la alcaldía '{alcaldia}' no encontrado."}), 404

    # 2. Parámetros de los camiones
    camiones_data = data.get('camiones', [])
    camiones = []
    for c in camiones_data:
        try:
            camion = Camion(
                capacidad=float(c.get('capacidad', 0)),
                factor_reserva=float(c.get('factor_reserva', 1)),
                cantidad_camiones=int(c.get('cantidad_camiones', 1)),
                nombre=str(c.get('nombre', 'Camion'))
            )
            camiones.append(camion)
        except Exception as e:
            return jsonify({'error': f"Error al crear un camión: {e}"}), 400

    # 3. Parámetros para el agrupamiento
    agrupamiento_data = data.get('agrupamiento', {})
    limite_peso = agrupamiento_data.get('limite_peso')  # Puede ser None, en cuyo caso se usará la capacidad máxima de los camiones
    tamano_poblacion = int(agrupamiento_data.get('tamano_poblacion', 500))
    num_generaciones = int(agrupamiento_data.get('num_generaciones', 500))
    tasa_mutacion = float(agrupamiento_data.get('tasa_mutacion', 0.01))
    factor_basura = float(agrupamiento_data.get('factor_basura', 1.071))
    reconectar_grupos = bool(agrupamiento_data.get('reconectar_grupos', True))
    semilla_random = agrupamiento_data.get('semilla_random', None)
    if semilla_random is not None:
        semilla_random = int(semilla_random)

    try:
        # Crear la instancia de AgrupamientoAGEB con los parámetros recibidos
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

        # Ejecutar el algoritmo de agrupamiento y asignar camiones
        mejor_individuo, grupos, pesos_grupos = agrupador.ejecutar_agrupamiento()
        asignaciones, camiones_restantes = agrupador.asignar_camiones(grupos, pesos_grupos)

        resultado = {
            "mejor_individuo": mejor_individuo,
            "grupos": grupos,
            "pesos_grupos": pesos_grupos,
            "asignaciones": {str(g): (camion.nombre if camion is not None else None)
                             for g, camion in asignaciones.items()},
            "camiones_restantes": [camion.nombre for camion in camiones_restantes]
        }
        return jsonify(resultado)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5005)