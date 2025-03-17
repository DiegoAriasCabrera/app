document.addEventListener('DOMContentLoaded', function() {
  // Al cambiar la selección de alcaldía, enviamos el formulario para recargar la vista
  const selectAlcaldia = document.getElementById('alcaldía');
  if (selectAlcaldia) {
    selectAlcaldia.addEventListener('change', function() {
      document.getElementById('alcaldiaForm').submit();
    });
  }

  // Botón para agregar más filas de camiones
  const addCamionBtn = document.getElementById('addCamionBtn');
  if (addCamionBtn) {
    addCamionBtn.addEventListener('click', function() {
      const camionesContainer = document.getElementById('camionesContainer');
      // Contamos cuántas filas hay actualmente en el tbody

      // Creamos una nueva fila
      const newRow = document.createElement('tr');
      newRow.innerHTML = 
        `<td>
          <input type="text" name="camiones[${rowCount}][nombre]" required>
          </td><td>
            <input type="number" name="camiones[${rowCount}][capacidad]" min="0" required>
            </td><td>
            <input type="number" name="camiones[${rowCount}][factor_reserva]" min="0" required>
            </td><td>
            <input type="number" name="camiones[${rowCount}][cantidad_camiones]" min="0" required>
            </td><td></td>`;

      // Añadimos la fila al tbody
      camionesContainer.appendChild(newRow);
    });
  }
});