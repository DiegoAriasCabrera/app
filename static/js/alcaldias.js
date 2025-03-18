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
      const rowCount = camionesContainer.getElementsByTagName('tr').length;

      // Creamos una nueva fila
      const newRow = document.createElement('tr');
      newRow.innerHTML = 
        `<td> <input type="text" name="camiones[${rowCount}][nombre]" required > </td>
         <td><input type="text" name="camiones[${rowCount}][capacidad]" min="0" required inputmode="decimal" oninput="this.value = this.value.replace(/[^0-9.]/g, '');" > </td>
         <td><input type="text" name="camiones[${rowCount}][factor_reserva]" min="0" required inputmode="decimal" oninput="this.value = this.value.replace(/[^0-9.]/g, '');" > </td>
         <td><input type="text" name="camiones[${rowCount}][cantidad_camiones]" min="0" required inputmode="numeric" oninput="this.value = this.value.replace(/[^0-9]/g, '');" > </td>
         <td></td>`;

      // Añadimos la fila al tbody
      camionesContainer.appendChild(newRow);
    });
  }

  // Botón para quitar la última fila de camiones
  const removeCamionBtn = document.getElementById('removeCamionBtn');
  if (removeCamionBtn) {
    removeCamionBtn.addEventListener('click', function() {
      const camionesContainer = document.getElementById('camionesContainer');
      const rowCount = camionesContainer.getElementsByTagName('tr').length;
      // Solo quitar la fila si hay más de una
      if (rowCount > 1) {
        camionesContainer.removeChild(camionesContainer.lastChild);
      }
    });
  }
});