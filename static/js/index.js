// Inicia SSE hacia /Alcaldías/<alcaldia>/progreso_agrupamiento
function iniciarProceso1(url) {
    const barra1 = document.getElementById("barra1");
    const barra1Inner = document.getElementById("barra1-inner");
    const msg1 = document.getElementById("msg1");
    barra1.style.display = "block"; 
    barra1Inner.style.width = "0%";
    barra1Inner.textContent = "0%";

    const source = new EventSource(url);

    source.onmessage = function(evt) {
      const data = evt.data.trim();
      if (data === "DONE") {
        // Oculta la barra
        barra1.style.display = "none";
        // Muestra mensaje
        msg1.innerHTML = "<p>¡Agrupamiento finalizado! Imagen generada.</p>";
        source.close();

        // Ahora iniciamos proceso2
        iniciarProceso2(url.replace("progreso_agrupamiento","progreso_calles"));
      } else if (data === "ERROR") {
        barra1.style.display = "none";
        msg1.innerHTML = "<p style='color:red'>Error en Agrupamiento</p>";
        source.close();
      } else {
        // data es un número (porcentaje)
        const pct = parseInt(data);
        barra1Inner.style.width = pct + "%";
        barra1Inner.textContent = pct + "%";
      }
    };

    source.onerror = function(e) {
      console.error("Error SSE en proceso1:", e);
      source.close();
      barra1.style.display = "none";
      msg1.innerHTML = "<p style='color:red'>Ocurrió un error</p>";
    };
  }

  // Inicia SSE para el segundo proceso
  function iniciarProceso2(url) {
    const barra2 = document.getElementById("barra2");
    const barra2Inner = document.getElementById("barra2-inner");
    const msg2 = document.getElementById("msg2");

    barra2.style.display = "block";
    barra2Inner.style.width = "0%";
    barra2Inner.textContent = "0%";

    const source2 = new EventSource(url);

    source2.onmessage = function(evt) {
      const data = evt.data.trim();
      if (data === "DONE2") {
        // Oculta la barra
        barra2.style.display = "none";
        msg2.innerHTML = "<p>¡ProcesadorCalles finalizado! Imagen generada.</p>";
        source2.close();

        // Ir a la página final (muestra ambas imágenes)
        window.location = "/final";
      } else if (data === "ERROR") {
        barra2.style.display = "none";
        msg2.innerHTML = "<p style='color:red'>Error en ProcesadorCalles</p>";
        source2.close();
      } else {
        const pct = parseInt(data);
        barra2Inner.style.width = pct + "%";
        barra2Inner.textContent = pct + "%";
      }
    };

    source2.onerror = function(e) {
      console.error("Error SSE en proceso2:", e);
      source2.close();
      barra2.style.display = "none";
      msg2.innerHTML = "<p style='color:red'>Ocurrió un error</p>";
    };
  }

  // Al cargar la página, inicia el primer proceso SSE
  window.onload = function() {
    // Ejemplo: la URL SSE es la actual con /progreso_agrupamiento
    // /Alcaldías/<alcaldia>/sse => /Alcaldías/<alcaldia>/progreso_agrupamiento
    const currentPath = window.location.pathname; // "/Alcaldías/<alcaldia>/sse"
    // Reemplazamos "/sse" por "/progreso_agrupamiento":
    const url1 = currentPath.replace("/sse", "/progreso_agrupamiento");
    iniciarProceso1(url1);
  };