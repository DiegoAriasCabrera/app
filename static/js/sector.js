document.addEventListener("DOMContentLoaded", function() {
    // Inicializar el mapa centrado en Ciudad de México
    var map = L.map('map').setView([19.432608, -99.133209], 11);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);
    
    var sectorData = [];
    
    // Procesa cada sector inyectado desde la plantilla
    rawSectores.forEach(function(sector) {
      var geojson = JSON.parse(sector.geojson);
      var unionGeom = null;
      // Unir las geometrías de cada sector en una sola entidad
      geojson.features.forEach(function(feat) {
        if (!unionGeom) {
          unionGeom = feat;
        } else {
          try {
            unionGeom = turf.union(unionGeom, feat);
          } catch(err) {
            console.error("Error al unir geometrías para sector " + sector.id + ": ", err);
          }
        }
      });
      if (!unionGeom && geojson.features.length > 0) {
        unionGeom = geojson.features[0];
      }
      sectorData.push({
        id: sector.id,
        geojson: geojson,
        unionGeom: unionGeom,
        neighbors: [],
        color: null
      });
    });
    
    // Determinar adyacencias: dos sectores son vecinos si sus geometrías unidas se intersectan.
    for (var i = 0; i < sectorData.length; i++) {
      for (var j = i + 1; j < sectorData.length; j++) {
        try {
          if (turf.booleanIntersects(sectorData[i].unionGeom, sectorData[j].unionGeom)) {
            sectorData[i].neighbors.push(sectorData[j].id);
            sectorData[j].neighbors.push(sectorData[i].id);
          }
        } catch(err) {
          console.error("Error al verificar intersección entre sectores " + sectorData[i].id + " y " + sectorData[j].id + ": ", err);
        }
      }
    }
    
    // Ordenar sectores según la cantidad de vecinos (de mayor a menor)
    sectorData.sort(function(a, b) {
      return b.neighbors.length - a.neighbors.length;
    });
    
    // Asignar colores a cada sector utilizando un algoritmo greedy.
    var palette = ["#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#A020F0", "#808080"];
    sectorData.forEach(function(sector, index) {
      var used = new Set();
      sector.neighbors.forEach(function(nId) {
        var neighbor = sectorData.find(function(s) { return s.id === nId; });
        if (neighbor && neighbor.color !== null) {
          used.add(neighbor.color);
        }
      });
      for (var k = 0; k < palette.length; k++) {
        if (!used.has(palette[k])) {
          sector.color = palette[k];
          break;
        }
      }
      if (!sector.color) {
        sector.color = palette[0];
      }
    });
    
    // Crear un diccionario para mapear cada sector a su color asignado.
    var sectorColorMap = {};
    sectorData.forEach(function(sector) {
      sectorColorMap[sector.id] = sector.color;
    });
    
    var allLayers = [];
    
    // Función para asignar eventos a cada capa de sector
    function attachSectorEvents(layer, sectorId) {
      // Al hacer clic, se solicita confirmación para procesar el sector
      layer.on('click', function(e) {
        var confirmacion = confirm("¿Desea procesar el sector " + sectorId + "?");
        if (confirmacion) {
          document.getElementById("sector_id").value = sectorId;
          document.getElementById("sectorForm").submit();
        }
      });
      // Efectos visuales al pasar el ratón
      layer.on('mouseover', function(e) {
        layer.setStyle({ weight: 4, fillOpacity: 0.5 });
        layer.bringToFront();
      });
      layer.on('mouseout', function(e) {
        layer.setStyle({ weight: 2, fillOpacity: 0.3 });
      });
    }
    
    // Dibujar cada sector en el mapa
    rawSectores.forEach(function(sector, index) {
      var geojson = JSON.parse(sector.geojson);
      // Asignar el id de sector a cada feature para referencia
      geojson.features.forEach(function(feature) {
        feature.properties.sector_id = sector.id;
      });
      // Obtener el color asignado para el sector
      var fillColor = sectorColorMap[sector.id] || palette[index % palette.length];
      // Crear la capa GeoJSON con estilos
      var sectorLayer = L.geoJSON(geojson, {
        style: function(feature) {
          return {
            color: fillColor,
            weight: 2,
            fillColor: fillColor,
            fillOpacity: 0.3
          };
        }
      });
      // Asignar eventos de interacción a la capa
      attachSectorEvents(sectorLayer, sector.id);
      sectorLayer.addTo(map);
      allLayers.push(sectorLayer);
      
      // Colocar una etiqueta en el centro del sector
      var center = sectorLayer.getBounds().getCenter();
      L.marker(center, {
        icon: L.divIcon({
          className: 'sector-label',
          html: '<div>' + sector.id + '</div>',
          iconSize: [60, 25],
          iconAnchor: [30, 12]
        }),
        interactive: false
      }).addTo(map);
    });
    
    // Ajustar el mapa para que todos los sectores se vean
    if (allLayers.length > 0) {
      var group = L.featureGroup(allLayers);
      map.fitBounds(group.getBounds(), { padding: [20, 20] });
    }
    
    // Forzar la actualización del tamaño del mapa (útil si el contenedor cambia de tamaño)
    setTimeout(function() {
      map.invalidateSize();
    }, 200);
  });