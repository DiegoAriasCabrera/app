// Pop-up de ayuda para parámetros
const helpBtn = document.getElementById('helpBtn');
const helpPopup = document.getElementById('helpPopup');
const closeHelpPopup = document.getElementById('closeHelpPopup');

if (helpBtn && helpPopup && closeHelpPopup) {
  // Abrir pop-up
  helpBtn.addEventListener('click', () => {
    helpPopup.style.display = 'block';
  });
  // Cerrar pop-up al clicar la X
  closeHelpPopup.addEventListener('click', () => {
    helpPopup.style.display = 'none';
  });
  // Cerrar al dar clic fuera del contenido
  window.addEventListener('click', (event) => {
    if (event.target === helpPopup) {
      helpPopup.style.display = 'none';
    }
  });
}

// Pop-up de ayuda para camiones
const helpTrucksBtn = document.getElementById('helpTrucksBtn');
const helpTrucksPopup = document.getElementById('helpTrucksPopup');
const closeHelpTrucksPopup = document.getElementById('closeHelpTrucksPopup');

if (helpTrucksBtn && helpTrucksPopup && closeHelpTrucksPopup) {
  // Abrir pop-up
  helpTrucksBtn.addEventListener('click', () => {
    helpTrucksPopup.style.display = 'block';
  });
  // Cerrar pop-up al clicar la X
  closeHelpTrucksPopup.addEventListener('click', () => {
    helpTrucksPopup.style.display = 'none';
  });
  // Cerrar al dar clic fuera del contenido
  window.addEventListener('click', (event) => {
    if (event.target === helpTrucksPopup) {
      helpTrucksPopup.style.display = 'none';
    }
  });
}

// Pop-up de ayuda para agrupamiento
const helpSectorBtn = document.getElementById('helpSectorBtn');
const helpSectorPopup = document.getElementById('helpSectorPopup');
const closeHelpSectorPopup = document.getElementById('closeHelpSectorPopup');

if (helpSectorBtn && helpSectorPopup && closeHelpSectorPopup) {
  // Abrir pop-up
  helpSectorBtn.addEventListener('click', () => {
    helpSectorPopup.style.display = 'block';
  });
  // Cerrar pop-up al clicar la X
  closeHelpSectorPopup.addEventListener('click', () => {
    helpSectorPopup.style.display = 'none';
  });
  // Cerrar al dar clic fuera del contenido
  window.addEventListener('click', (event) => {
    if (event.target === helpSectorPopup) {
      helpSectorPopup.style.display = 'none';
    }
  });
}

// Pop-up de ayuda para TSP
const helpTSPBtn = document.getElementById('helpTSPBtn');
const helpTSPPopup = document.getElementById('helpTSPPopup');
const closeHelpTSPPopup = document.getElementById('closeHelpTSPPopup');

if (helpTSPBtn && helpTSPPopup && closeHelpTSPPopup) {
  // Abrir pop-up
  helpTSPBtn.addEventListener('click', () => {
    helpTSPPopup.style.display = 'block';
  });
  // Cerrar pop-up al clicar la X
  closeHelpTSPPopup.addEventListener('click', () => {
    helpTSPPopup.style.display = 'none';
  });
  // Cerrar al dar clic fuera del contenido
  window.addEventListener('click', (event) => {
    if (event.target === helpTSPPopup) {
      helpTSPPopup.style.display = 'none';
    }
  });
}