const textarea = document.getElementById('email_text');
const charCount = document.getElementById('char-count');

if (textarea && charCount) {
  textarea.addEventListener('input', function() {
    charCount.textContent = this.value.length;
  });
}

const fileInput = document.getElementById('email_file');
const fileName = document.getElementById('file-name');

if (fileInput && fileName) {
  fileInput.addEventListener('change', function() {
    if (this.files.length > 0) {
      fileName.textContent = this.files[0].name;
    } else {
      fileName.textContent = 'Nenhum arquivo selecionado';
    }
  });
}

const copyResponseBtn = document.getElementById('copy-response');
if (copyResponseBtn) {
  copyResponseBtn.addEventListener('click', function() {
    const responseText = document.getElementById('suggested-response').textContent;
    navigator.clipboard.writeText(responseText).then(() => {
      this.innerHTML = '<i class="fas fa-check"></i> Copiado!';
      setTimeout(() => {
        this.innerHTML = '<i class="fas fa-copy"></i> Copiar resposta';
      }, 2000);
    });
  });
}

function initTabs() {
  const tabButtons = document.querySelectorAll('.tab-button');
  const tabContents = document.querySelectorAll('.tab-content');

  tabButtons.forEach(button => {
    button.addEventListener('click', () => {
      const targetTab = button.getAttribute('data-tab');

      tabButtons.forEach(btn => btn.classList.remove('active'));
      tabContents.forEach(content => content.classList.remove('active'));

      button.classList.add('active');
      const targetContent = document.getElementById(targetTab);
      if (targetContent) {
        targetContent.classList.add('active');
      }
    });
  });
}

function initFormValidation() {
  const form = document.getElementById('email-form');
  if (!form) return;

  form.addEventListener('submit', async function(e) {
    e.preventDefault();

    const activeTab = document.querySelector('.tab-content.active');
    const emailText = document.getElementById('email_text');
    const emailFile = document.getElementById('email_file');

    if (activeTab && activeTab.id === 'text-tab') {
      if (!emailText || !emailText.value.trim()) {
        alert('Por favor, insira o texto do e-mail.');
        return;
      }
    } else if (activeTab && activeTab.id === 'file-tab') {
      if (!emailFile || !emailFile.files[0]) {
        alert('Por favor, selecione um arquivo.');
        return;
      }
    }

    const formOverlay = document.getElementById('form-overlay');
    const submitBtn = document.getElementById('submit-btn');
    const resultsContainer = document.getElementById('results-container');
    const errorContainer = document.getElementById('error-container');

    if (formOverlay) formOverlay.classList.remove('hidden');
    if (submitBtn) submitBtn.classList.add('loading');
    if (resultsContainer) resultsContainer.classList.add('hidden');
    if (errorContainer) errorContainer.classList.add('hidden');

    try {
      const formData = new FormData(form);
      const response = await fetch('/analyze', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (response.ok) {
        displayResults(data);
      } else {
        displayError(data.error || 'Erro desconhecido');
      }
    } catch (error) {
      displayError('Erro de conexão: ' + error.message);
    } finally {
      if (formOverlay) formOverlay.classList.add('hidden');
      if (submitBtn) submitBtn.classList.remove('loading');
    }
  });
}

function displayResults(data) {
  const resultsContainer = document.getElementById('results-container');
  const formSection = document.querySelector('.form-section');
  const categoryElement = document.getElementById('category');
  const responseElement = document.getElementById('suggested-response');

  if (categoryElement) {
    categoryElement.textContent = data.categoria || 'N/A';
  }

  if (responseElement) {
    responseElement.textContent = data.resposta_automatica || 'Nenhuma resposta sugerida';
  }


  if (resultsContainer && formSection) {
    formSection.classList.add('hidden');
    resultsContainer.classList.remove('hidden');
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
  }
}

function displayError(errorMessage) {
  const errorContainer = document.getElementById('error-container');
  const errorElement = document.getElementById('error-message');

  if (errorElement) {
    errorElement.textContent = errorMessage;
  }

  if (errorContainer) {
    errorContainer.classList.remove('hidden');
    errorContainer.scrollIntoView({ behavior: 'smooth' });
  }
}

function switchTab(tabName) {
  const tabButton = document.querySelector(`[data-tab="${tabName}"]`);
  if (tabButton) {
    tabButton.click();
  }
}

function clearForm() {
  const textarea = document.getElementById('email_text');
  const fileInput = document.getElementById('email_file');
  const fileName = document.getElementById('file-name');
  const charCount = document.getElementById('char-count');
  const resultsContainer = document.getElementById('results-container');
  const formSection = document.querySelector('.form-section');
  const errorContainer = document.getElementById('error-container');
  const formOverlay = document.getElementById('form-overlay');

  if (textarea) textarea.value = '';
  if (fileInput) fileInput.value = '';
  if (fileName) fileName.textContent = 'Nenhum arquivo selecionado';
  if (charCount) charCount.textContent = '0';
  if (resultsContainer) resultsContainer.classList.add('hidden');
  if (formSection) formSection.classList.remove('hidden');
  if (errorContainer) errorContainer.classList.add('hidden');
  if (formOverlay) formOverlay.classList.add('hidden');

  switchTab('text-tab');
}

function backToForm() {
  const resultsContainer = document.getElementById('results-container');
  const formSection = document.querySelector('.form-section');
  const errorContainer = document.getElementById('error-container');

  if (resultsContainer) resultsContainer.classList.add('hidden');
  if (formSection) formSection.classList.remove('hidden');
  if (errorContainer) errorContainer.classList.add('hidden');

  formSection.scrollIntoView({ behavior: 'smooth' });
}

document.addEventListener('DOMContentLoaded', function() {
  initTabs();
  initFormValidation();
  
  const backToFormBtn = document.getElementById('back-to-form');
  if (backToFormBtn) {
    backToFormBtn.addEventListener('click', backToForm);
  }
});