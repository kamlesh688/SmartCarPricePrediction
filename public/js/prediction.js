const fillSample = document.getElementById('fillSample');
const brandSelect = document.getElementById('brand-select');
const modelSelect = document.getElementById('model-select');

function updateModelOptions(preferredModel = '') {
  if (!brandSelect || !modelSelect) return;

  const brand = brandSelect.value;
  const modelsByBrand = featureMetadata?.models_by_brand || {};
  const models = modelsByBrand[brand] || (brand ? featureMetadata?.models || [] : []);

  modelSelect.innerHTML = '';
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = brand ? 'Select a model' : 'Select a brand first';
  modelSelect.appendChild(placeholder);

  models.forEach((model) => {
    const option = document.createElement('option');
    option.value = model;
    option.textContent = model;
    modelSelect.appendChild(option);
  });

  modelSelect.disabled = !brand;
  if (preferredModel && models.includes(preferredModel)) {
    modelSelect.value = preferredModel;
  }
}

if (brandSelect && modelSelect) {
  updateModelOptions();
  brandSelect.addEventListener('change', () => updateModelOptions());
}

if (fillSample) {
  fillSample.addEventListener('click', () => {
    if (brandSelect) {
      brandSelect.value = brandSelect.options[1]?.value || '';
      updateModelOptions();
    }
    if (modelSelect) {
      modelSelect.value = modelSelect.options[1]?.value || '';
    }
    document.querySelector('input[name="year"]').value = new Date().getFullYear() - 3;
    document.querySelector('input[name="engine_size_cc"]').value = 1800;
    document.querySelector('input[name="mileage_km"]').value = 62000;
    document.querySelector('input[name="owner_count"]').value = 1;
  });
}
