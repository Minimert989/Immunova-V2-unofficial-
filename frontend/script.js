// frontend/script.js

async function uploadData() {
  const fileInput = document.getElementById('dataFile');
  const output = document.getElementById('output');

  if (!fileInput.files.length) {
    output.textContent = "Please upload a CSV file.";
    return;
  }

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  output.textContent = "Processing...";

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      body: formData,
    });

    const result = await response.json();
    output.textContent = JSON.stringify(result, null, 2);
  } catch (error) {
    output.textContent = "Error: " + error.message;
  }
}
