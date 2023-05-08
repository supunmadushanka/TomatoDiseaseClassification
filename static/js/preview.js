const input = document.querySelector('#image-upload');
const preview = document.querySelector('#image-preview');

input.addEventListener('change', () => {
  const file = input.files[0];

  if (file) {
    const reader = new FileReader();

    reader.addEventListener('load', () => {
      preview.src = reader.result;
    });

    reader.readAsDataURL(file);
  }
});
