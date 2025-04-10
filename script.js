document.addEventListener('DOMContentLoaded', function () {
    const generateBtn = document.getElementById('generate-btn');
    const resetBtn = document.getElementById('reset-btn');
    const promptInput = document.getElementById('prompt');
    const stepsInput = document.getElementById('steps');
    const stepsValue = document.getElementById('steps-value');
    const guidanceInput = document.getElementById('guidance');
    const guidanceValue = document.getElementById('guidance-value');
    const styleInput = document.getElementById('style');
    const colorInput = document.getElementById('color');
    const sizeInput = document.getElementById('size');
    const sizeValue = document.getElementById('size-value');
    const backgroundInput = document.getElementById('background');
    const backendUrlInput = document.getElementById('backend-url');
    const generatedImage = document.getElementById('generated-image');
    const loadingSection = document.querySelector('.loading');
    const placeholder = document.querySelector('.pokemon-placeholder');
    const gallery = document.getElementById('gallery');
    const historySection = document.getElementById('history-section');

    stepsInput.addEventListener('input', () => stepsValue.textContent = stepsInput.value);
    guidanceInput.addEventListener('input', () => guidanceValue.textContent = guidanceInput.value);
    sizeInput.addEventListener('input', () => {
        const sizes = ['Tiny', 'Medium', 'Large'];
        sizeValue.textContent = sizes[sizeInput.value];
    });

    document.querySelectorAll('.example-prompt').forEach(example => {
        example.addEventListener('click', () => {
            promptInput.value = example.textContent;
            console.log(`Main: Example prompt selected: "${example.textContent}"`);
        });
    });

    generateBtn.addEventListener('click', generatePokemon);
    resetBtn.addEventListener('click', resetForm);

    function generatePokemon() {
        const basePrompt = promptInput.value.trim();
        if (!basePrompt) {
            alert('Please enter a description for your Pokémon!');
            return;
        }

        const style = styleInput.value !== 'default' ? `${styleInput.value} style` : '';
        const color = `a ${hexToColorName(colorInput.value)}`;
        const size = `${sizeValue.textContent.toLowerCase()}`;
        const background = backgroundInput.value !== 'none' ? `in a ${backgroundInput.value} background` : '';
        const fullPrompt = `${style} ${color} ${size} Pokémon, ${basePrompt} ${background}`.trim().replace(/\s+/g, ' ');
        const backendUrl = backendUrlInput.value || 'http://localhost:5001';

        loadingSection.style.display = 'block';
        placeholder.style.display = 'flex';
        generatedImage.style.display = 'none';
        generateBtn.disabled = true;

        console.log(`Main: Sending request to ${backendUrl}/start-generation | Prompt: "${fullPrompt}"`);
        fetch(`${backendUrl}/start-generation`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: fullPrompt,
                steps: parseInt(stepsInput.value),
                guidance: parseFloat(guidanceInput.value)
            }),
        })
        .then(res => {
            console.log(`Main: Response status: ${res.status} ${res.statusText}`);
            if (!res.ok) {
                return res.json().then(errData => {
                    throw new Error(`HTTP error ${res.status}: ${errData.error || res.statusText}`);
                });
            }
            return res.blob();
        })
        .then(blob => {
            console.log(`Main: Blob received, size: ${blob.size} bytes, type: ${blob.type}`);
            if (blob.size === 0 || blob.type !== 'image/png') {
                throw new Error('Invalid image data received');
            }
            const imageUrl = URL.createObjectURL(blob);
            console.log(`Main: Setting image src to: ${imageUrl}`);
            generatedImage.src = imageUrl;
            generatedImage.onload = () => {
                console.log('Main: Image loaded successfully');
                loadingSection.style.display = 'none';
                placeholder.style.display = 'none';
                generatedImage.style.display = 'block';
                addToGallery(imageUrl, fullPrompt);
                historySection.style.display = 'block';
                generateBtn.disabled = false;
            };
            generatedImage.onerror = () => {
                console.error('Main: Image failed to load');
                URL.revokeObjectURL(imageUrl);
                throw new Error('Image failed to load in the browser');
            };
        })
        .catch(err => {
            console.error(`Main: Generation failed: ${err.message}`);
            alert(`Failed to generate image: ${err.message}`);
            loadingSection.style.display = 'none';
            generateBtn.disabled = false;
        });
    }

    function resetForm() {
        promptInput.value = '';
        styleInput.value = 'default';
        colorInput.value = '#ff0000';
        sizeInput.value = '1';
        sizeValue.textContent = 'Medium';
        backgroundInput.value = 'none';
        stepsInput.value = '50';
        stepsValue.textContent = '50';
        guidanceInput.value = '7.5';
        guidanceValue.textContent = '7.5';
        generatedImage.style.display = 'none';
        placeholder.style.display = 'flex';
        loadingSection.style.display = 'none';
        console.log('Main: Form reset');
    }

    function addToGallery(imageUrl, prompt) {
        const galleryItem = document.createElement('div');
        galleryItem.className = 'gallery-item';
        const image = document.createElement('img');
        image.src = imageUrl;
        image.alt = prompt;
        const caption = document.createElement('div');
        caption.className = 'gallery-caption';
        caption.textContent = prompt.length > 30 ? prompt.substring(0, 30) + '...' : prompt;
        galleryItem.appendChild(image);
        galleryItem.appendChild(caption);
        galleryItem.addEventListener('click', () => {
            generatedImage.src = imageUrl;
            generatedImage.style.display = 'block';
            placeholder.style.display = 'none';
            promptInput.value = prompt.split(', ')[1] || prompt;
            window.scrollTo({ top: 0, behavior: 'smooth' });
            console.log(`Main: Gallery item clicked, displaying ${imageUrl}`);
        });
        gallery.prepend(galleryItem);
        console.log(`Main: Added to gallery: ${imageUrl}`);
    }

    function hexToColorName(hex) {
        const colors = {
            '#ff0000': 'red',
            '#00ff00': 'green',
            '#0000ff': 'blue',
            '#ffff00': 'yellow',
            '#ff00ff': 'pink',
            '#00ffff': 'cyan',
            '#000000': 'black',
            '#ffffff': 'white'
        };
        return colors[hex.toLowerCase()] || 'custom-colored';
    }
});