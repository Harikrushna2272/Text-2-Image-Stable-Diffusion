import os
import gc
import uuid
import time
import torch
import traceback
from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify, send_from_directory, send_file
from diffusers import StableDiffusionPipeline
from peft import PeftModel

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

base_model_id = "runwayml/stable-diffusion-v1-5"
lora_model_id = "AryanMakadiya/pokemon_lora"

pipe = StableDiffusionPipeline.from_pretrained(base_model_id, use_safetensors=True)
unet_base = pipe.unet
unet = PeftModel.from_pretrained(unet_base, lora_model_id)
pipe.unet = unet
pipe = pipe.to("cpu")

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@app.route('/')
def serve_index():
    return send_file('index.html')

@app.route('/start-generation', methods=['POST'])
@cross_origin()
def start_generation():
    data = request.get_json()
    prompt = data.get('prompt', "A bubbly Pokémon with a round, smiling face and twinkling stars.")
    steps = int(data.get('steps', 50))
    guidance = float(data.get('guidance', 7.5))

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    if steps < 1 or steps > 100:
        return jsonify({"error": "Steps must be between 1 and 100"}), 400
    if guidance < 1 or guidance > 20:
        return jsonify({"error": "Guidance scale must be between 1 and 20"}), 400

    task_id = str(uuid.uuid4())
    print(f"Task {task_id}: Initialized at {time.strftime('%H:%M:%S')}")

    try:
        clear_memory()
        print(f"Task {task_id}: Starting generation at {time.strftime('%H:%M:%S')} | Prompt: {prompt} | Steps: {steps} | Guidance: {guidance}")
        with torch.no_grad():
            image = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance).images[0]
        image_path = os.path.join(STATIC_DIR, f"generated_image_{task_id}.png")
        image.save(image_path)
        file_size = os.path.getsize(image_path)
        print(f"Task {task_id}: Generation completed at {time.strftime('%H:%M:%S')} | Image saved at {image_path} | Size: {file_size} bytes")
        if file_size == 0:
            raise Exception("Generated image file is empty")
        clear_memory()
        return send_file(image_path, mimetype='image/png')
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Task {task_id}: Failed with error: {error_msg}")
        clear_memory()
        return jsonify({
            "task_id": task_id,
            "error": error_msg
        }), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)