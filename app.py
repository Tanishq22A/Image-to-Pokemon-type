import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, render_template_string, jsonify
import tensorflow as tf

# ---- Settings ----
IMG_SIZE = (120, 120)
NORMALIZE_255 = True
MODEL_PATH = os.environ.get("MODEL_PATH", "pokemon_cnn.h5")
ALLOWED_EXT = {"png", "jpg", "jpeg"}

IDX2LABEL = [
    "Water", "Normal", "Fire", "Grass", "Ghost",
    "Bug", "Electric", "Poison", "Psychic", "Rock", "Fighting"
]

# ---- Load model ----
model = tf.keras.models.load_model(MODEL_PATH)

app = Flask(__name__)

HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Pokémon Type Classifier</title>
    <style>
      :root { --bg:#0f172a; --card:#111827; --text:#e5e7eb; --accent:#60a5fa; --muted:#94a3b8; }
      * { box-sizing: border-box; }
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background: linear-gradient(180deg,#0b1022,#0f172a); color: var(--text); margin:0; }
      .container { max-width: 880px; margin: 32px auto; padding: 0 16px; }
      .title { font-weight: 700; letter-spacing: .3px; margin-bottom: 18px; }
      .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
      @media (max-width: 860px){ .grid { grid-template-columns: 1fr; } }
      .card { background: rgba(17,24,39,.85); backdrop-filter: blur(6px); border:1px solid #1f2937; border-radius: 14px; padding: 18px; }
      .drop { border: 2px dashed #334155; border-radius: 12px; padding: 16px; text-align:center; transition: .2s border-color, .2s background; }
      .drop.dragover { border-color: var(--accent); background: rgba(96,165,250,.08); }
      .btn { background: var(--accent); color:#0b1022; border:0; padding:10px 16px; border-radius:10px; font-weight:600; cursor:pointer; }
      .btn:disabled { opacity:.6; cursor:not-allowed; }
      .muted { color: var(--muted); font-size: 14px; }
      .row { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
      img.preview { width: 100%; max-height: 320px; object-fit: contain; border-radius: 10px; background:#0b1022; }
      .result { font-size: 18px; margin-top: 10px; }
      table { width: 100%; border-collapse: collapse; margin-top: 10px; }
      th, td { border-bottom: 1px solid #1f2937; padding: 8px 6px; text-align:left; }
      .tag { background:#1f2937; padding:4px 8px; border-radius:8px; border:1px solid #334155; }
      footer { margin-top: 16px; font-size: 13px; color: var(--muted); text-align:center; }
    </style>
  </head>
  <body>
    <div class="container">
      <h2 class="title">Pokémon Type Classifier</h2>
      <div class="grid">
        <div class="card">
          <form id="form" method="POST" action="/predict" enctype="multipart/form-data">
            <div id="drop" class="drop">
              <p><strong>Drop</strong> an image here or click to select.</p>
              <p class="muted">Accepted: PNG, JPG, JPEG. Max ~5 MB.</p>
              <input id="file" type="file" name="file" accept="image/*" style="display:none" required />
              <button type="button" id="choose" class="btn">Choose image</button>
            </div>
            <div id="previewWrap" style="margin-top:12px; display:none;">
              <div class="row">
                <span class="tag" id="filename">image.png</span>
                <span class="muted" id="filesize"></span>
              </div>
              <img id="preview" class="preview" alt="preview"/>
            </div>
            <div class="row" style="margin-top:12px;">
              <button class="btn" type="submit" id="submitBtn" disabled>Predict</button>
              <span class="muted">Model: <span class="tag">pokemon_cnn.h5</span></span>
            </div>
          </form>
          <div id="err" style="color:#fca5a5; margin-top:8px;"></div>
        </div>

        <div class="card" id="resCard" style="display:none;">
          <div class="result">Prediction: <span class="tag" id="predLabel"></span></div>
          <table id="probsTable">
            <thead><tr><th>Class</th><th>Probability</th></tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
      <footer>This demo runs entirely on the server; no image is stored.</footer>
    </div>

    <script>
      const fileInput = document.getElementById('file');
      const chooseBtn = document.getElementById('choose');
      const drop = document.getElementById('drop');
      const submitBtn = document.getElementById('submitBtn');
      const previewWrap = document.getElementById('previewWrap');
      const preview = document.getElementById('preview');
      const filenameEl = document.getElementById('filename');
      const filesizeEl = document.getElementById('filesize');
      const errEl = document.getElementById('err');
      const resCard = document.getElementById('resCard');
      const predLabel = document.getElementById('predLabel');
      const probsBody = document.querySelector('#probsTable tbody');

      const MAX_MB = 5;
      function validateFile(f){
        const okTypes = ['image/png','image/jpeg','image/jpg'];
        if (!okTypes.includes(f.type)) return 'Unsupported file type';
        if (f.size > MAX_MB*1024*1024) return 'File too large';
        return '';
      }

      function showPreview(f){
        filenameEl.textContent = f.name;
        const mb = (f.size/1024/1024).toFixed(2);
        filesizeEl.textContent = `(${mb} MB)`;
        const reader = new FileReader();
        reader.onload = e => { preview.src = e.target.result; previewWrap.style.display = 'block'; };
        reader.readAsDataURL(f);
      }

      chooseBtn.addEventListener('click', () => fileInput.click());
      drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('dragover'); });
      drop.addEventListener('dragleave', () => drop.classList.remove('dragover'));
      drop.addEventListener('drop', e => {
        e.preventDefault(); drop.classList.remove('dragover');
        if (e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; fileInput.dispatchEvent(new Event('change')); }
      });
      drop.addEventListener('click', () => fileInput.click());

      fileInput.addEventListener('change', () => {
        errEl.textContent = '';
        resCard.style.display = 'none';
        if (!fileInput.files.length){ submitBtn.disabled = true; return; }
        const f = fileInput.files[0];
        const msg = validateFile(f);
        if (msg){ errEl.textContent = msg; submitBtn.disabled = true; return; }
        showPreview(f);
        submitBtn.disabled = false;
      });

      // Submit via fetch to stay on the same page and render results
      document.getElementById('form').addEventListener('submit', async (e) => {
        e.preventDefault();
        errEl.textContent = '';
        submitBtn.disabled = true;
        const data = new FormData();
        if (!fileInput.files.length){ errEl.textContent = 'Please select an image.'; submitBtn.disabled = false; return; }
        data.append('file', fileInput.files[0]);
        try {
          const resp = await fetch('/api/predict', { method:'POST', body:data });
          const json = await resp.json();
          if (!resp.ok){ throw new Error(json.error || 'Prediction failed'); }
          predLabel.textContent = json.label;
          probsBody.innerHTML = '';
          if (Array.isArray(json.probs)){
            const probs = json.probs.map((p,i)=>({ name: "{{ labels|safe }}".split(',' )[i], prob: p }));
            probs.sort((a,b)=>b.prob - a.prob);
            for (const {name, prob} of probs){
              const tr = document.createElement('tr');
              tr.innerHTML = `<td>${name}</td><td>${(prob).toFixed(3)}</td>`;
              probsBody.appendChild(tr);
            }
          }
          resCard.style.display = 'block';
        } catch (err) {
          errEl.textContent = err.message;
        } finally {
          submitBtn.disabled = false;
        }
      });
    </script>
  </body>
</html>
"""

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    if NORMALIZE_255:
        arr = arr / 255.0
    return np.expand_dims(arr, 0)

@app.route("/", methods=["GET"])
def index():
    labels = ",".join(IDX2LABEL)
    return render_template_string(HTML, labels=labels)

@app.route("/predict", methods=["POST"])
def predict_page():
    # Kept for compatibility, but UI uses /api/predict to avoid page reload
    if "file" not in request.files or request.files["file"].filename == "":
        return render_template_string(HTML, labels=",".join(IDX2LABEL))
    img = Image.open(io.BytesIO(request.files["file"].read()))
    x = preprocess_image(img)
    preds = model.predict(x)
    idx = int(np.argmax(preds, axis=1)[0])
    label = IDX2LABEL[idx] if 0 <= idx < len(IDX2LABEL) else f"Class {idx}"
    # Simple render
    return render_template_string(HTML, labels=",".join(IDX2LABEL))

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "file is required"}), 400
    f = request.files["file"]
    img = Image.open(io.BytesIO(f.read()))
    x = preprocess_image(img)
    preds = model.predict(x)
    idx = int(np.argmax(preds, axis=1)[0])
    label = IDX2LABEL[idx] if 0 <= idx < len(IDX2LABEL) else f"Class {idx}"
    return jsonify({"label": label, "index": idx, "probs": preds[0].tolist(), "classes": IDX2LABEL})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
