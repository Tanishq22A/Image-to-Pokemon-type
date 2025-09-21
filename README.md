#Pokemon Type Classifier (CNN + Flask)
A complete, reproducible pipeline that trains a convolutional neural network (CNN) to predict a Pokémon’s primary type from an image and serves real-time predictions via a lightweight Flask web app with drag‑and‑drop upload and a JSON API.

Features
CNN trained on 120×120 RGB inputs normalized to for type prediction.

Web UI: client‑side preview, file validation, and sorted per‑class probabilities.

API: POST /api/predict returns label, index, and probabilities for integration.

Clear separation of training (notebook) and serving (Flask app).
Repository structure
app.py — Flask server with UI and /api/predict endpoint.

notebooks/pokemon.ipynb — exploration, training, and model export.

data/pokemon.csv — Pokémon names, types, and evolution metadata.

model.meta.json — auxiliary metadata (classes and encodings).

models/ — place trained model here (e.g., models/pokemoncnn.h5).

Suggested ancillary files:

requirements.txt — Python dependencies.

Procfile, runtime.txt — for PaaS deploys.

Dockerfile, .dockerignore — for containerized runs.

.gitignore — ignore venv, caches, large artifacts.

Requirements
Python 3.11+

TensorFlow/Keras, Flask, Pillow, NumPy, Gunicorn (for prod)

Example requirements.txt:

flask==3.0.3

gunicorn==21.2.0

pillow==10.4.0

numpy==1.26.4

tensorflow==2.15.0

keras==2.15.0

Quickstart
Setup

python -m venv venv

source venv/bin/activate # Windows: venv\Scripts\activate

pip install -r requirements.txt

Train and export model

Open notebooks/pokemon.ipynb and run cells to train the CNN.

Save the model to models/pokemoncnn.h5 (ensure the models/ folder exists).

Run the server

export MODELPATH=models/pokemoncnn.h5

python app.py

Open http://localhost:5000 to use the web UI.

API usage
Endpoint

POST /api/predict

Request

Content-Type: multipart/form-data

Key: file Value: <image file: PNG/JPG/JPEG, up to ~5 MB>

Response (JSON)

{
"label": "Water",
"index": 0,
"probs": [0.42, 0.18, ...],
"classes": ["Water","Normal","Fire","Grass","Ghost","Bug","Electric","Poison","Psychic","Rock","Fighting"]
}

Model expectations
Input: 120×120 RGB, float32, normalized by 255.0

Output: probabilities over the configured label order (ensure label order during training matches inference).

Docker (optional)
docker build -t pokemon-type-app .

docker run -p 5000:5000 -e MODELPATH=models/pokemoncnn.h5 pokemon-type-app

Deployment (optional)
PaaS (e.g., Render/Heroku):

Include Procfile (web: gunicorn app:app --timeout 120) and runtime.txt (python-3.11.x).

Set MODELPATH env var to the model path.

Ensure the model file is available (commit to repo or fetch from storage on startup).

Data
data/pokemon.csv includes Name, Type1, Type2, and Evolution columns used for exploration and potential label mapping.

The training notebook demonstrates reading this file and preparing data; ensure any custom datasets match the model’s input pipeline.

Customization
Changing labels: update the label list consistently in training and serving.

Changing image size: retrain the model and update preprocess settings to match.

Using .keras format: adjust MODELPATH and load routine accordingly.

Troubleshooting
Model not found: verify MODELPATH and that models/pokemoncnn.h5 exists.

Mismatched labels: ensure the served label order matches the trained model’s classes.

Large files rejected: UI enforces ~5 MB; compress or resize images.

License
MIT (or adapt to project needs).

Acknowledgments
Thanks to the broader open-source community for tools that make rapid prototyping, training, and serving ML models straightforward.



