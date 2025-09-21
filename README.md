# Image-to-Pokemon-type
Pokemon Type Classifier: A CNN-based project that predicts a Pokémon’s primary type from an image and serves real-time results via a Flask app. Includes a training notebook, metadata CSV, and a web UI plus JSON API. Upload an image, get top label and sorted probabilities; model expects 120×120 RGB normalized inputs.



Project: Pokemon Type Classifier (CNN + Flask)

Overview

Train a CNN on 120x120 RGB sprites to classify primary type into 11 classes and serve predictions via Flask with a drag-and-drop UI.

The app exposes a web UI at GET / and a JSON API at POST /api/predict that returns label, index, and per-class probabilities.

Repo layout

notebooks/pokemon.ipynb: end-to-end data exploration, augmentation, CNN training, and model saving.

app.py: Flask server loading models/pokemoncnn.h5 and predicting on uploaded PNG/JPG/JPEG.

data/pokemon.csv: names, types, evolutions source.

model.meta.json: auxiliary metadata of label classes and tabular encodings.

models/: place pokemoncnn.h5 after training.

Setup

Python

Create venv and install requirements:

python -m venv venv && source venv/bin/activate # Windows: venv\Scripts\activate

pip install -r requirements.txt

Train and export model

Open notebooks/pokemon.ipynb, run through to train CNN, then save:

cnn.save("models/pokemoncnn.h5") # ensure models/ exists

Alternatively, place an existing pokemoncnn.h5 into models/.

Run locally

export MODELPATH=models/pokemoncnn.h5

python app.py

Visit http://localhost:5000 and upload an image.

API

POST /api/predict form-data file=<image>

Response: { "label": "Water", "index": 0, "probs": [..], "classes": ["Water","Normal",...]}

Docker

docker build -t pokemon-type-app .

docker run -p 5000:5000 -e MODELPATH=models/pokemoncnn.h5 pokemon-type-app

Deploy (Heroku/Render)

Include Procfile and runtime.txt; set MODELPATH to models/pokemoncnn.h5 and ensure the model is committed or uploaded to a persistent store.

Notes

The model expects 120x120 RGB and applies float32 normalization by 255.0.

Allowed file types: png, jpg, jpeg; max 5 MB enforced in UI.

LICENSE (MIT)
Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy of this software…

requirements.txt
flask==3.0.3

gunicorn==21.2.0

pillow==10.4.0

numpy==1.26.4

tensorflow==2.15.0

keras==2.15.0

runtime.txt
python-3.11.9

Procfile
web: gunicorn app:app --timeout 120

Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODELPATH=models/pokemoncnn.h5

EXPOSE 5000

CMD ["python", "app.py"]

.dockerignore
venv

pycache

.git

models/*.keras

dist

build

app.py (use attached version verbatim)
Already includes:

Settings: IMGSIZE (120,120), NORMALIZE255=True, MODELPATH env with default pokemoncnn.h5, allowed extensions, IDX2LABEL.

Model load: tf.keras.models.load_model(MODELPATH).

Routes:

GET /: renders upload UI.

POST /predict: legacy compatibility.

POST /api/predict: JSON response with label/index/probs/classes.

Frontend: drag-and-drop, preview, client-side checks, results table sorted by probability.

Action: place this file at repo root as app.py.

notebooks/pokemon.ipynb
Place the provided notebook in notebooks/pokemon.ipynb.

It:

Loads data/pokemon.csv.

Builds augmented dataset with ImageDataGenerator.

Defines a CNN with Conv2D/MaxPool/Dropout/Flatten/Dense softmax over 11 classes.

Trains with accuracy improving across epochs; then saves model to pokemoncnn.h5.

Ensure the final cell saves to models/pokemoncnn.h5 relative to repo root.

data/pokemon.csv
Place the provided CSV in data/pokemon.csv; notebook expects this path.

model.meta.json
Place at repo root as provided; optional for app, useful for aux tooling or future features.

src/train/prepare_data.py (optional utility)
Read data/pokemon.csv, validate columns [Name, Type1, Type2, Evolution], and print label distribution of Type1 to help balance decisions.

Example content:

Reads CSV, prints value_counts(Type1), checks images if integrated.

src/train/export_model.py (optional utility)
Loads a trained Keras model path and saves to models/pokemoncnn.h5 or .keras.

Environment variables
MODELPATH: path to .h5 (default models/pokemoncnn.h5).

PORT: Flask port (default 5000).

How to use
Copy the files into a fresh folder matching the tree above.

Commit and push to GitHub:

git init && git add . && git commit -m "Initial commit: Pokemon Type Classifier" && git branch -M main && git remote add origin <repo-url> && git push -u origin main

Train and save model via notebook, then run app locally or with Docker.

Notes and caveats
Ensure the saved model name matches MODELPATH; the app defaults to pokemoncnn.h5 in models/.

The notebook’s final cells include guidance to save and write app.py; in this repo, app.py is already present—only execute the save step.

If converting to the newer Keras format, update MODELPATH and load routine accordingly (e.g., mymodel.keras). The current app expects .h5.

