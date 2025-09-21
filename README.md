# Pokémon Type Classifier 🎮🔮

A Flask web app that classifies Pokémon images into their respective **types** using a trained **CNN model**.  
Upload an image, and the app predicts the Pokémon’s type with probability scores.

---

## 📂 Project Files
- `app.py` → Flask app (frontend + backend API)
- `model.keras` → Trained TensorFlow/Keras model
- `pokemon.csv` → Dataset reference
- `README.md` → Project documentation

---

## 🚀 Features
- Upload Pokémon images (PNG, JPG, JPEG)
- Predicts among **11 Pokémon types**:
- Clean drag-and-drop web interface
- REST API endpoint for programmatic predictions

---

## 🛠️ Setup
```bash
git clone <repo_url>
cd pokemon-type-classifier
python -m venv venv
source venv/bin/activate   # (Windows: venv\Scripts\activate)
pip install flask tensorflow pillow numpy

python app.py

