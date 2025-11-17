# Car Price Predictor — Profesional mloop

This folder contains a saved car price prediction model and a tiny Flask app to serve it locally.

Files added:
- `serve_car.py` — Flask app that loads `car_price_predictor_final_model.joblib` and provides `/`, `/predict`, and `/health`.
- `templates/index.html` — Minimal HTML UI titled **Profesional mloop** where you can paste a JSON object of features and get predictions.
- `requirements_flask.txt` — Python packages needed to run the server.

Quick start (from this folder):

1. Create & activate a venv

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies

```bash
python -m pip install -r requirements_flask.txt
```

3. Run the app

```bash
python3 serve_car.py
```

Open http://localhost:5000/ to use the UI. The UI expects a JSON object with the feature names used when training the model. If you are unsure of the exact feature names, inspect the training notebook `car_price_prediction.ipynb`.

API examples

Health:

```bash
curl http://localhost:5000/health
# -> {"status":"ok","model_loaded":true}
```

Predict (example):

```bash
curl -X POST http://localhost:5000/predict -H 'Content-Type: application/json' -d '{"feature_1":1.0, "feature_2":2.0}'
# -> {"prediction": 12345.67}
```

If the model fails to load, ensure `car_price_predictor_final_model.joblib` is present in this directory. Tell me if you want me to run the server here, or if you'd like the app changed to use a different port or to provide a form with specific named inputs.
