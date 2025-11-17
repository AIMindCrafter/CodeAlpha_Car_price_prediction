from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

APP = Flask(__name__, template_folder='templates', static_folder='static')

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'car_price_predictor_final_model.joblib')


def load_model(path=None):
    p = path or MODEL_PATH
    p = os.path.abspath(p)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Model file not found at {p}")
    return joblib.load(p)


MODEL = None
try:
    MODEL = load_model()
except Exception as e:
    MODEL = None
    print(f"Warning: failed to load model during startup: {e}")


@APP.route('/')
def index():
    return render_template('index.html', model_loaded=MODEL is not None)


@APP.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': MODEL is not None})


@APP.route('/predict', methods=['POST'])
def predict():
    # Accept either JSON body or form field named 'input_json'
    data = None
    if request.is_json:
        data = request.get_json()
    else:
        txt = request.form.get('input_json') or request.form.get('json')
        if txt:
            try:
                import json
                data = json.loads(txt)
            except Exception as e:
                return jsonify({'error': f'invalid json input: {e}'}), 400

    # also support form fields posted from the HTML form
    if data is None:
        # collect named fields from form
        form_keys = ['Selling_type', 'Fuel_Type', 'Present_Price', 'Car_Age', 'Driven_kms', 'Owner', 'Brand', 'Transmission']
        data = {}
        for k in form_keys:
            v = request.form.get(k)
            if v is not None:
                data[k] = v

    if not data:
        return jsonify({'error': 'No input provided. Send JSON or use the HTML form.'}), 400

    # If model exposes feature names, use them; otherwise fall back to common set
    if MODEL is not None and hasattr(MODEL, 'feature_names_in_'):
        expected = list(MODEL.feature_names_in_)
    else:
        expected = ['Selling_type', 'Fuel_Type', 'Present_Price', 'Car_Age', 'Driven_kms', 'Owner', 'Brand', 'Transmission']

    missing = set(expected) - set(data.keys())
    if missing:
        return jsonify({'error': f'columns are missing: {sorted(list(missing))}'}), 400

    # coerce common numeric fields
    for num_field in ['Present_Price', 'Car_Age', 'Driven_kms', 'Owner']:
        if num_field in data:
            try:
                # Owner may be integer
                if num_field == 'Owner':
                    data[num_field] = int(float(data[num_field]))
                else:
                    data[num_field] = float(data[num_field])
            except Exception:
                # leave as-is; model may handle or error will be returned
                pass

    try:
        df = pd.DataFrame([data], columns=expected)
        preds = MODEL.predict(df)
        # If prediction returns array-like
        val = preds[0].item() if hasattr(preds[0], 'item') else float(preds[0])
        return jsonify({'prediction': val})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    APP.run(host='0.0.0.0', port=5000)
