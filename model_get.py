import joblib

def get_pymodel():
    model_path = 'static/model/model.pkl'
    model = joblib.load(model_path)
    return model