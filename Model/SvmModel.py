import os
from joblib import load

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'svc_model.joblib')
svc = load(model_path)

dicLabeled = ['Accept','Back','Idle','Next','Previous']

def predict_svc(landmarks):
    prediction = svc.predict([landmarks])[0]
    return dicLabeled[prediction]