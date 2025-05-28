import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load the trained model
model = load_model('house_price.keras', compile=False)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

# Load the saved scaler — recommended instead of fitting again
scaler =MinMaxScaler()  # <-- You should save it during training using joblib.dump()

def predict_from_dict(input_dict):
    df = pd.DataFrame([input_dict])
    data_scaled = scaler.fit_transform(df)
    predictions = model.predict(data_scaled)
    y_pred = np.expm1(predictions)  # Reverse log1p transform if you used log1p(y) during training
    return float(y_pred[0][0])

def predict_from_csv(x_csv_path: str, y_csv_path: str = None):
    data = pd.read_csv(x_csv_path)

    if 'target' in data.columns:
        data = data.drop(columns=['target'])

    data_scaled = scaler.transform(data)
    predictions = model.predict(data_scaled)
    y_pred = np.expm1(predictions)

    if y_csv_path:
        y_true = pd.read_csv(y_csv_path)
        df = pd.concat((y_true, pd.Series(y_pred.flatten(), name="Predicted")), axis=1)
        return df
    else:
        return pd.Series(y_pred.flatten(), name="Predicted")
