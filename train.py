import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

print("Loading dataset...")
df = pd.read_csv('eyewear_synthetic_orders.csv', keep_default_na=False)

# Define features and target
features = ['lens_type', 'lens_index', 'sph_power', 'coating', 'in_house_stock']
X = df[features].copy()
y = df['sla_breached']

# Encode categorical features
label_encoders = {}
for col in ['lens_type', 'coating']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Train a lightweight Random Forest Model
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X, y)

# Save the model and encoders for the FastAPI server
print("Saving model and encoders...")
joblib.dump(model, 'sla_predictor_model.joblib')
joblib.dump(label_encoders, 'label_encoders.joblib')
print("Success! Model saved as 'sla_predictor_model.joblib'")